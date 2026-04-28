import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pyrallis
import torch
import wandb
import yaml

import common_utils
from common_utils import ibrl_utils as utils
from bc.diffusion_policy_adapter import DiffusionPolicyAdapter
from env.manifeel_bulb_wrapper import ManiFeelBulbVecEnv
from rl.q_agent import QAgent, QAgentConfig
from rl.vec_replay import VecReplayBuffer


def _default_q_agent_cfg() -> QAgentConfig:
    cfg = QAgentConfig()
    cfg.use_prop = 1
    cfg.act_method = "ibrl"
    cfg.bootstrap_method = "ibrl"
    return cfg


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1

    diffusion_policy_root: str = "../Diffusion Policy/diffusion_policy"
    manifeel_root: str = "../Diffusion Policy/manifeel"
    isaacgym_envs_path: str = "../Diffusion Policy/manifeel-isaacgymenvs"
    dp_checkpoint: str = ""

    num_train_envs: int = 8
    num_eval_envs: int = 4
    num_eval_episode: int = 20
    num_eval_videos: int = 2
    episode_length: int = 800
    image_size: int = 256
    rl_camera: str = "wrist"
    external_camera: str = "client"
    light_factor: float = 1.0
    env_reward_scale: float = 1.0
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    graphics_device_id: int = 0
    headless: int = 1
    force_render: int = 1
    eval_force_render: int = 1
    video_fps: int = 10

    q_agent: QAgentConfig = field(default_factory=_default_q_agent_cfg)
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 1
    num_warm_up_episode: int = 20
    num_train_step: int = 200000
    log_per_step: int = 10000
    add_bc_loss: int = 0

    save_dir: str = "exps/rl/manifeel_bulb/run1"
    use_wb: int = 0

    def __post_init__(self):
        self.q_agent.use_prop = 1
        if self.q_agent.act_method == "rl":
            self.q_agent.bootstrap_method = "rl"

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:
    def __init__(self, cfg: MainConfig, from_main: bool = True):
        self.cfg = cfg
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

        if from_main:
            common_utils.set_all_seeds(cfg.seed)
            sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)
            pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore[arg-type]
            print(common_utils.wrap_ruler("config"))
            with open(cfg.cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))
        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self.global_iter = 0

        self.bc_policy = DiffusionPolicyAdapter(
            checkpoint_path=cfg.dp_checkpoint,
            diffusion_policy_root=cfg.diffusion_policy_root,
            manifeel_root=cfg.manifeel_root,
            device=cfg.rl_device,
        )
        self.n_obs_steps = self.bc_policy.n_obs_steps

        self.train_env = self._make_env(cfg.num_train_envs, cfg.force_render, seed_offset=0)
        self.eval_env = self._make_env(cfg.num_eval_envs, cfg.eval_force_render, seed_offset=1000)

        self.agent = QAgent(
            False,
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            cfg.rl_camera,
            cfg.q_agent,
        )
        self.ref_agent = None
        if cfg.add_bc_loss:
            self.ref_agent = copy.deepcopy(self.agent)
            self.ref_agent.cfg.act_method = "rl"
        self.agent.add_bc_policy(self.bc_policy)

        self.replay = VecReplayBuffer(
            num_envs=self.train_env.num_envs,
            nstep=cfg.nstep,
            gamma=cfg.discount,
            max_episode_length=cfg.episode_length,
            replay_size=cfg.replay_buffer_size,
        )

    def _make_env(self, num_envs: int, force_render: int, seed_offset: int) -> ManiFeelBulbVecEnv:
        return ManiFeelBulbVecEnv(
            manifeel_root=self.cfg.manifeel_root,
            isaacgym_envs_path=self.cfg.isaacgym_envs_path,
            num_envs=num_envs,
            sim_device=self.cfg.sim_device,
            rl_device=self.cfg.rl_device,
            graphics_device_id=self.cfg.graphics_device_id,
            headless=bool(self.cfg.headless),
            seed=self.cfg.seed + seed_offset,
            n_obs_steps=self.n_obs_steps,
            max_episode_length=self.cfg.episode_length,
            env_reward_scale=self.cfg.env_reward_scale,
            rl_camera=self.cfg.rl_camera,
            external_camera=self.cfg.external_camera,
            image_hw=(self.cfg.image_size, self.cfg.image_size),
            light_factor=self.cfg.light_factor,
            force_render=bool(force_render),
        )

    def _pack_replay_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "bc_wrist": obs["bc_wrist"].to(dtype=torch.uint8),
            "bc_state": obs["bc_state"].to(dtype=torch.float32),
        }

    def _inflate_obs(self, obs: Dict[str, torch.Tensor]) -> None:
        if "wrist" not in obs:
            obs["wrist"] = obs["bc_wrist"][:, -1]
        if "prop" not in obs:
            obs["prop"] = obs["bc_state"][:, -1]
        if "state" not in obs:
            obs["state"] = obs["bc_state"][:, -1]

    def _inflate_batch(self, batch) -> None:
        self._inflate_obs(batch.obs)
        self._inflate_obs(batch.next_obs)

    def warm_up(self) -> None:
        obs = self.train_env.reset()
        self.replay.reset_current_episodes()
        self.replay.new_episodes(self._pack_replay_obs(obs))

        while self.replay.size() < self.cfg.num_warm_up_episode:
            with torch.no_grad(), utils.eval_mode(self.bc_policy):
                actions = self.bc_policy.act(obs, cpu=False)
            next_obs, rewards, dones, successes = self.train_env.step(actions)
            self.replay.add_step(
                self._pack_replay_obs(next_obs),
                actions,
                rewards,
                dones,
                successes,
            )
            obs = next_obs

        print(f"Warm up done. #episodes: {self.replay.size()}")

    def _record_eval_videos(self, videos: List[np.ndarray]) -> Dict[str, "wandb.Video"]:
        payload: Dict[str, "wandb.Video"] = {}
        for idx, video in enumerate(videos):
            payload[f"test/video_{idx}"] = wandb.Video(
                video.transpose(0, 3, 1, 2),
                fps=self.cfg.video_fps,
                format="mp4",
            )
        return payload

    def eval(self) -> Dict[str, object]:
        scores: List[float] = []
        episode_rewards: List[float] = []
        obs = self.eval_env.reset()
        running_reward = torch.zeros(self.eval_env.num_envs, device=self.eval_env.device)

        num_video_envs = min(self.cfg.num_eval_videos, self.eval_env.num_envs)
        video_frames: List[List[np.ndarray]] = [[] for _ in range(num_video_envs)]
        finished_videos: List[np.ndarray] = []
        initial_frames = self.eval_env.render()
        for env_id in range(num_video_envs):
            video_frames[env_id].append(initial_frames[env_id])

        with torch.no_grad(), utils.eval_mode(self.agent):
            while len(scores) < self.cfg.num_eval_episode:
                actions = self.agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                obs, rewards, dones, successes = self.eval_env.step(actions)
                running_reward += rewards

                frames = self.eval_env.render()
                for env_id in range(num_video_envs):
                    video_frames[env_id].append(frames[env_id])

                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                for env_id in done_ids.tolist():
                    if len(scores) >= self.cfg.num_eval_episode:
                        break
                    scores.append(float(successes[env_id].item()))
                    episode_rewards.append(float(running_reward[env_id].item()))
                    running_reward[env_id] = 0.0
                    if env_id < num_video_envs and len(finished_videos) < self.cfg.num_eval_videos:
                        finished_videos.append(np.stack(video_frames[env_id], axis=0).astype(np.uint8))
                        video_frames[env_id] = [frames[env_id]]

        return {
            "scores": scores,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "videos": finished_videos,
        }

    def rl_train(self, stat: common_utils.MultiCounter) -> None:
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for idx in range(self.cfg.num_critic_update):
            batch = self.replay.sample(self.cfg.batch_size, self.cfg.rl_device)
            self._inflate_batch(batch)
            update_actor = idx == self.cfg.num_critic_update - 1
            bc_batch = None
            ref_agent = None
            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample(self.cfg.batch_size, self.cfg.rl_device)
                self._inflate_batch(bc_batch)
                ref_agent = self.ref_agent
                assert ref_agent is not None
            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, ref_agent)
            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ) -> None:
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / max(elapsed_time, 1e-6))
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["other/replay"].append(self.replay.size())
        stat["score/num_success"].append(self.replay.num_success)

        with stopwatch.time("eval"):
            eval_metrics = self.eval()
            stat["test/mean_score"].append(eval_metrics["mean_score"])
            stat["test/mean_reward"].append(eval_metrics["mean_reward"])

            if self.cfg.use_wb:
                video_payload = self._record_eval_videos(eval_metrics["videos"])
                if video_payload:
                    wandb.log(video_payload, step=self.global_step)

        saved = saver.save(self.agent.state_dict(), eval_metrics["mean_score"], save_latest=True)
        stat.summary(self.global_step, reset=True)
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def train(self) -> None:
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        self.warm_up()
        stopwatch = common_utils.Stopwatch()

        obs = self.train_env.reset()
        self.replay.reset_current_episodes()
        self.replay.new_episodes(self._pack_replay_obs(obs))

        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                actions = self.agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
                stat["data/stddev"].append(stddev)

            with stopwatch.time("env step"):
                next_obs, rewards, dones, successes = self.train_env.step(actions)

            with stopwatch.time("add"):
                self.replay.add_step(
                    self._pack_replay_obs(next_obs),
                    actions,
                    rewards,
                    dones,
                    successes,
                )
                self.global_iter += 1
                self.global_step += self.train_env.num_envs

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self.global_episode += int(done_ids.numel())
                stat["score/train_score"].append(
                    float(self.train_env.last_done_successes.float().sum().item()),
                    int(done_ids.numel()),
                )
                stat["data/episode_len"].append(
                    float(self.train_env.last_done_steps.float().sum().item()),
                    int(done_ids.numel()),
                )
                stat["data/episode_reward"].append(
                    float(self.train_env.last_done_rewards.sum().item()),
                    int(done_ids.numel()),
                )

            obs = next_obs

            if self.global_iter % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1

            if self.global_step % self.cfg.log_per_step < self.train_env.num_envs:
                self.log_and_save(stopwatch, stat, saver)


def main() -> None:
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore[arg-type]
    workspace = Workspace(cfg)
    workspace.train()
    if cfg.use_wb:
        wandb.finish()


if __name__ == "__main__":
    from rich.traceback import install

    install()
    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    main()
