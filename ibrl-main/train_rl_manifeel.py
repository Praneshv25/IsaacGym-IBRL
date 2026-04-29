import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import isaacgym  # noqa: F401
import numpy as np
import pyrallis
import torch
import wandb
import yaml
import hydra
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import common_utils
from bc.diffusion_policy_adapter import DiffusionPolicyAdapter
from common_utils import ibrl_utils as utils
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
    isaac_camera: str = "wrist"
    external_camera: str = "client"
    env_reward_scale: float = 1.0
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    graphics_device_id: int = 0
    headless: int = 1
    force_render: int = 0
    eval_force_render: int = 0
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


class _QAgentEvalPolicy:
    def __init__(self, agent: QAgent, device: str, rl_camera: str) -> None:
        self.agent = agent
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.rl_camera = rl_camera

    def reset(self) -> None:
        return

    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wrist = obs[self.rl_camera].to(self.device).float()
        if float(wrist.max()) <= 1.01:
            wrist = wrist * 255.0
        state = obs["state"].to(self.device).float()

        agent_obs = {
            self.rl_camera: wrist[:, -1],
            "prop": state[:, -1],
            "state": state[:, -1],
            "bc_wrist": wrist,
            "bc_state": state,
        }
        action = self.agent.act(agent_obs, eval_mode=True, stddev=0.0, cpu=False)
        return {"action": action.unsqueeze(1)}


def _dbg(msg: str) -> None:
    print(msg, flush=True)


class Workspace:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

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

        if cfg.isaac_camera != cfg.rl_camera:
            raise ValueError(
                f"isaac_camera={cfg.isaac_camera!r} is not supported in the restart path. "
                f"Use rl_camera={cfg.rl_camera!r} so train and eval both see the same ManiFeel view."
            )

        _dbg("[ibrl] creating train env")
        self.train_env = self._make_env(cfg.num_train_envs, cfg.force_render, seed_offset=0)
        _dbg("[ibrl] train env ready")
        self.eval_runner = None
        _dbg("[ibrl] eval runner deferred")

        self.agent = QAgent(
            False,
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            cfg.rl_camera,
            cfg.q_agent,
        )
        self.agent.add_bc_policy(self.bc_policy)

        self.ref_agent = None
        if cfg.add_bc_loss:
            self.ref_agent = copy.deepcopy(self.agent)
            self.ref_agent.cfg.act_method = "rl"

        self.eval_policy = _QAgentEvalPolicy(
            agent=self.agent,
            device=cfg.rl_device,
            rl_camera=cfg.rl_camera,
        )

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
            env_reward_scale=self.cfg.env_reward_scale,
            rl_camera=self.cfg.rl_camera,
            isaac_camera=self.cfg.isaac_camera,
            image_hw=(self.cfg.image_size, self.cfg.image_size),
            max_episode_length=self.cfg.episode_length,
            force_render=bool(force_render),
        )

    def _make_eval_runner(self):
        abs_root = os.path.abspath(self.cfg.manifeel_root)
        if abs_root not in sys.path:
            sys.path.insert(0, abs_root)

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        config_dir = os.path.join(abs_root, "manifeel", "config")
        output_dir = str(Path(self.work_dir).joinpath("eval_runner"))
        os.makedirs(output_dir, exist_ok=True)
        overrides = [
            "task=vision_wrist",
            "isaacgym_cfg_name=isaacgym_config_bulb.yaml",
            f"training.seed={self.cfg.seed}",
            f"n_obs_steps={self.n_obs_steps}",
            "n_action_steps=1",
            f"task.env_runner.n_test={self.cfg.num_eval_envs}",
            f"task.env_runner.n_test_vis={self.cfg.num_eval_videos}",
            f"task.env_runner.max_steps={self.cfg.episode_length}",
        ]

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = hydra.compose(config_name="train_diffusion_workspace", overrides=overrides)
            cfg.training.device = self.cfg.rl_device
            cfg.task.env_runner.test_start_seed = self.cfg.seed + 100000
            runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=output_dir,
            )
        return runner

    def _pack_replay_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bc_wrist = obs["bc_wrist"].reshape(
            obs["bc_wrist"].shape[0],
            obs["bc_wrist"].shape[1] * obs["bc_wrist"].shape[2],
            obs["bc_wrist"].shape[3],
            obs["bc_wrist"].shape[4],
        )
        bc_state = obs["bc_state"].reshape(
            obs["bc_state"].shape[0],
            obs["bc_state"].shape[1] * obs["bc_state"].shape[2],
        )
        return {
            self.cfg.rl_camera: obs[self.cfg.rl_camera].to(dtype=torch.uint8).contiguous(),
            "prop": obs["prop"].to(dtype=torch.float32).contiguous(),
            "state": obs["state"].to(dtype=torch.float32).contiguous(),
            "bc_wrist": bc_wrist.to(dtype=torch.uint8).contiguous(),
            "bc_state": bc_state.to(dtype=torch.float32).contiguous(),
        }

    def _inflate_obs(self, obs: Dict[str, torch.Tensor]) -> None:
        if obs["bc_wrist"].dim() == 4:
            b, tc, h, w = obs["bc_wrist"].shape
            obs["bc_wrist"] = obs["bc_wrist"].reshape(b, self.n_obs_steps, 3, h, w)
        if obs["bc_state"].dim() == 2:
            b, ts = obs["bc_state"].shape
            obs["bc_state"] = obs["bc_state"].reshape(b, self.n_obs_steps, self.train_env.prop_shape[0])
        if self.cfg.rl_camera not in obs:
            obs[self.cfg.rl_camera] = obs["bc_wrist"][:, -1].float()
        if "prop" not in obs:
            obs["prop"] = obs["bc_state"][:, -1]
        if "state" not in obs:
            obs["state"] = obs["bc_state"][:, -1]

    def _inflate_batch(self, batch) -> None:
        self._inflate_obs(batch.obs)
        self._inflate_obs(batch.next_obs)

    def warm_up(self) -> None:
        _dbg("[ibrl] warmup reset")
        obs = self.train_env.reset()
        _dbg("[ibrl] warmup reset ok")
        self.replay.reset_current_episodes()
        _dbg("[ibrl] warmup new_episodes")
        self.replay.new_episodes(self._pack_replay_obs(obs))
        _dbg("[ibrl] warmup new_episodes ok")

        while self.replay.size() < self.cfg.num_warm_up_episode:
            _dbg("[ibrl] warmup bc act")
            with torch.no_grad(), utils.eval_mode(self.bc_policy):
                actions = self.bc_policy.act(obs, cpu=False)
            _dbg("[ibrl] warmup bc act ok")
            _dbg("[ibrl] warmup step")
            next_obs, rewards, dones, successes = self.train_env.step(actions)
            _dbg("[ibrl] warmup step ok")
            self.replay.add_step(
                self._pack_replay_obs(next_obs),
                actions,
                rewards,
                dones,
                successes,
            )
            obs = next_obs

        print(f"Warm up done. #episodes: {self.replay.size()}")

    def eval(self) -> Dict[str, object]:
        if self.eval_runner is None:
            _dbg("[ibrl] creating eval runner")
            self.eval_runner = self._make_eval_runner()
            _dbg("[ibrl] eval runner ready")
        with torch.no_grad(), utils.eval_mode(self.agent):
            self.eval_policy.reset()
            log_data = self.eval_runner.run(self.eval_policy)

        videos = [value for key, value in log_data.items() if key.startswith("test/sim_video_")]
        return {
            "mean_score": float(log_data.get("test/mean_score", 0.0)),
            "videos": videos,
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

            if self.cfg.use_wb:
                video_payload = {
                    f"test/video_{idx}": video
                    for idx, video in enumerate(eval_metrics["videos"])
                }
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

        _dbg("[ibrl] train reset")
        obs = self.train_env.reset()
        _dbg("[ibrl] train reset ok")
        self.replay.reset_current_episodes()
        self.replay.new_episodes(self._pack_replay_obs(obs))
        running_lengths = torch.zeros(self.train_env.num_envs, dtype=torch.long, device=self.train_env.device)
        running_rewards = torch.zeros(self.train_env.num_envs, dtype=torch.float32, device=self.train_env.device)

        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                actions = self.agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
                stat["data/stddev"].append(stddev)

            with stopwatch.time("env step"):
                next_obs, rewards, dones, successes = self.train_env.step(actions)

            running_lengths += 1
            running_rewards += rewards

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
                    float(successes[done_ids].float().sum().item()),
                    int(done_ids.numel()),
                )
                stat["data/episode_len"].append(
                    float(running_lengths[done_ids].float().sum().item()),
                    int(done_ids.numel()),
                )
                stat["data/episode_reward"].append(
                    float(running_rewards[done_ids].sum().item()),
                    int(done_ids.numel()),
                )
                running_lengths[done_ids] = 0
                running_rewards[done_ids] = 0.0

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
