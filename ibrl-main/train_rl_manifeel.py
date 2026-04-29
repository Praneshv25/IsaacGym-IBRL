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


class _GpuBatch:
    def __init__(self, obs, next_obs, action, reward, bootstrap):
        self.obs = obs
        self.next_obs = next_obs
        self.action = action
        self.reward = reward
        self.bootstrap = bootstrap


class _GpuReplayBuffer:
    def __init__(self, capacity: int, gamma: float, device: str):
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self._obs = None
        self._next_obs = None
        self._action = None
        self._reward = None
        self._bootstrap = None
        self._ptr = 0
        self._size = 0
        self.num_episode = 0
        self.num_success = 0

    def _ensure_storage(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> None:
        if self._obs is not None:
            return
        self._obs = {
            k: torch.empty((self.capacity, *v.shape[1:]), dtype=v.dtype, device=self.device)
            for k, v in obs.items()
        }
        self._next_obs = {
            k: torch.empty((self.capacity, *v.shape[1:]), dtype=v.dtype, device=self.device)
            for k, v in obs.items()
        }
        self._action = torch.empty((self.capacity, *action.shape[1:]), dtype=action.dtype, device=self.device)
        self._reward = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self._bootstrap = torch.empty(self.capacity, dtype=torch.float32, device=self.device)

    def add_step(
        self,
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        successes: torch.Tensor,
    ) -> None:
        self._ensure_storage(obs, actions)
        n = actions.shape[0]
        assert self._obs is not None and self._next_obs is not None
        assert self._action is not None and self._reward is not None and self._bootstrap is not None
        idx = (torch.arange(n, device=self.device) + self._ptr) % self.capacity
        for k in self._obs:
            self._obs[k][idx] = obs[k].detach().clone()
            self._next_obs[k][idx] = next_obs[k].detach().clone()
        self._action[idx] = actions.detach().clone()
        self._reward[idx] = rewards.detach().float()
        self._bootstrap[idx] = (~dones).detach().float() * self.gamma
        self._ptr = int((self._ptr + n) % self.capacity)
        self._size = min(self.capacity, self._size + n)
        self.num_episode += int(dones.sum().item())
        self.num_success += int((dones & successes).sum().item())

    def sample(self, batchsize: int, device: str):
        assert self._size > 0
        sample_device = torch.device(device)
        idx = torch.randint(0, self._size, (batchsize,), device=self.device)
        assert self._obs is not None and self._next_obs is not None
        assert self._action is not None and self._reward is not None and self._bootstrap is not None
        obs = {k: v[idx].to(sample_device) for k, v in self._obs.items()}
        next_obs = {k: v[idx].to(sample_device) for k, v in self._next_obs.items()}
        action = {"action": self._action[idx].to(sample_device)}
        reward = self._reward[idx].to(sample_device)
        bootstrap = self._bootstrap[idx].to(sample_device)
        return _GpuBatch(obs, next_obs, action, reward, bootstrap)

    def size(self) -> int:
        return self._size


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

        self.replay = _GpuReplayBuffer(
            capacity=cfg.replay_buffer_size * self.train_env.num_envs * max(cfg.episode_length, 1),
            gamma=cfg.discount,
            device=cfg.rl_device,
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
        wrist = obs[self.cfg.rl_camera]
        prop = obs["prop"]
        state = obs["state"]
        print(
            "[ibrl] pack input "
            f"{self.cfg.rl_camera}:shape={tuple(wrist.shape)} dtype={wrist.dtype} device={wrist.device}; "
            f"prop:shape={tuple(prop.shape)} dtype={prop.dtype} device={prop.device}; "
            f"state:shape={tuple(state.shape)} dtype={state.dtype} device={state.device}",
            flush=True,
        )
        wrist_out = wrist.detach().clone().to(dtype=torch.float32).contiguous()
        print(
            f"[ibrl] pack {self.cfg.rl_camera} ok shape={tuple(wrist_out.shape)} dtype={wrist_out.dtype}",
            flush=True,
        )
        prop_out = prop.detach().clone().to(dtype=torch.float32).contiguous()
        print(f"[ibrl] pack prop ok shape={tuple(prop_out.shape)} dtype={prop_out.dtype}", flush=True)
        state_out = state.detach().clone().to(dtype=torch.float32).contiguous()
        print(f"[ibrl] pack state ok shape={tuple(state_out.shape)} dtype={state_out.dtype}", flush=True)

        return {
            self.cfg.rl_camera: wrist_out,
            "prop": prop_out,
            "state": state_out,
        }

    def _inflate_obs(self, obs: Dict[str, torch.Tensor]) -> None:
        if "bc_wrist" in obs and obs["bc_wrist"].dim() == 4:
            b, tc, h, w = obs["bc_wrist"].shape
            obs["bc_wrist"] = obs["bc_wrist"].reshape(b, self.n_obs_steps, 3, h, w)
        if "bc_state" in obs and obs["bc_state"].dim() == 2:
            b, ts = obs["bc_state"].shape
            obs["bc_state"] = obs["bc_state"].reshape(b, self.n_obs_steps, self.train_env.prop_shape[0])
        if self.cfg.rl_camera not in obs and "bc_wrist" in obs:
            obs[self.cfg.rl_camera] = obs["bc_wrist"][:, -1].float()
        if "prop" not in obs and "bc_state" in obs:
            obs["prop"] = obs["bc_state"][:, -1]
        if "state" not in obs and "bc_state" in obs:
            obs["state"] = obs["bc_state"][:, -1]

    def _inflate_batch(self, batch) -> None:
        self._inflate_obs(batch.obs)
        self._inflate_obs(batch.next_obs)

    def warm_up(self) -> Dict[str, torch.Tensor]:
        _dbg("[ibrl] warmup reset")
        obs = self.train_env.reset()
        _dbg("[ibrl] warmup reset ok")

        while self.replay.num_episode < self.cfg.num_warm_up_episode:
            _dbg("[ibrl] warmup bc act")
            with torch.no_grad(), utils.eval_mode(self.bc_policy):
                actions = self.bc_policy.act(obs, cpu=False)
            _dbg("[ibrl] warmup bc act ok")
            _dbg("[ibrl] warmup step")
            next_obs, rewards, dones, successes = self.train_env.step(actions)
            _dbg("[ibrl] warmup step ok")
            print(
                f"[ibrl] warmup step stats reward_shape={tuple(rewards.shape)} "
                f"done_sum={int(dones.sum().item())} success_sum={int(successes.sum().item())}",
                flush=True,
            )
            _dbg("[ibrl] warmup add_step")
            self.replay.add_step(
                self._pack_replay_obs(obs),
                self._pack_replay_obs(next_obs),
                actions,
                rewards,
                dones,
                successes,
            )
            _dbg("[ibrl] warmup add_step ok")
            obs = next_obs

        print(f"Warm up done. #episodes: {self.replay.size()}")
        return obs

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
        _dbg("[ibrl] rl_train start")
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for idx in range(self.cfg.num_critic_update):
            print(f"[ibrl] rl_train sample idx={idx}", flush=True)
            batch = self.replay.sample(self.cfg.batch_size, self.cfg.rl_device)
            _dbg("[ibrl] rl_train sample ok")
            self._inflate_batch(batch)
            _dbg("[ibrl] rl_train inflate batch ok")
            update_actor = idx == self.cfg.num_critic_update - 1
            bc_batch = None
            ref_agent = None
            if update_actor and self.cfg.add_bc_loss:
                _dbg("[ibrl] rl_train bc sample")
                bc_batch = self.replay.sample(self.cfg.batch_size, self.cfg.rl_device)
                self._inflate_batch(bc_batch)
                _dbg("[ibrl] rl_train bc sample ok")
                ref_agent = self.ref_agent
                assert ref_agent is not None
            _dbg("[ibrl] rl_train agent.update")
            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, ref_agent)
            _dbg("[ibrl] rl_train agent.update ok")
            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())
        _dbg("[ibrl] rl_train done")

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

        obs = self.warm_up()
        _dbg("[ibrl] warmup returned obs")
        stopwatch = common_utils.Stopwatch()
        running_lengths = torch.zeros(self.train_env.num_envs, dtype=torch.long, device=self.train_env.device)
        running_rewards = torch.zeros(self.train_env.num_envs, dtype=torch.float32, device=self.train_env.device)
        _dbg("[ibrl] train loop start")

        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                _dbg("[ibrl] train act")
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                actions = self.agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
                stat["data/stddev"].append(stddev)
                _dbg("[ibrl] train act ok")

            with stopwatch.time("env step"):
                _dbg("[ibrl] train env step")
                next_obs, rewards, dones, successes = self.train_env.step(actions)
                _dbg("[ibrl] train env step ok")
                print(
                    f"[ibrl] train step stats reward_shape={tuple(rewards.shape)} "
                    f"done_sum={int(dones.sum().item())} success_sum={int(successes.sum().item())}",
                    flush=True,
                )

            running_lengths += 1
            running_rewards += rewards

            with stopwatch.time("add"):
                _dbg("[ibrl] train add_step")
                self.replay.add_step(
                    self._pack_replay_obs(obs),
                    self._pack_replay_obs(next_obs),
                    actions,
                    rewards,
                    dones,
                    successes,
                )
                _dbg("[ibrl] train add_step ok")
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
                    _dbg("[ibrl] train update block")
                    self.rl_train(stat)
                    self.train_step += 1
                    _dbg("[ibrl] train update block ok")

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
