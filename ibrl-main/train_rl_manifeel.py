import copy
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
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
from tqdm.auto import tqdm

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
    eval_max_steps: int = 0
    num_eval_videos: int = 2
    episode_length: int = 800
    image_size: int = 256
    rl_camera: str = "wrist"
    isaac_camera: str = "wrist"
    external_camera: str = "client"
    env_reward_scale: float = 1.0
    reward_mode: str = "dense"
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
    gpu_replay_capacity: int = 20000
    demo_replay_capacity: int = 20000
    batch_size: int = 256
    demo_batch_ratio: float = 0.5
    num_critic_update: int = 1
    update_freq: int = 1
    num_warm_up_episode: int = 20
    num_train_step: int = 200000
    log_per_step: int = 10000
    add_bc_loss: int = 1

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

        self.total_envs = cfg.num_train_envs + cfg.num_eval_envs
        self.train_env = self._make_env(self.total_envs, cfg.force_render, seed_offset=0)
        self.train_idx = torch.arange(cfg.num_train_envs, device=self.train_env.device)
        self.eval_idx = torch.arange(cfg.num_train_envs, self.total_envs, device=self.train_env.device)
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

        self.replay = _GpuReplayBuffer(
            capacity=cfg.gpu_replay_capacity,
            gamma=cfg.discount,
            device=cfg.rl_device,
        )
        self.demo_replay = _GpuReplayBuffer(
            capacity=cfg.demo_replay_capacity,
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
            reward_mode=self.cfg.reward_mode,
            rl_camera=self.cfg.rl_camera,
            isaac_camera=self.cfg.isaac_camera,
            external_camera=self.cfg.external_camera,
            image_hw=(self.cfg.image_size, self.cfg.image_size),
            max_episode_length=self.cfg.episode_length,
            force_render=bool(force_render),
        )

    def _slice_obs(self, obs: Dict[str, torch.Tensor], env_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v.index_select(0, env_idx) for k, v in obs.items()}

    def _compose_full_action(
        self,
        *,
        train_actions: Optional[torch.Tensor] = None,
        eval_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        full = torch.zeros(
            (self.total_envs, self.train_env.action_dim),
            dtype=torch.float32,
            device=self.train_env.device,
        )
        if train_actions is not None:
            full[self.train_idx] = train_actions
        if eval_actions is not None and self.eval_idx.numel() > 0:
            full[self.eval_idx] = eval_actions
        return full

    def _obs_to_video_frame(self, obs: Dict[str, torch.Tensor], env_idx: int) -> np.ndarray:
        camera_key = self.cfg.external_camera if self.cfg.external_camera in obs else self.cfg.rl_camera
        frame = obs[camera_key][env_idx].detach().float()
        if float(frame.max()) <= 1.01:
            frame = frame * 255.0
        frame = frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return frame

    def _pack_replay_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wrist = obs[self.cfg.rl_camera]
        prop = obs["prop"]
        state = obs["state"]
        wrist_out = wrist.detach().clone().to(dtype=torch.uint8).contiguous()
        prop_out = prop.detach().clone().to(dtype=torch.float32).contiguous()
        state_out = state.detach().clone().to(dtype=torch.float32).contiguous()

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

    def _merge_batches(self, first: _GpuBatch, second: _GpuBatch) -> _GpuBatch:
        obs = {k: torch.cat([first.obs[k], second.obs[k]], dim=0) for k in first.obs}
        next_obs = {k: torch.cat([first.next_obs[k], second.next_obs[k]], dim=0) for k in first.next_obs}
        action = {"action": torch.cat([first.action["action"], second.action["action"]], dim=0)}
        reward = torch.cat([first.reward, second.reward], dim=0)
        bootstrap = torch.cat([first.bootstrap, second.bootstrap], dim=0)
        return _GpuBatch(obs, next_obs, action, reward, bootstrap)

    def _sample_train_batch(self) -> _GpuBatch:
        if self.replay.size() <= 0:
            assert self.demo_replay.size() > 0, "both online and demo replay are empty"
            batch = self.demo_replay.sample(self.cfg.batch_size, self.cfg.rl_device)
            self._inflate_batch(batch)
            return batch

        demo_ratio = float(np.clip(self.cfg.demo_batch_ratio, 0.0, 1.0))
        demo_bsize = 0
        if self.demo_replay.size() > 0 and demo_ratio > 0.0:
            demo_bsize = int(round(self.cfg.batch_size * demo_ratio))
            demo_bsize = min(max(demo_bsize, 1), self.cfg.batch_size)
        online_bsize = self.cfg.batch_size - demo_bsize

        if online_bsize <= 0:
            batch = self.demo_replay.sample(self.cfg.batch_size, self.cfg.rl_device)
            self._inflate_batch(batch)
            return batch

        online_batch = self.replay.sample(online_bsize, self.cfg.rl_device)
        self._inflate_batch(online_batch)
        if demo_bsize == 0:
            return online_batch

        demo_batch = self.demo_replay.sample(demo_bsize, self.cfg.rl_device)
        self._inflate_batch(demo_batch)
        return self._merge_batches(online_batch, demo_batch)

    def _sample_demo_batch(self) -> Optional[_GpuBatch]:
        if not self.cfg.add_bc_loss or self.demo_replay.size() <= 0:
            return None
        batch = self.demo_replay.sample(self.cfg.batch_size, self.cfg.rl_device)
        self._inflate_batch(batch)
        return batch

    def _append_pending_transition(
        self,
        pending_episode: Dict[str, object],
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        obs_store = pending_episode["obs"]
        next_obs_store = pending_episode["next_obs"]
        assert isinstance(obs_store, dict)
        assert isinstance(next_obs_store, dict)
        for k, v in obs.items():
            obs_store[k].append(v.detach().clone())
            next_obs_store[k].append(next_obs[k].detach().clone())

        cast_list = pending_episode["action"]
        assert isinstance(cast_list, list)
        cast_list.append(action.detach().clone())
        reward_list = pending_episode["reward"]
        assert isinstance(reward_list, list)
        reward_list.append(reward.detach().clone())
        done_list = pending_episode["done"]
        assert isinstance(done_list, list)
        done_list.append(done.detach().clone())
        success_list = pending_episode["success"]
        assert isinstance(success_list, list)
        success_list.append(success.detach().clone())

    def _flush_pending_demo_episode(self, pending_episode: Dict[str, object], *, keep: bool) -> None:
        obs_store = pending_episode["obs"]
        next_obs_store = pending_episode["next_obs"]
        action_store = pending_episode["action"]
        reward_store = pending_episode["reward"]
        done_store = pending_episode["done"]
        success_store = pending_episode["success"]
        assert isinstance(obs_store, dict)
        assert isinstance(next_obs_store, dict)
        assert isinstance(action_store, list)
        assert isinstance(reward_store, list)
        assert isinstance(done_store, list)
        assert isinstance(success_store, list)

        if keep and action_store:
            obs_batch = {k: torch.stack(v, dim=0) for k, v in obs_store.items()}
            next_obs_batch = {k: torch.stack(v, dim=0) for k, v in next_obs_store.items()}
            action_batch = torch.stack(action_store, dim=0)
            reward_batch = torch.stack(reward_store, dim=0)
            done_batch = torch.stack(done_store, dim=0)
            success_batch = torch.stack(success_store, dim=0)
            self.demo_replay.add_step(
                obs_batch,
                next_obs_batch,
                action_batch,
                reward_batch,
                done_batch,
                success_batch,
            )

        pending_episode["obs"] = {self.cfg.rl_camera: [], "prop": [], "state": []}
        pending_episode["next_obs"] = {self.cfg.rl_camera: [], "prop": [], "state": []}
        pending_episode["action"] = []
        pending_episode["reward"] = []
        pending_episode["done"] = []
        pending_episode["success"] = []

    def warm_up(self) -> Dict[str, torch.Tensor]:
        obs = self.train_env.reset()
        pending_demo_episodes: List[Dict[str, object]] = []
        for _ in range(self.cfg.num_train_envs):
            pending_demo_episodes.append(
                {
                    "obs": {self.cfg.rl_camera: [], "prop": [], "state": []},
                    "next_obs": {self.cfg.rl_camera: [], "prop": [], "state": []},
                    "action": [],
                    "reward": [],
                    "done": [],
                    "success": [],
                }
            )
        warmup_bar = tqdm(
            total=max(self.cfg.num_warm_up_episode * self.cfg.episode_length, 1),
            desc="Warmup",
            leave=False,
            mininterval=0.2,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        while self.replay.num_episode < self.cfg.num_warm_up_episode:
            train_obs = self._slice_obs(obs, self.train_idx)
            with torch.no_grad(), utils.eval_mode(self.bc_policy):
                train_actions = self.bc_policy.act(train_obs, cpu=False)
            full_actions = self._compose_full_action(train_actions=train_actions)
            next_obs, rewards, dones, successes = self.train_env.step(full_actions)
            packed_obs = self._pack_replay_obs(train_obs)
            packed_next_obs = self._pack_replay_obs(self._slice_obs(next_obs, self.train_idx))
            train_rewards = rewards.index_select(0, self.train_idx)
            train_dones = dones.index_select(0, self.train_idx)
            train_successes = successes.index_select(0, self.train_idx)
            self.replay.add_step(
                packed_obs,
                packed_next_obs,
                train_actions,
                train_rewards,
                train_dones,
                train_successes,
            )
            for env_id in range(self.cfg.num_train_envs):
                per_obs = {k: v[env_id] for k, v in packed_obs.items()}
                per_next_obs = {k: v[env_id] for k, v in packed_next_obs.items()}
                self._append_pending_transition(
                    pending_demo_episodes[env_id],
                    per_obs,
                    per_next_obs,
                    train_actions[env_id],
                    train_rewards[env_id],
                    train_dones[env_id],
                    train_successes[env_id],
                )
                if bool(train_dones[env_id].item()):
                    self._flush_pending_demo_episode(
                        pending_demo_episodes[env_id],
                        keep=bool(train_successes[env_id].item()),
                    )
            obs = next_obs
            warmup_bar.update(self.cfg.num_train_envs)
            warmup_bar.set_postfix(episodes=f"{self.replay.num_episode}/{self.cfg.num_warm_up_episode}")

        warmup_bar.close()
        print(
            f"Warm up done. online replay: {self.replay.size()}, "
            f"demo replay: {self.demo_replay.size()}, "
            f"demo successes: {self.demo_replay.num_success}"
        )
        return obs

    def eval(self) -> Dict[str, object]:
        if self.eval_idx.numel() == 0:
            return {"mean_score": 0.0, "videos": [], "obs": self.train_env.reset()}

        obs = self.train_env.reset()
        success_count = 0
        episode_count = 0
        eval_steps = 0
        frames: List[np.ndarray] = []
        max_eval_episodes = max(self.cfg.num_eval_episode, self.cfg.num_eval_envs)
        max_eval_steps = int(self.cfg.eval_max_steps)
        eval_bar = tqdm(
            total=max_eval_episodes,
            desc="Eval",
            leave=False,
            mininterval=0.2,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        with torch.no_grad(), utils.eval_mode(self.agent):
            while episode_count < max_eval_episodes:
                if max_eval_steps > 0 and eval_steps >= max_eval_steps:
                    break
                eval_obs = self._slice_obs(obs, self.eval_idx)
                eval_actions = self.agent.act(eval_obs, eval_mode=True, stddev=0.0, cpu=False)
                full_actions = self._compose_full_action(eval_actions=eval_actions)
                next_obs, _, dones, successes = self.train_env.step(full_actions)
                eval_steps += int(self.eval_idx.numel())
                eval_bar.set_postfix(
                    success=success_count,
                    episodes=episode_count,
                    steps=eval_steps,
                    refresh=False,
                )
                eval_bar.refresh()
                if self.cfg.num_eval_videos > 0 and len(frames) < self.cfg.episode_length:
                    frames.append(self._obs_to_video_frame(next_obs, int(self.eval_idx[0].item())))
                eval_dones = dones.index_select(0, self.eval_idx)
                eval_successes = successes.index_select(0, self.eval_idx)
                if eval_dones.any():
                    done_count = int(eval_dones.sum().item())
                    success_batch = int((eval_dones & eval_successes).sum().item())
                    episode_count += done_count
                    success_count += success_batch
                    eval_bar.update(done_count)
                    eval_bar.set_postfix(
                        success=success_count,
                        episodes=episode_count,
                        steps=eval_steps,
                        refresh=False,
                    )
                obs = next_obs
        eval_bar.close()

        videos = []
        if self.cfg.num_eval_videos > 0 and frames:
            video_dir = Path(self.work_dir).joinpath("eval_videos")
            video_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=video_dir) as tmp:
                video_path = tmp.name
            imageio.mimsave(
                video_path,
                frames,
                fps=self.cfg.video_fps,
                codec="libx264",
            )
            videos.append(wandb.Video(video_path, fps=self.cfg.video_fps, format="mp4"))

        mean_score = success_count / max(episode_count, 1)
        obs = self.train_env.reset()
        return {
            "mean_score": float(mean_score),
            "videos": videos,
            "obs": obs,
        }

    def rl_train(self, stat: common_utils.MultiCounter) -> None:
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for idx in range(self.cfg.num_critic_update):
            batch = self._sample_train_batch()
            update_actor = idx == self.cfg.num_critic_update - 1
            bc_batch = None
            ref_agent = None
            if update_actor:
                bc_batch = self._sample_demo_batch()
            if update_actor and bc_batch is not None and self.cfg.add_bc_loss:
                ref_agent = self.ref_agent
                assert ref_agent is not None
            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, ref_agent)
            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())
            if self.demo_replay.size() > 0:
                stat["data/demo_frac"].append(float(min(max(self.cfg.demo_batch_ratio, 0.0), 1.0)))

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / max(elapsed_time, 1e-6))
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["other/replay"].append(self.replay.size())
        stat["other/demo_replay"].append(self.demo_replay.size())
        stat["score/num_success"].append(self.replay.num_success)
        stat["score/demo_num_success"].append(self.demo_replay.num_success)

        saver.save(self.agent.state_dict(), None, save_latest=True)

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
        obs = eval_metrics["obs"]
        running_lengths = torch.zeros(self.cfg.num_train_envs, dtype=torch.long, device=self.train_env.device)
        running_rewards = torch.zeros(self.cfg.num_train_envs, dtype=torch.float32, device=self.train_env.device)
        return obs, running_lengths, running_rewards

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
        stopwatch = common_utils.Stopwatch()
        running_lengths = torch.zeros(self.cfg.num_train_envs, dtype=torch.long, device=self.train_env.device)
        running_rewards = torch.zeros(self.cfg.num_train_envs, dtype=torch.float32, device=self.train_env.device)
        train_bar = tqdm(
            total=self.cfg.num_train_step,
            desc="Train",
            initial=min(self.global_step, self.cfg.num_train_step),
            mininterval=0.2,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                train_obs = self._slice_obs(obs, self.train_idx)
                train_actions = self.agent.act(train_obs, eval_mode=False, stddev=stddev, cpu=False)
                actions = self._compose_full_action(train_actions=train_actions)
                stat["data/stddev"].append(stddev)

            with stopwatch.time("env step"):
                next_obs, rewards, dones, successes = self.train_env.step(actions)
                train_rewards = rewards.index_select(0, self.train_idx)
                train_dones = dones.index_select(0, self.train_idx)
                train_successes = successes.index_select(0, self.train_idx)

            running_lengths += 1
            running_rewards += train_rewards

            with stopwatch.time("add"):
                self.replay.add_step(
                    self._pack_replay_obs(train_obs),
                    self._pack_replay_obs(self._slice_obs(next_obs, self.train_idx)),
                    train_actions,
                    train_rewards,
                    train_dones,
                    train_successes,
                )
                self.global_iter += 1
                self.global_step += self.cfg.num_train_envs
                train_bar.n = min(self.global_step, self.cfg.num_train_step)
                train_bar.set_postfix(
                    episodes=self.global_episode,
                    replay=self.replay.size(),
                    refresh=False,
                )
                train_bar.refresh()

            done_ids = train_dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self.global_episode += int(done_ids.numel())
                stat["score/train_score"].append(
                    float(train_successes[done_ids].float().sum().item()),
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

            if self.global_step > 0 and self.global_step % self.cfg.log_per_step == 0:
                obs, running_lengths, running_rewards = self.log_and_save(stopwatch, stat, saver)

        train_bar.close()


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
