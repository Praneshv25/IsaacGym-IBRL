from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def _ensure_import_path(path: str) -> None:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


class ManiFeelBulbVecEnv:
    """Thin adapter around ManiFeel's env_runner-owned rollout env."""

    observation_shape: Tuple[int, ...]
    prop_shape: Tuple[int, ...]
    action_dim: int

    def __init__(
        self,
        *,
        manifeel_root: str,
        isaacgym_envs_path: str,
        num_envs: int,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        seed: int,
        n_obs_steps: int,
        max_episode_length: int,
        env_reward_scale: float = 1.0,
        rl_camera: str = "wrist",
        isaac_camera: str = "wrist",
        image_hw: Tuple[int, int] = (256, 256),
        force_render: bool = False,
    ) -> None:
        global torch
        import torch

        self.manifeel_root = os.path.abspath(manifeel_root)
        self.isaacgym_envs_path = os.path.abspath(isaacgym_envs_path)
        self.num_envs = int(num_envs)
        self.device = torch.device(rl_device)
        self.env_reward_scale = float(env_reward_scale)
        self.rl_camera = str(rl_camera)
        self.isaac_camera = str(isaac_camera)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.n_obs_steps = int(n_obs_steps)
        self.max_episode_length = int(max_episode_length)
        self.action_dim = 7
        self.observation_shape = (3, self.image_hw[0], self.image_hw[1])
        self.prop_shape = (7,)
        self._seed = int(seed)

        _ensure_import_path(self.manifeel_root)
        _ensure_import_path(self.isaacgym_envs_path)
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        if self.image_hw != (256, 256):
            raise ValueError(f"ManiFeel restart path expects 256x256 observations, got {self.image_hw}.")
        if self.isaac_camera != self.rl_camera:
            raise ValueError(
                f"isaac_camera={self.isaac_camera!r} must match rl_camera={self.rl_camera!r} "
                "in the ManiFeel-owned rollout path."
            )

        config_dir = os.path.join(self.manifeel_root, "manifeel", "config")
        overrides = [
            "task=vision_wrist",
            "isaacgym_cfg_name=isaacgym_config_bulb.yaml",
            f"training.seed={self._seed}",
            f"n_obs_steps={self.n_obs_steps}",
            "n_action_steps=1",
            f"task.env_runner.n_test={self.num_envs}",
            "task.env_runner.n_test_vis=1",
            f"task.env_runner.max_steps={self.max_episode_length}",
        ]
        output_dir = os.path.join(self.manifeel_root, "data", "outputs", "ibrl_train_env")
        os.makedirs(output_dir, exist_ok=True)

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name="train_diffusion_workspace", overrides=overrides)
            cfg.training.device = rl_device
            cfg.task.env_runner.test_start_seed = self._seed
            runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=output_dir,
            )

        self._runner = runner
        self._env = runner.env
        self._task_env = runner.env.env.env.envs
        self._episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_done_steps = torch.zeros(0, dtype=torch.long, device=self.device)
        self._last_done_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._last_done_successes = torch.zeros(0, dtype=torch.bool, device=self.device)

    def _format_obs(self, obs_np: Dict[str, np.ndarray]) -> Dict[str, "torch.Tensor"]:
        global torch
        bc_wrist = torch.from_numpy(obs_np[self.isaac_camera]).to(self.device).float()
        if float(bc_wrist.max()) <= 1.01:
            bc_wrist = bc_wrist * 255.0
        bc_state = torch.from_numpy(obs_np["state"]).to(self.device).float()
        return {
            self.rl_camera: bc_wrist[:, -1],
            "prop": bc_state[:, -1],
            "state": bc_state[:, -1],
            "bc_wrist": bc_wrist,
            "bc_state": bc_state,
        }

    def reset(self) -> Dict[str, "torch.Tensor"]:
        self._episode_reward.zero_()
        self._episode_step.zero_()
        obs_np = self._env.reset()
        return self._format_obs(obs_np)

    def step(self, actions: "torch.Tensor"):
        global torch

        actions_np = actions.detach().to("cpu", dtype=torch.float32).numpy()[:, None, :]
        obs_np, _, dones_np, _ = self._env.step(actions_np)

        rewards = self._task_env.rew_buf.detach().clone().to(self.device).float() * self.env_reward_scale
        dones = torch.from_numpy(dones_np).to(device=self.device, dtype=torch.bool)
        successes = self._task_env._check_success().bool().to(self.device)

        self._episode_reward += rewards
        self._episode_step += 1

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._last_done_steps = self._episode_step[done_ids].clone()
            self._last_done_rewards = self._episode_reward[done_ids].clone()
            self._last_done_successes = successes[done_ids].clone()
            self._episode_reward[done_ids] = 0.0
            self._episode_step[done_ids] = 0
        else:
            self._last_done_steps = torch.zeros(0, dtype=torch.long, device=self.device)
            self._last_done_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
            self._last_done_successes = torch.zeros(0, dtype=torch.bool, device=self.device)

        return self._format_obs(obs_np), rewards, dones, successes

    def render(self):
        return self._env.render()

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def last_done_steps(self):
        return self._last_done_steps.clone()

    @property
    def last_done_rewards(self):
        return self._last_done_rewards.clone()

    @property
    def last_done_successes(self):
        return self._last_done_successes.clone()
