from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def _ensure_import_path(path: str) -> None:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


class ManiFeelBulbVecEnv:
    """Vectorized ManiFeel bulb environment with IBRL-friendly observations."""

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
        external_camera: str = "client",
        image_hw: Tuple[int, int] = (256, 256),
        light_factor: float = 1.0,
        force_render: bool = True,
        capture_video: bool = False,
        extra_overrides: Optional[List[str]] = None,
    ) -> None:
        global torch
        import torch

        self.manifeel_root = os.path.abspath(manifeel_root)
        self.isaacgym_envs_path = os.path.abspath(isaacgym_envs_path)
        self.num_envs = int(num_envs)
        self.device = torch.device(rl_device)
        self.env_reward_scale = float(env_reward_scale)
        self.rl_camera = str(rl_camera)
        self.external_camera = str(external_camera)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.n_obs_steps = int(n_obs_steps)
        self.action_dim = 7
        self.observation_shape = (3, self.image_hw[0], self.image_hw[1])
        self.prop_shape = (7,)

        _ensure_import_path(self.manifeel_root)
        _ensure_import_path(self.isaacgym_envs_path)
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        config_dir = os.path.join(self.manifeel_root, "manifeel", "config")
        vision_cfg_path = os.path.join(config_dir, "task", "vision_wrist.yaml")
        vision_cfg = OmegaConf.load(vision_cfg_path)

        overrides = [
            f"num_envs={self.num_envs}",
            f"+seed={int(seed)}",
            f"sim_device={sim_device}",
            f"rl_device={rl_device}",
            f"graphics_device_id={int(graphics_device_id)}",
            f"headless={str(bool(headless)).lower()}",
            f"capture_video={str(bool(capture_video)).lower()}",
            f"force_render={str(bool(force_render)).lower()}",
            f"light_factor={float(light_factor)}",
            f"task.rl.max_episode_length={int(max_episode_length)}",
            "task.env.use_camera_obs=true",
            "task.env.use_camera=true",
            "task.env.enableCameraSensors=true",
            "task.env.use_isaac_gym_tactile=false",
            "task.env.use_tactile_field_obs=false",
            "task.env.use_shear_force=false",
            "task.env.use_physx_gripper_control=false",
            "task.env.franka_open_gripper_width=0.01",
            "task.env.franka_close_gripper_width=0.01",
            f"task.env.obsDims.{self.rl_camera}=[{self.image_hw[0]},{self.image_hw[1]},3]",
            f"task.env.obsDims.{self.external_camera}=[{self.image_hw[0]},{self.image_hw[1]},3]",
        ]
        if extra_overrides:
            overrides.extend(extra_overrides)

        from manifeel.envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name="isaacgym_config_bulb", overrides=overrides)
            cfg.shape_meta = vision_cfg.shape_meta
            cfg.light_factor = float(light_factor)
            cfg.task.env.use_isaac_gym_tactile = False
            cfg.task.env.use_tactile_field_obs = False
            cfg.task.env.use_shear_force = False
            cfg.task.env.use_gelsight = False
            self._env = MultipleIsaacEnvWrapper(cfg)
        self._seed = int(seed)
        self._episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._total_episodes = 0
        self._total_successes = 0
        self._last_done_steps = torch.zeros(0, dtype=torch.long, device=self.device)
        self._last_done_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._last_done_successes = torch.zeros(0, dtype=torch.bool, device=self.device)
        self._bc_wrist_hist: Optional[torch.Tensor] = None
        self._bc_state_hist: Optional[torch.Tensor] = None

    def _extract_current_obs(self, obs_np: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        wrist = torch.from_numpy(obs_np[self.rl_camera]).to(self.device).float()
        if float(wrist.max()) <= 1.01:
            wrist = wrist * 255.0
        state = torch.from_numpy(obs_np["state"]).to(self.device).float()
        return wrist, state

    def _set_history(self, wrist: torch.Tensor, state: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> None:
        if self._bc_wrist_hist is None or self._bc_state_hist is None:
            self._bc_wrist_hist = wrist.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1).contiguous()
            self._bc_state_hist = state.unsqueeze(1).repeat(1, self.n_obs_steps, 1).contiguous()
            return

        if env_ids is None:
            self._bc_wrist_hist = wrist.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1).contiguous()
            self._bc_state_hist = state.unsqueeze(1).repeat(1, self.n_obs_steps, 1).contiguous()
            return

        self._bc_wrist_hist[env_ids] = wrist[env_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
        self._bc_state_hist[env_ids] = state[env_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1)

    def _append_history(self, wrist: torch.Tensor, state: torch.Tensor, done_ids: torch.Tensor) -> None:
        assert self._bc_wrist_hist is not None
        assert self._bc_state_hist is not None
        active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        active_mask[done_ids] = False
        if active_mask.any():
            self._bc_wrist_hist[active_mask, :-1] = self._bc_wrist_hist[active_mask, 1:].clone()
            self._bc_wrist_hist[active_mask, -1] = wrist[active_mask]
            self._bc_state_hist[active_mask, :-1] = self._bc_state_hist[active_mask, 1:].clone()
            self._bc_state_hist[active_mask, -1] = state[active_mask]
        if done_ids.numel() > 0:
            self._set_history(wrist, state, done_ids)

    def _format_obs(self, obs_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        wrist, state = self._extract_current_obs(obs_np)
        assert self._bc_wrist_hist is not None
        assert self._bc_state_hist is not None
        return {
            self.rl_camera: wrist,
            "prop": state,
            "state": state,
            "bc_wrist": self._bc_wrist_hist,
            "bc_state": self._bc_state_hist,
        }

    def _replace_done_obs(
        self,
        obs_np: Dict[str, np.ndarray],
        done_ids: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        if done_ids.numel() == 0:
            return obs_np
        self._env.envs.reset_idx(done_ids)
        self._env.envs.compute_observations()
        reset_np = self._env._transform_obs_data(self._env.envs.obs_dict)
        done_idx = done_ids.detach().cpu().numpy()
        for key in obs_np.keys():
            obs_np[key][done_idx] = reset_np[key][done_idx]
        return obs_np

    def reset(self) -> Dict[str, torch.Tensor]:
        self._episode_reward.zero_()
        self._episode_step.zero_()
        obs_np = self._env.reset()
        self._env.render_cache = obs_np
        wrist, state = self._extract_current_obs(obs_np)
        self._set_history(wrist, state)
        return self._format_obs(obs_np)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        actions_np = actions.detach().to("cpu", dtype=torch.float32).numpy()
        obs_np, _, dones_np, _ = self._env.step(actions_np)

        rewards = self._env.envs.rew_buf.clone().to(self.device).float() * self.env_reward_scale
        dones = torch.from_numpy(dones_np).to(device=self.device, dtype=torch.bool)
        successes = self._env.envs._check_success().bool()
        self._episode_reward += rewards
        self._episode_step += 1

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._last_done_steps = self._episode_step[done_ids].clone()
            self._last_done_rewards = self._episode_reward[done_ids].clone()
            self._last_done_successes = successes[done_ids].clone()
            self._total_episodes += int(done_ids.numel())
            self._total_successes += int(successes[done_ids].sum().item())
        else:
            self._last_done_steps = torch.zeros(0, dtype=torch.long, device=self.device)
            self._last_done_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
            self._last_done_successes = torch.zeros(0, dtype=torch.bool, device=self.device)

        obs_np = self._replace_done_obs(obs_np, done_ids)
        self._env.render_cache = obs_np

        wrist, state = self._extract_current_obs(obs_np)
        self._append_history(wrist, state, done_ids)

        if done_ids.numel() > 0:
            self._episode_reward[done_ids] = 0.0
            self._episode_step[done_ids] = 0

        return self._format_obs(obs_np), rewards, dones, successes

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def success_rate(self) -> float:
        if self._total_episodes == 0:
            return 0.0
        return self._total_successes / self._total_episodes

    @property
    def episode_reward(self) -> torch.Tensor:
        return self._episode_reward.clone()

    @property
    def last_done_steps(self) -> torch.Tensor:
        return self._last_done_steps.clone()

    @property
    def last_done_rewards(self) -> torch.Tensor:
        return self._last_done_rewards.clone()

    @property
    def last_done_successes(self) -> torch.Tensor:
        return self._last_done_successes.clone()
