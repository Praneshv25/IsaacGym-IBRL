from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

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
    """Direct TacSL bulb wrapper for dense-reward RL training."""

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
            raise ValueError(f"Direct train wrapper expects 256x256 observations, got {self.image_hw}.")

        config_dir = os.path.join(self.manifeel_root, "manifeel", "config")
        from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb
        from isaacgymenvs.utils.reformat import omegaconf_to_dict
        from isaacgymenvs.utils.utils import set_seed

        self._set_seed = set_seed
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name="isaacgym_config_bulb")

            cfg.num_envs = self.num_envs
            cfg.sim_device = sim_device
            cfg.rl_device = rl_device
            cfg.graphics_device_id = int(graphics_device_id)
            cfg.headless = bool(headless)
            cfg.capture_video = False
            cfg.force_render = bool(force_render)
            cfg.task.rl.max_episode_length = self.max_episode_length

            self._cfg = cfg
            cfg_dict = omegaconf_to_dict(cfg.task)
            self.envs = TacSLTaskBulb(
                cfg=cfg_dict,
                rl_device=cfg.rl_device,
                sim_device=cfg.sim_device,
                graphics_device_id=cfg.graphics_device_id,
                headless=cfg.headless,
                virtual_screen_capture=cfg.capture_video,
                force_render=cfg.force_render,
            )

        self._episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_done_steps = torch.zeros(0, dtype=torch.long, device=self.device)
        self._last_done_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._last_done_successes = torch.zeros(0, dtype=torch.bool, device=self.device)
        self._bc_wrist_hist: Optional[torch.Tensor] = None
        self._bc_state_hist: Optional[torch.Tensor] = None
        self.seed(self._seed)

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = self._set_seed(
            seed,
            torch_deterministic=self._cfg.torch_deterministic,
            rank=0,
        )

    def _extract_current_obs(self, obs: Dict[str, "torch.Tensor"]) -> Tuple["torch.Tensor", "torch.Tensor"]:
        global torch

        wrist = obs[self.isaac_camera].detach().to(self.device).float()
        if wrist.dim() == 4 and wrist.shape[-1] == 3:
            wrist = wrist.permute(0, 3, 1, 2).contiguous()
        if float(wrist.max()) <= 1.01:
            wrist = wrist * 255.0

        ee_pos = obs["ee_pos"].detach().to(self.device).float()
        ee_quat = obs["ee_quat"].detach().to(self.device).float()
        state = torch.cat([ee_pos, ee_quat], dim=1)
        return wrist, state

    def _set_history(self, wrist: "torch.Tensor", state: "torch.Tensor", env_ids: Optional["torch.Tensor"] = None) -> None:
        wrist_hist = wrist.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1).contiguous()
        state_hist = state.unsqueeze(1).repeat(1, self.n_obs_steps, 1).contiguous()
        if self._bc_wrist_hist is None or self._bc_state_hist is None or env_ids is None:
            self._bc_wrist_hist = wrist_hist
            self._bc_state_hist = state_hist
            return
        self._bc_wrist_hist[env_ids] = wrist_hist[env_ids]
        self._bc_state_hist[env_ids] = state_hist[env_ids]

    def _append_history(self, wrist: "torch.Tensor", state: "torch.Tensor", done_ids: "torch.Tensor") -> None:
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

    def _format_obs(self, obs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        wrist, state = self._extract_current_obs(obs)
        assert self._bc_wrist_hist is not None
        assert self._bc_state_hist is not None
        return {
            self.rl_camera: wrist,
            "prop": state,
            "state": state,
            "bc_wrist": self._bc_wrist_hist,
            "bc_state": self._bc_state_hist,
        }

    def _raw_reset(self) -> Dict[str, "torch.Tensor"]:
        import torch

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.envs.reset_idx(env_ids)
        self.envs.compute_observations()
        reset_out = self.envs.reset()
        return reset_out["obs"]

    def _reset_done_envs(self, done_ids: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        self.envs.reset_idx(done_ids)
        self.envs.compute_observations()
        return self.envs.obs_dict

    def reset(self) -> Dict[str, "torch.Tensor"]:
        self._episode_reward.zero_()
        self._episode_step.zero_()
        obs = self._raw_reset()
        wrist, state = self._extract_current_obs(obs)
        self._set_history(wrist, state)
        return self._format_obs(obs)

    def step(self, actions: "torch.Tensor"):
        global torch

        if isinstance(actions, torch.Tensor):
            action_tensor = actions.to(dtype=torch.float32, device=self.device)
        else:
            action_tensor = torch.from_numpy(actions).to(dtype=torch.float32, device=self.device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)

        obs_out, reward, reset, info = self.envs.step(action_tensor)
        obs = obs_out["obs"]
        rewards = reward.detach().clone().to(self.device).float() * self.env_reward_scale
        dones = reset.detach().clone().to(self.device).bool()
        successes = self.envs._check_success().detach().clone().to(self.device).bool()

        self._episode_reward += rewards
        self._episode_step += 1

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._last_done_steps = self._episode_step[done_ids].clone()
            self._last_done_rewards = self._episode_reward[done_ids].clone()
            self._last_done_successes = successes[done_ids].clone()
            reset_obs = self._reset_done_envs(done_ids)
            for key in obs.keys():
                obs[key][done_ids] = reset_obs[key][done_ids]
        else:
            self._last_done_steps = torch.zeros(0, dtype=torch.long, device=self.device)
            self._last_done_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
            self._last_done_successes = torch.zeros(0, dtype=torch.bool, device=self.device)

        wrist, state = self._extract_current_obs(obs)
        self._append_history(wrist, state, done_ids)

        if done_ids.numel() > 0:
            self._episode_reward[done_ids] = 0.0
            self._episode_step[done_ids] = 0

        return self._format_obs(obs), rewards, dones, successes

    def render(self):
        return None

    def close(self) -> None:
        return

    @property
    def last_done_steps(self):
        return self._last_done_steps.clone()

    @property
    def last_done_rewards(self):
        return self._last_done_rewards.clone()

    @property
    def last_done_successes(self):
        return self._last_done_successes.clone()
