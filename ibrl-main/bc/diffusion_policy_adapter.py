import os
import sys
from pathlib import Path
from typing import Dict

import dill
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf


def _ensure_import_path(path: str) -> None:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


class DiffusionPolicyAdapter(nn.Module):
    """Frozen ManiFeel Diffusion Policy exposed through the IBRL BC API."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        diffusion_policy_root: str,
        manifeel_root: str,
        device: str,
        obs_image_key: str = "bc_wrist",
        obs_state_key: str = "bc_state",
    ) -> None:
        super().__init__()
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.device = torch.device(device)
        self.obs_image_key = obs_image_key
        self.obs_state_key = obs_state_key

        _ensure_import_path(diffusion_policy_root)
        _ensure_import_path(manifeel_root)

        try:
            OmegaConf.register_new_resolver("eval", eval, replace=True)
        except Exception:
            pass

        from diffusion_policy.workspace.base_workspace import BaseWorkspace

        payload = torch.load(open(self.checkpoint_path, "rb"), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        OmegaConf.resolve(cfg)

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=str(Path(self.checkpoint_path).parent))
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if getattr(cfg.training, "use_ema", False) and getattr(workspace, "ema_model", None) is not None:
            policy = workspace.ema_model

        self.cfg = cfg
        self.workspace = workspace
        self.policy = policy.to(self.device)
        self.policy.eval()
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.horizon = int(cfg.horizon)
        super().train(False)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def train(self, mode: bool = True):
        super().train(False)
        self.policy.eval()
        return self

    def _prepare_history(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.obs_image_key in obs and self.obs_state_key in obs:
            wrist = obs[self.obs_image_key]
            state = obs[self.obs_state_key]
        else:
            wrist = obs["wrist"]
            state = obs.get("prop", obs["state"])
            if wrist.dim() == 4:
                wrist = wrist.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            if state.dim() == 2:
                state = state.unsqueeze(1).repeat(1, self.n_obs_steps, 1)

        wrist = wrist.to(self.device).float()
        if float(wrist.max()) > 1.01:
            wrist = wrist / 255.0
        state = state.to(self.device).float()

        expected_wrist = None
        expected_state = None
        try:
            shape_meta = self.cfg.shape_meta
            expected_wrist = tuple(shape_meta.obs.wrist.shape)
            expected_state = tuple(shape_meta.obs.state.shape)
        except Exception:
            try:
                shape_meta = self.cfg.task.shape_meta
                expected_wrist = tuple(shape_meta.obs.wrist.shape)
                expected_state = tuple(shape_meta.obs.state.shape)
            except Exception:
                pass

        print(
            "[bc] history shapes "
            f"wrist={tuple(wrist.shape)} expected_frame={expected_wrist}; "
            f"state={tuple(state.shape)} expected_state={expected_state}",
            flush=True,
        )

        return {
            "wrist": wrist,
            "state": state,
        }

    def act(self, obs: Dict[str, torch.Tensor], *, eval_mode=True, cpu=True, **kwargs) -> torch.Tensor:
        del eval_mode, kwargs
        with torch.inference_mode():
            action_dict = self.policy.predict_action(self._prepare_history(obs))
            action = action_dict["action"]
            if action.dim() == 3:
                action = action[:, 0, :]
        action = action.detach()
        if cpu:
            action = action.cpu()
        return action
