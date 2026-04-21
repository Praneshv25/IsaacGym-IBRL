#!/usr/bin/env python3
"""
Evaluate a ManiFeel diffusion checkpoint on the Isaac Gym bulb insertion env.

This script intentionally bypasses ManiFeel's default env runner and instead
uses the working IBRL bulb wrapper, so the checkpoint can be tested directly
on TacSLTaskBulb with:
  - wrist image
  - 7-D EE state
  - 7-D actions (including gripper)
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
_WORKSPACE_ROOT = os.path.dirname(_REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaacgym  # noqa: F401

import dill
import hydra
import torch
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from env.isaac_gym_wrapper import IsaacGymBulbEnv


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_workspace_and_policy_from_checkpoint(ckpt_path: Path, device: str):
    payload = torch.load(ckpt_path.open("rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(ckpt_path.parent))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    loaded_from = "model"
    if getattr(cfg.training, "use_ema", False) and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
        loaded_from = "ema_model"

    policy.to(torch.device(device))
    policy.eval()
    del payload
    return cfg, workspace, policy, loaded_from


def _obs_from_env(obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    wrist = obs["wrist"].detach().clone().float()
    if float(wrist.max()) > 1.01:
        wrist = wrist / 255.0
    state = obs["prop"].detach().clone().float()
    return wrist, state


def _stack_history(
    image_hist: Deque[torch.Tensor],
    state_hist: Deque[torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    wrist = torch.stack(list(image_hist), dim=1).to(device)
    state = torch.stack(list(state_hist), dim=1).to(device)
    return {"wrist": wrist, "state": state}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ManiFeel checkpoint on Isaac bulb env")
    parser.add_argument("--checkpoint", required=True, help="Path to ManiFeel .ckpt file")
    parser.add_argument("--manifeel_root", default="", help="Path to ManiFeel repo root if not sibling")
    parser.add_argument("--isaacgym_envs_path", required=True, help="Path to manifeel-isaacgymenvs-* repo")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_episode_length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Override diffusion denoising steps during eval")
    parser.add_argument("--log_every", type=int, default=1, help="Print step/reward every N env steps (0 disables)")
    args = parser.parse_args()

    manifeel_root = args.manifeel_root.strip()
    if not manifeel_root:
        manifeel_root = os.path.join(_WORKSPACE_ROOT, "manifeel")
    manifeel_root = os.path.abspath(manifeel_root)
    if manifeel_root not in sys.path:
        sys.path.insert(0, manifeel_root)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    device = torch.device(args.device)

    cfg, workspace, policy, loaded_from = _load_workspace_and_policy_from_checkpoint(ckpt_path, args.device)
    n_obs_steps = int(cfg.n_obs_steps)
    if args.num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(args.num_inference_steps)

    env = IsaacGymBulbEnv(
        isaacgym_envs_path=os.path.abspath(args.isaacgym_envs_path),
        num_envs=1,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id,
        headless=bool(args.headless),
        seed=int(args.seed),
        max_episode_length=int(args.max_episode_length),
        rl_camera="wrist",
        isaac_camera="wrist_2",
        image_hw=(256, 256),
    )

    if args.verbose:
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Policy weights source: {loaded_from}")
        print(
            f"Using n_obs_steps={n_obs_steps}, action chunk={policy.n_action_steps}, "
            f"horizon={policy.horizon}, num_inference_steps={getattr(policy, 'num_inference_steps', 'n/a')}"
        )
        print(f"Evaluating on Isaac bulb env for {args.num_episodes} episode(s)")

    total_successes = 0

    for ep in range(args.num_episodes):
        obs = env.reset()
        wrist, state = _obs_from_env(obs)

        image_hist: Deque[torch.Tensor] = deque(maxlen=n_obs_steps)
        state_hist: Deque[torch.Tensor] = deque(maxlen=n_obs_steps)
        for _ in range(n_obs_steps):
            image_hist.append(wrist)
            state_hist.append(state)

        done = False
        success = False
        step_idx = 0
        policy.reset()

        while not done and step_idx < args.max_episode_length:
            obs_dict = _stack_history(image_hist, state_hist, device)
            with torch.inference_mode():
                action_dict = policy.predict_action(obs_dict)

            action_chunk = action_dict["action"]  # (B, T, 7)
            if action_chunk.dim() != 3 or action_chunk.shape[0] != 1:
                raise ValueError(f"Expected action chunk shape (1,T,7), got {tuple(action_chunk.shape)}")

            # Keep evaluation logic as simple as possible: query policy, execute
            # the next action only, then re-plan from the fresh observation.
            act = action_chunk[:, 0, :].to(torch.device(args.rl_device)).float()
            obs, reward, dones, successes = env.step(act)
            wrist, state = _obs_from_env(obs)
            image_hist.append(wrist)
            state_hist.append(state)

            done = bool(dones[0].item())
            success = bool(successes[0].item())
            step_idx += 1
            if args.log_every > 0 and step_idx % args.log_every == 0:
                print(
                    f"[eval_manifeel_bulb] step={step_idx} reward={float(reward[0].item()):.6f} "
                    f"done={int(done)} success={int(success)}"
                )

        total_successes += int(success)
        if args.verbose:
            print(f"[eval_manifeel_bulb] episode={ep+1}/{args.num_episodes} success={int(success)} steps={step_idx}")

    success_rate = total_successes / max(args.num_episodes, 1)
    print(f"success_rate={success_rate:.4f} ({total_successes}/{args.num_episodes})")


if __name__ == "__main__":
    main()
