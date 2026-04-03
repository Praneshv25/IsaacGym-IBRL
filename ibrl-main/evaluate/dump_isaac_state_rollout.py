#!/usr/bin/env python
"""
Headless IsaacGym (TacSL bulb): run **state-only** BC for one episode and save
``states`` / ``actions`` to ``.npz`` for offline visualization (e.g. PyBullet).

No viewer, no sim cameras — avoids GPU rgbImage / viewer issues.  Does **not**
support vision BC (needs images every step).

Replay::

    python evaluate/viz_bc_pybullet_franka.py --replay_npz rollout.npz

Run from ``ibrl-main``::

    python evaluate/dump_isaac_state_rollout.py \\
        --checkpoint exps/bc_isaac/shard9/model0.pt \\
        --out rollout.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaacgym  # noqa: E402, F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.isaac_gym_wrapper import IsaacGymBulbEnv  # noqa: E402
from evaluate.eval_bc_isaac_min import _build_policy, _load_cfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Dump one state-BC Isaac rollout to .npz")
    p.add_argument("--checkpoint", required=True, help="model0.pt (cfg.yaml alongside)")
    p.add_argument("--out", default="isaac_state_rollout.npz", help="Output .npz path")
    p.add_argument("--isaacgym_envs_path", default="", help="Else from cfg.yaml")
    p.add_argument("--max_episode_length", type=int, default=1000)
    p.add_argument("--sim_device", default="")
    p.add_argument("--rl_device", default="")
    p.add_argument(
        "--graphics_device_id",
        type=int,
        default=-999,
        help="GPU for graphics; -999 uses cfg.yaml",
    )
    p.add_argument("--seed", type=int, default=-1)
    args = p.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")
    run_dir = os.path.dirname(ckpt)
    cfg_path = os.path.join(run_dir, "cfg.yaml")
    if not os.path.isfile(cfg_path):
        sys.exit(f"cfg.yaml not found: {cfg_path}")

    cfg_y = _load_cfg(cfg_path)
    ds = cfg_y.get("dataset") or {}
    if isinstance(ds, dict):
        ik = ds.get("image_keys") or []
        csv = (ds.get("image_keys_csv") or "").strip()
        if ik or csv:
            sys.exit(
                "dump_isaac_state_rollout: vision dataset in cfg.yaml. "
                "Use a state-only BC checkpoint (train_bc_isaac.py) or record in Isaac with cameras."
            )

    ig_path = args.isaacgym_envs_path or cfg_y.get("isaacgym_envs_path") or ""
    if not str(ig_path).strip():
        sys.exit("Set --isaacgym_envs_path or isaacgym_envs_path in cfg.yaml")
    ig_path = os.path.expanduser(str(ig_path))
    if not os.path.isabs(ig_path):
        ig_path = os.path.normpath(os.path.join(os.getcwd(), ig_path))

    sim_dev = args.sim_device or cfg_y.get("sim_device") or "cuda:0"
    rl_dev = args.rl_device or cfg_y.get("rl_device") or sim_dev
    gdev = (
        args.graphics_device_id
        if args.graphics_device_id != -999
        else int(cfg_y.get("graphics_device_id", -1))
    )
    seed = int(cfg_y.get("seed", 0)) if args.seed < 0 else args.seed
    mel = None if args.max_episode_length <= 0 else int(args.max_episode_length)

    device = torch.device(rl_dev if torch.cuda.is_available() else "cpu")
    policy = _build_policy(cfg_y, ckpt, device)
    policy.train(False)

    scale_pt = os.path.join(run_dir, "action_scale.pt")
    action_scale: Optional[torch.Tensor] = None
    if os.path.isfile(scale_pt):
        action_scale = torch.load(scale_pt, map_location=device)

    env = IsaacGymBulbEnv(
        isaacgym_envs_path=ig_path,
        num_envs=1,
        sim_device=sim_dev,
        rl_device=rl_dev,
        graphics_device_id=gdev,
        headless=True,
        seed=seed,
        max_episode_length=mel,
    )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    obs = env.reset()
    states.append(obs["state"][0].detach().cpu().numpy().astype(np.float32))

    with torch.no_grad():
        while True:
            act = policy.act(obs, eval_mode=True, stddev=0.0, cpu=False)
            if action_scale is not None:
                act = act * action_scale.to(act.device).view(1, -1)
            obs, _r, dones, successes = env.step(act)
            actions.append(act[0].detach().cpu().numpy().astype(np.float32))
            states.append(obs["state"][0].detach().cpu().numpy().astype(np.float32))
            if bool(dones[0].item()):
                success = bool(successes[0].item())
                break

    st = np.stack(states, axis=0)
    ac = np.stack(actions, axis=0)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        states=st,
        actions=ac,
        success=np.array([1 if success else 0], dtype=np.int8),
        checkpoint=np.array([ckpt], dtype=object),
    )
    print(
        f"[dump_isaac_state_rollout] saved {out_path} | T={ac.shape[0]} | "
        f"success={success} | states={st.shape} actions={ac.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
