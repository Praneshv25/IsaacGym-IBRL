#!/usr/bin/env python
"""
Record **one** state-BC rollout in IsaacGym (TacSL bulb) to an MP4.

Uses the Isaac Gym viewer framebuffer (``write_viewer_image_to_file``). Run with
a real display (**headless=False**), or on a headless machine pass
``--virtual_display`` (requires ``pyvirtualdisplay``).

Run from ``ibrl-main``::

    python evaluate/record_bc_isaac_episode.py \\
        --checkpoint exps/bc_isaac/run1/model0.pt \\
        --out episode.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaacgym  # noqa: E402, F401

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.isaac_gym_wrapper import IsaacGymBulbEnv  # noqa: E402
from evaluate.eval_bc_isaac_min import _build_policy, _load_cfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Record one BC episode (IsaacGym bulb) to MP4")
    p.add_argument("--checkpoint", required=True, help="model0.pt (cfg.yaml alongside)")
    p.add_argument("--out", default="bc_isaac_episode.mp4", help="Output video path")
    p.add_argument("--isaacgym_envs_path", default="", help="Else from cfg.yaml")
    p.add_argument("--max_episode_length", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--virtual_display",
        action="store_true",
        help="Use pyvirtualdisplay (for headless servers; install pyvirtualdisplay)",
    )
    p.add_argument("--sim_device", default="")
    p.add_argument("--rl_device", default="")
    p.add_argument("--graphics_device_id", type=int, default=0)
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
    ig_path = args.isaacgym_envs_path or cfg_y.get("isaacgym_envs_path") or ""
    if not str(ig_path).strip():
        sys.exit("Set --isaacgym_envs_path or isaacgym_envs_path in cfg.yaml")
    ig_path = os.path.expanduser(str(ig_path))
    if not os.path.isabs(ig_path):
        ig_path = os.path.normpath(os.path.join(os.getcwd(), ig_path))

    sim_dev = args.sim_device or cfg_y.get("sim_device") or "cuda:0"
    rl_dev = args.rl_device or cfg_y.get("rl_device") or sim_dev
    seed = int(cfg_y.get("seed", 0)) if args.seed < 0 else args.seed

    mel = None if args.max_episode_length <= 0 else int(args.max_episode_length)

    device = torch.device(rl_dev if torch.cuda.is_available() else "cpu")
    policy = _build_policy(cfg_y, ckpt, device)
    policy.train(False)

    print("[record_bc_isaac_episode] building env (viewer on; num_envs=1) ...", flush=True)
    env = IsaacGymBulbEnv(
        isaacgym_envs_path=ig_path,
        num_envs=1,
        sim_device=sim_dev,
        rl_device=rl_dev,
        graphics_device_id=args.graphics_device_id,
        headless=False,
        seed=seed,
        max_episode_length=mel,
        virtual_screen_capture=bool(args.virtual_display),
        force_render=False,
    )
    print(env, flush=True)

    frames: list[np.ndarray] = []

    obs = env.reset()
    img0 = env.capture_viewer_frame()
    if img0 is None:
        sys.exit(
            "capture_viewer_frame() returned None (no viewer). "
            "Use a machine with a display or --virtual_display."
        )
    frames.append(np.asarray(img0))

    with torch.no_grad():
        while True:
            actions = policy.act(obs, eval_mode=True, stddev=0.0, cpu=False)
            obs, _rew, dones, successes = env.step(actions)
            img = env.capture_viewer_frame()
            if img is not None:
                frames.append(np.asarray(img))
            if bool(dones[0].item()):
                ok = bool(successes[0].item())
                print(
                    f"[record_bc_isaac_episode] episode ended | success={ok} | "
                    f"frames={len(frames)}",
                    flush=True,
                )
                break

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=int(args.fps))
    print(f"[record_bc_isaac_episode] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
