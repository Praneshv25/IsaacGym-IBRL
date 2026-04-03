#!/usr/bin/env python
"""
Headless IsaacGym (TacSL bulb): run BC for one episode and save ``states`` /
``actions`` to a compressed ``.npz`` for offline visualization (e.g. PyBullet).

Optional **Isaac wrist (or other sim) RGB** via GPU camera sensors. Sim cameras
need a **non-negative** ``graphics_device_id`` in ``create_sim`` (state-only
cfgs often use ``-1``, which breaks ``create_camera_sensor``); this script bumps
to the GPU index from ``sim_device`` (e.g. ``cuda:0`` → ``0``) when saving RGB.
Use ``--graphics_device_id N`` to set explicitly. If headless capture still fails,
try ``--no-headless`` (viewer on, like ``record_bc_isaac_episode.py``).

Use ``--save_wrist_camera`` for **state-only** checkpoints, or a **vision BC**
checkpoint (``cfg.yaml`` with ``dataset.image_keys``) to enable cameras automatically.

Replay::

    python evaluate/viz_bc_pybullet_franka.py --replay_npz rollout.npz

Run from ``ibrl-main``::

    python evaluate/dump_isaac_state_rollout.py \\
        --checkpoint exps/bc_isaac/shard9/model0.pt \\
        --out rollout.npz \\
        --save_wrist_camera

Vision BC (``train_bc_isaac_vis.py``)::

    python evaluate/dump_isaac_state_rollout.py \\
        --checkpoint exps/bc_isaac_vis/run1/model0.pt \\
        --out rollout.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaacgym  # noqa: E402, F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.isaac_gym_wrapper import IsaacGymBulbEnv  # noqa: E402
from evaluate.eval_bc_isaac_min import _build_policy, _load_cfg  # noqa: E402
from evaluate.record_bc_isaac_episode import (  # noqa: E402
    _build_policy_to_isaac_map,
    _image_keys_from_cfg_yaml,
    _parse_cam_alias_args,
    _policy_obs_from_ig,
    _vision_hydra_overrides,
)


def _isaac_frame_u8_hwc(env: IsaacGymBulbEnv, isaac_key: str) -> np.ndarray:
    """One env’s RGB from ``obs_dict`` as uint8 ``(H, W, 3)``."""
    ig = env.ig_env
    if isaac_key not in ig.obs_dict:
        sys.exit(
            f"Isaac obs_dict has no key {isaac_key!r}. Keys: {sorted(ig.obs_dict.keys())}"
        )
    raw = ig.obs_dict[isaac_key]
    if raw.dim() != 4:
        sys.exit(f"Expected {isaac_key} (N,H,W,C), got {tuple(raw.shape)}")
    x = raw[0].detach().cpu().numpy()
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    if float(xf.max()) <= 1.01:
        xf = xf * 255.0
    return np.clip(xf, 0.0, 255.0).astype(np.uint8)


def _rgb_save_name(policy_cam: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in policy_cam)
    return f"{safe}_rgb"


def _graphics_device_for_sim_cameras(graphics_device_id: int, sim_device: str) -> int:
    """Isaac ``create_sim(..., graphics_device=...)`` must be a real GPU index for
    ``create_camera_sensor`` / ``get_camera_image_gpu_tensor``. ``-1`` (common in
    state-only cfgs) yields camera handle -1 and None tensors.
    """
    if graphics_device_id >= 0:
        return graphics_device_id
    s = (sim_device or "").strip().lower()
    if "cuda" in s or "gpu" in s:
        if ":" in s:
            try:
                return int(s.rsplit(":", 1)[-1])
            except ValueError:
                return 0
        return 0
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Dump one BC Isaac rollout to .npz")
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
        help="GPU for graphics / camera render; -999 uses cfg.yaml",
    )
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument(
        "--save_wrist_camera",
        "--save-wrist-camera",
        action="store_true",
        help="Enable Isaac sim cameras and save RGB (default key wrist_2 @ 256²; "
        "use --isaac_cam_key / --cam_h / --cam_w to change). For state-only BC.",
    )
    p.add_argument(
        "--isaac_cam_key",
        default="wrist_2",
        help="Isaac obs_dict key for RGB when using --save_wrist_camera (state BC).",
    )
    p.add_argument("--cam_h", type=int, default=256)
    p.add_argument("--cam_w", type=int, default=256)
    p.add_argument(
        "--isaac_camera_policy",
        action="append",
        default=[],
        metavar="POLICY_CAM:ISAAC_KEY",
        help="Repeatable. Map training image key to Isaac obs key (vision BC).",
    )
    p.add_argument(
        "--force_render",
        action="store_true",
        help="VecTask force_render=True (slower; try if camera buffers stay stale).",
    )
    p.add_argument(
        "--no-headless",
        "--no_headless",
        action="store_true",
        help="Create the Gym viewer (like record_bc_isaac_episode). Use if sim cameras "
        "still fail headless on your driver.",
    )
    args = p.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")
    run_dir = os.path.dirname(ckpt)
    cfg_path = os.path.join(run_dir, "cfg.yaml")
    if not os.path.isfile(cfg_path):
        sys.exit(f"cfg.yaml not found: {cfg_path}")

    cfg_y = _load_cfg(cfg_path)
    image_keys = _image_keys_from_cfg_yaml(cfg_y)
    vision = len(image_keys) > 0

    if vision and args.save_wrist_camera:
        print(
            "[dump_isaac_state_rollout] note: vision cfg already enables cameras; "
            "--save_wrist_camera is redundant.",
            flush=True,
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

    want_rgb = vision or args.save_wrist_camera
    headless = not bool(args.no_headless)
    if want_rgb:
        gdev_resolved = _graphics_device_for_sim_cameras(gdev, sim_dev)
        if gdev_resolved != gdev:
            print(
                "[dump_isaac_state_rollout] GPU sim cameras need graphics_device_id>=0 in "
                f"create_sim; using {gdev_resolved} (was {gdev}). Override with "
                "--graphics_device_id N.",
                flush=True,
            )
        gdev = gdev_resolved

    device = torch.device(rl_dev if torch.cuda.is_available() else "cpu")
    action_scale: Optional[torch.Tensor] = None
    extra_overrides: Optional[List[str]] = None
    pol_to_isaac: dict = {}
    save_rgb_name: str = "wrist_rgb"
    isaac_key_for_rgb: str = args.isaac_cam_key.strip() or "wrist_2"

    if vision:
        from train_bc_isaac_vis import load_bc_policy_vis  # noqa: WPS433

        try:
            policy, action_scale = load_bc_policy_vis(ckpt, str(device))
        except Exception as e:
            sys.exit(
                "Failed to load vision BC policy (needs train_bc_isaac_vis cfg + dataset path). "
                f"Original error: {e}"
            )
        policy.train(False)
        from bc.bc_policy import BcPolicy  # noqa: WPS433

        assert isinstance(policy, BcPolicy)
        h, w = int(policy.encoder.obs_shape[1]), int(policy.encoder.obs_shape[2])
        cli_map = _parse_cam_alias_args(list(args.isaac_camera_policy))
        pol_to_isaac = _build_policy_to_isaac_map(policy.rl_cameras, (h, w), cli_map)
        isaac_obs_dims = [(pol_to_isaac[c], h, w) for c in policy.rl_cameras]
        extra_overrides = _vision_hydra_overrides(isaac_obs_dims)
        extra_overrides.append("task.env.enableCameraSensors=true")
        primary_cam = policy.rl_cameras[0]
        save_rgb_name = _rgb_save_name(primary_cam)
        isaac_key_for_rgb = pol_to_isaac[primary_cam]
        print(
            f"[dump_isaac_state_rollout] vision BC | rl_cameras={policy.rl_cameras} | "
            f"save array {save_rgb_name!r} ← Isaac {isaac_key_for_rgb!r} ({h}x{w})",
            flush=True,
        )
    else:
        policy = _build_policy(cfg_y, ckpt, device)
        policy.train(False)
        if args.save_wrist_camera:
            ch, cw = int(args.cam_h), int(args.cam_w)
            extra_overrides = _vision_hydra_overrides([(isaac_key_for_rgb, ch, cw)])
            extra_overrides.append("task.env.enableCameraSensors=true")
            save_rgb_name = "wrist_rgb"
            print(
                f"[dump_isaac_state_rollout] state BC + sim camera | "
                f"save {save_rgb_name!r} ← Isaac {isaac_key_for_rgb!r} ({ch}x{cw})",
                flush=True,
            )

    scale_pt = os.path.join(run_dir, "action_scale.pt")
    if os.path.isfile(scale_pt):
        action_scale = torch.load(scale_pt, map_location=device)

    env = IsaacGymBulbEnv(
        isaacgym_envs_path=ig_path,
        num_envs=1,
        sim_device=sim_dev,
        rl_device=rl_dev,
        graphics_device_id=gdev,
        headless=headless,
        seed=seed,
        max_episode_length=mel,
        extra_overrides=extra_overrides,
        force_render=bool(args.force_render),
    )

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rgb_frames: List[np.ndarray] = []

    obs = env.reset()
    states.append(obs["state"][0].detach().cpu().numpy().astype(np.float32))
    if want_rgb:
        rgb_frames.append(_isaac_frame_u8_hwc(env, isaac_key_for_rgb))

    with torch.no_grad():
        while True:
            if vision:
                from bc.bc_policy import BcPolicy  # noqa: WPS433

                assert isinstance(policy, BcPolicy)
                po = _policy_obs_from_ig(env, policy, pol_to_isaac)
                act = policy.act(po, eval_mode=True, cpu=False)
            else:
                act = policy.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            if action_scale is not None:
                act = act * action_scale.to(act.device).view(1, -1)

            obs, _r, dones, successes = env.step(act)
            actions.append(act[0].detach().cpu().numpy().astype(np.float32))
            states.append(obs["state"][0].detach().cpu().numpy().astype(np.float32))
            if want_rgb:
                rgb_frames.append(_isaac_frame_u8_hwc(env, isaac_key_for_rgb))

            if bool(dones[0].item()):
                success = bool(successes[0].item())
                break

    st = np.stack(states, axis=0)
    ac = np.stack(actions, axis=0)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    save_kw = dict(
        states=st,
        actions=ac,
        success=np.array([1 if success else 0], dtype=np.int8),
        checkpoint=np.array([ckpt], dtype=object),
    )
    if want_rgb:
        rg = np.stack(rgb_frames, axis=0)
        if rg.shape[0] != st.shape[0]:
            sys.exit(
                f"RGB length {rg.shape[0]} != states length {st.shape[0]} "
                "(camera / state desync)."
            )
        save_kw[save_rgb_name] = rg
        save_kw["isaac_rgb_key"] = np.array([isaac_key_for_rgb], dtype=object)

    np.savez_compressed(out_path, **save_kw)
    msg = (
        f"[dump_isaac_state_rollout] saved {out_path} | T={ac.shape[0]} | "
        f"success={success} | states={st.shape} actions={ac.shape}"
    )
    if want_rgb:
        msg += f" | {save_rgb_name}={save_kw[save_rgb_name].shape}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
