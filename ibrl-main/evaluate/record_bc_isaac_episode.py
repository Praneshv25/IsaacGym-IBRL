#!/usr/bin/env python
"""
Record **one** BC rollout in IsaacGym (TacSL bulb) to an MP4.

Supports:

* **State BC** checkpoints from ``train_bc_isaac.py`` (``StateBcPolicy``).
* **Vision BC** checkpoints from ``train_bc_isaac_vis.py`` (``BcPolicy`` + images).

Uses the Isaac Gym viewer framebuffer. Use a real display (**headless=False**), or
``--virtual_display`` (requires ``pyvirtualdisplay``) on headless servers.

Vision runs need the **dataset path** in ``cfg.yaml`` (same as training) so shapes
and ``BcPolicy`` can be rebuilt. Default Isaac mapping: policy camera ``wrist`` at
256×256 uses sim tensor ``wrist_2`` (matches ``TacSLTaskBulb.yaml``). Override with
``--isaac_camera_policy wrist:wrist_2`` (repeatable).

Run from ``ibrl-main``::

    python evaluate/record_bc_isaac_episode.py \\
        --checkpoint exps/bc_isaac/run1/model0.pt \\
        --out episode.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaacgym  # noqa: E402, F401

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.isaac_gym_wrapper import OBS_KEYS, IsaacGymBulbEnv  # noqa: E402
from evaluate.eval_bc_isaac_min import _build_policy, _load_cfg  # noqa: E402


def _image_keys_from_cfg_yaml(cfg_y: dict) -> List[str]:
    ds = cfg_y.get("dataset") or {}
    if not isinstance(ds, dict):
        return []
    ik = ds.get("image_keys") or []
    if isinstance(ik, str) and ik.strip():
        return [ik.strip()]
    if isinstance(ik, list):
        return [str(x).strip() for x in ik if str(x).strip()]
    csv = (ds.get("image_keys_csv") or "").strip()
    if csv:
        return [k.strip() for k in csv.split(",") if k.strip()]
    return []


def _vision_hydra_overrides(
    isaac_obs_dims: List[Tuple[str, int, int]],
) -> List[str]:
    """Enable RGB observations and register ``obsDims`` for each Isaac camera key."""
    ovr = [
        "task.env.use_camera_obs=true",
        "task.env.use_camera=true",
        "task.env.use_isaac_gym_tactile=false",
        "task.env.use_shear_force=false",
        "task.env.use_tactile_field_obs=false",
    ]
    seen = set()
    for name, h, w in isaac_obs_dims:
        if name in seen:
            continue
        seen.add(name)
        ovr.append(f"task.env.obsDims.{name}=[{int(h)},{int(w)},3]")
    return ovr


def _default_policy_to_isaac_cam(
    policy_cam: str, img_h: int, img_w: int
) -> str:
    """Map training camera name to an Isaac ``obs_dict`` / ``obsDims`` key."""
    if policy_cam == "wrist" and img_h == 256 and img_w == 256:
        return "wrist_2"
    return policy_cam


def _parse_cam_alias_args(specs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in specs:
        if ":" not in s:
            sys.exit(f"--isaac_camera_policy expects policy_cam:isaac_cam, got {s!r}")
        a, b = s.split(":", 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            sys.exit(f"Invalid --isaac_camera_policy {s!r}")
        out[a] = b
    return out


def _build_policy_to_isaac_map(
    rl_cameras: List[str],
    obs_shape_hw: Tuple[int, int],
    cli_aliases: Dict[str, str],
) -> Dict[str, str]:
    h, w = obs_shape_hw
    m: Dict[str, str] = {}
    for k in rl_cameras:
        m[k] = cli_aliases.get(k, _default_policy_to_isaac_cam(k, h, w))
    return m


def _state_vec_from_ig(env: IsaacGymBulbEnv) -> torch.Tensor:
    parts = [env.ig_env.obs_dict[k] for k in OBS_KEYS]
    return torch.cat(parts, dim=-1).float()


def _policy_obs_from_ig(
    env: IsaacGymBulbEnv,
    policy: torch.nn.Module,
    policy_to_isaac: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """Build the obs dict for ``BcPolicy.act`` from ``ig_env.obs_dict``."""
    from bc.bc_policy import BcPolicy  # local import

    assert isinstance(policy, BcPolicy)
    ig = env.ig_env
    dev = torch.device(env.device)
    out: Dict[str, torch.Tensor] = {}

    if policy.cfg.use_prop:
        out["prop"] = _state_vec_from_ig(env).to(dev)

    for pol_cam in policy.rl_cameras:
        ik = policy_to_isaac[pol_cam]
        if ik not in ig.obs_dict:
            sys.exit(
                f"Isaac obs_dict has no key {ik!r} (policy camera {pol_cam!r}). "
                f"Keys include: {sorted(ig.obs_dict.keys())}. "
                f"Fix --isaac_camera_policy or Hydra obsDims/camera_configs."
            )
        raw = ig.obs_dict[ik]
        if raw.dim() != 4:
            sys.exit(f"Expected {ik} (N,H,W,C), got shape {tuple(raw.shape)}")
        x = raw.permute(0, 3, 1, 2).contiguous().float()
        if float(x.max()) <= 1.01:
            x = x * 255.0
        out[pol_cam] = x.to(dev)

    return out


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
    p.add_argument(
        "--isaac_camera_policy",
        action="append",
        default=[],
        metavar="POLICY_CAM:ISAAC_KEY",
        help="Repeatable. Map training image key to Isaac obs key (e.g. wrist:wrist_2).",
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
    image_keys = _image_keys_from_cfg_yaml(cfg_y)
    vision = len(image_keys) > 0
    action_scale: Optional[torch.Tensor] = None

    if vision:
        from train_bc_isaac_vis import load_bc_policy_vis

        try:
            policy, action_scale = load_bc_policy_vis(ckpt, str(device))
        except Exception as e:
            sys.exit(
                f"Failed to load vision BC policy (needs train_bc_isaac_vis cfg + dataset path). "
                f"Original error: {e}"
            )
        policy.train(False)
        h, w = int(policy.encoder.obs_shape[1]), int(policy.encoder.obs_shape[2])
        cli_map = _parse_cam_alias_args(list(args.isaac_camera_policy))
        pol_to_isaac = _build_policy_to_isaac_map(policy.rl_cameras, (h, w), cli_map)
        isaac_obs_dims = [(pol_to_isaac[c], h, w) for c in policy.rl_cameras]
        extra = _vision_hydra_overrides(isaac_obs_dims)
        print(
            f"[record_bc_isaac_episode] vision BC | rl_cameras={policy.rl_cameras} | "
            f"obs_shape=(3,{h},{w}) | policy→isaac map={pol_to_isaac}",
            flush=True,
        )
    else:
        policy = _build_policy(cfg_y, ckpt, device)
        policy.train(False)
        extra = None
        pol_to_isaac = {}

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
        extra_overrides=extra,
    )
    print(env, flush=True)

    frames: list[np.ndarray] = []

    if vision:
        from bc.bc_policy import BcPolicy

        assert isinstance(policy, BcPolicy)
        env.reset()
        obs = _policy_obs_from_ig(env, policy, pol_to_isaac)
    else:
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
            if vision:
                from bc.bc_policy import BcPolicy

                assert isinstance(policy, BcPolicy)
                actions = policy.act(obs, eval_mode=True, cpu=False)
            else:
                actions = policy.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            if action_scale is not None:
                sc = action_scale.to(actions.device, dtype=actions.dtype).view(1, -1)
                actions = actions * sc

            obs, _rew, dones, successes = env.step(actions)

            if vision:
                obs = _policy_obs_from_ig(env, policy, pol_to_isaac)

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
