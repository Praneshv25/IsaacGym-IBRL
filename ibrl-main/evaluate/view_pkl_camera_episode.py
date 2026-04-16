#!/usr/bin/env python3
"""
Save one episode's stored camera frames from a MarsLab Isaac ``.pkl`` to an MP4.

This is an offline viewer for the data inside the pickle itself. It does not
replay the episode in Isaac; it simply writes the recorded observation images
(``obs[camera_key]``) in timestep order.

Examples::

    python evaluate/view_pkl_camera_episode.py \
      --pkl /path/to/2026-02-10-13-50-16_transitions.pkl \
      --out wrist_episode.mp4

    python evaluate/view_pkl_camera_episode.py \
      --pkl /path/to/transitions.pkl \
      --camera_key wrist \
      --episode_index 0 \
      --fps 20 \
      --out ep0.mp4
"""

import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np


def _to_u8_hwc(arr: object) -> np.ndarray:
    x = np.asarray(arr)
    while x.ndim > 3 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    if x.ndim != 3:
        raise ValueError(f"expected image with 3 dims after squeeze, got shape {x.shape}")

    # Convert CHW -> HWC if needed.
    if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    if np.issubdtype(x.dtype, np.integer):
        return np.clip(x, 0, 255).astype(np.uint8)

    xf = x.astype(np.float32)
    if float(xf.max()) <= 1.01 and float(xf.min()) >= -0.01:
        xf = xf * 255.0
    else:
        lo = float(xf.min())
        hi = float(xf.max())
        if hi > lo:
            xf = (xf - lo) / (hi - lo) * 255.0
    return np.clip(xf, 0.0, 255.0).astype(np.uint8)


def _load_transitions(path: str) -> List[dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected list of transitions, got {type(data)}")
    return data


def _group_episodes(transitions: List[dict]) -> List[List[dict]]:
    eps: Dict[int, List[dict]] = defaultdict(list)
    for tr in transitions:
        eps[int(tr["episode_id"])].append(tr)
    episode_ids = sorted(eps.keys())
    out = []
    for ep_id in episode_ids:
        seq = sorted(eps[ep_id], key=lambda t: int(t["timestep"]))
        out.append(seq)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Write one episode's stored camera frames from a .pkl to MP4")
    p.add_argument("--pkl", required=True, help="Path to one MarsLab transitions .pkl file")
    p.add_argument("--out", default="pkl_episode.mp4", help="Output MP4 path")
    p.add_argument("--camera_key", default="wrist", help="Observation camera key to render")
    p.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="0-based index into sorted episode_id values in this file",
    )
    p.add_argument("--fps", type=int, default=20, help="Output video FPS")
    p.add_argument("--max_frames", type=int, default=-1, help="Optional frame cap (-1 = all)")
    args = p.parse_args()

    pkl_path = os.path.abspath(os.path.expanduser(args.pkl))
    out_path = os.path.abspath(os.path.expanduser(args.out))
    transitions = _load_transitions(pkl_path)
    episodes = _group_episodes(transitions)
    if not episodes:
        raise SystemExit(f"No episodes found in {pkl_path}")
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise SystemExit(
            f"episode_index {args.episode_index} out of range; file has {len(episodes)} episodes"
        )

    episode = episodes[args.episode_index]
    sample_obs = episode[0].get("obs", {})
    if args.camera_key not in sample_obs:
        raise SystemExit(
            f"camera_key {args.camera_key!r} not found in obs keys {sorted(sample_obs.keys())}"
        )

    frames: List[np.ndarray] = []
    for tr in episode:
        frame = _to_u8_hwc(tr["obs"][args.camera_key])
        frames.append(frame)
        if args.max_frames > 0 and len(frames) >= args.max_frames:
            break

    if not frames:
        raise SystemExit("No frames extracted")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps)

    ep_id = int(episode[0]["episode_id"])
    print(
        f"[view_pkl_camera_episode] wrote {out_path} | "
        f"camera={args.camera_key} | episode_id={ep_id} | frames={len(frames)}"
    )


if __name__ == "__main__":
    main()
