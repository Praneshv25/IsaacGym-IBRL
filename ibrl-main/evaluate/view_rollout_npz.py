#!/usr/bin/env python3
"""Save rollout camera frames from an .npz dump to an MP4."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Write rollout frames from .npz to MP4")
    parser.add_argument("--npz", required=True, help="Path to rollout .npz")
    parser.add_argument("--key", default="wrist", help="Frame array key in the .npz")
    parser.add_argument("--out", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    data = np.load(npz_path, allow_pickle=False)
    if args.key not in data:
        raise KeyError(f"{args.key!r} not found in {npz_path}; keys={sorted(data.files)}")

    frames = np.asarray(data[args.key])
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected frames shaped (T,H,W,C) with C in (1,3,4), got {frames.shape}"
        )

    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif frames.shape[-1] == 4:
        frames = frames[..., :3]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, list(frames), fps=int(args.fps))
    print(f"saved_video={out_path}")


if __name__ == "__main__":
    main()
