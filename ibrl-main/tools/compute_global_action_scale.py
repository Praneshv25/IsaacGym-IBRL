#!/usr/bin/env python3
"""
compute_global_action_scale.py
==============================
Scan a directory of MarsLab bulb ``*.pkl`` transition files and compute the
global per-dimension max-abs action scale (same rule as ``IsaacPklDataset``
with ``normalize_actions=True``).

Run from the ``ibrl-main`` directory::

    python tools/compute_global_action_scale.py \\
        --data_dir /path/to/all_shards \\
        --out global_action_scale.pt

Then train with fixed scaling, e.g.::

    python train_bc_isaac_vis.py \\
        --dataset.path /path/to/all_shards \\
        --dataset.fixed_action_scale_path global_action_scale.pt \\
        --shard_cycle 1 \\
        ...
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root: .../ibrl-main
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from bc.isaac_dataset import IsaacPklDataset, ACTION_DIM


def main() -> None:
    p = argparse.ArgumentParser(description="Global max-abs action scale over many .pkl files.")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing non-recursive *.pkl shards (same layout as training).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="global_action_scale.pt",
        help="Output path for a float32 tensor of shape (7,).",
    )
    args = p.parse_args()

    files = IsaacPklDataset._resolve_pkl_files(args.data_dir)
    print(f"Found {len(files)} file(s); computing max-abs over actions (dim={ACTION_DIM})...")
    scale = IsaacPklDataset.compute_global_max_abs_action_scale(files)
    out_path = os.path.abspath(args.out)
    torch.save(scale, out_path)
    print(f"Saved {tuple(scale.shape)} to {out_path}")
    print(f"  values: {[f'{v:.6f}' for v in scale.tolist()]}")


if __name__ == "__main__":
    main()
