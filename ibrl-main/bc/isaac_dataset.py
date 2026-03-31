"""
bc/isaac_dataset.py
===================
Dataset for Behavioral Cloning training from MarsLab .pkl offline
transition files collected on the IsaacGym bulb-insertion task.

State dimension mismatch & padding
------------------------------------
Demo states are 7-D  (``ee_pos`` (3) + ``ee_quat`` (4)), while the live
``IsaacGymBulbEnv`` produces 14-D states
(+ ``socket_pos`` (3) + ``socket_quat`` (4)).

To keep BC and RL using the same observation format we **pad** the demo
state with the *default socket pose* taken directly from
``TacSLTaskBulb.yaml``::

    socket_pos_xyz_initial : [0.5, 0.0, 0.02]
    socket_rot_initial      : [0.0, 0.0, 0.0]   → quaternion [0, 0, 0, 1]

This is more semantically correct than zero-padding because the socket is
actually near that location, so the BC policy trains on a state
distribution close to what it will see at deployment time.

The trained ``StateBcPolicy`` can then be loaded into ``train_rl_isaac.py``
via ``--bc_policy <path>`` to enable the IBRL action-selection mechanism.

Public API
----------
``path`` may point to a **single .pkl file** or a **directory**.  When a
directory is given, every ``*.pkl`` file inside it (non-recursive) is
loaded and their transitions are merged into one flat dataset::

    # single file
    cfg     = IsaacDatasetConfig(path="demos.pkl")

    # whole folder
    cfg     = IsaacDatasetConfig(path="path/to/pkl_folder")

    dataset = IsaacPklDataset(cfg)

    batch = dataset.sample_bc(256, "cuda")
    # batch.obs    → {"state": Tensor(256, 14), "prop": Tensor(256, 14)}
    # batch.action → {"action": Tensor(256, 7)}

    # properties
    dataset.obs_shape     # (14,)
    dataset.action_dim    # 7
    dataset.num_steps     # total transitions across all files
    dataset.num_episodes  # total episodes across all files
"""

# from __future__ import annotations

import glob
import os
import pickle
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants  (must stay in sync with env/isaac_gym_wrapper.py)
# ---------------------------------------------------------------------------

DEMO_STATE_DIM: int = 7  # ee_pos(3) + ee_quat(4)  – what .pkl files contain
LIVE_STATE_DIM: int = 14  # + socket_pos(3) + socket_quat(4) – live env format
ACTION_DIM: int = 7  # 6-DoF task-space delta + gripper width

# Default socket pose from TacSLTaskBulb.yaml
#   socket_pos_xyz_initial : [0.5, 0.0, 0.02]
#   socket_rot_initial     : [0.0, 0.0, 0.0]  → xyzw quaternion [0, 0, 0, 1]
_DEFAULT_SOCKET_PAD: torch.Tensor = torch.tensor(
    [
        0.5,
        0.0,
        0.02,  # socket_pos
        0.0,
        0.0,
        0.0,
        1.0,
    ],  # socket_quat  (identity)
    dtype=torch.float32,
)

# Named tuple returned by sample_bc – mirrors the interface used by
# StateBcPolicy.loss() and RobomimicDataset.sample_bc()
Batch = namedtuple("Batch", ["obs", "action"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IsaacDatasetConfig:
    """Configuration for ``IsaacPklDataset``.

    Attributes
    ----------
    path:
        Path to either a single ``.pkl`` file **or a directory**.
        When a directory is given, every ``*.pkl`` file inside it
        (non-recursive, sorted alphabetically) is loaded and merged.
    max_episodes:
        Maximum number of episodes to load in total across all files.
        ``-1`` loads all.
    max_len:
        Truncate episodes longer than this many steps.  ``-1`` means no
        truncation.
    use_default_socket_pad:
        If ``True`` (default), pad the 7-D demo state with the default
        socket pose ``[0.5, 0, 0.02, 0, 0, 0, 1]``.
        If ``False``, pad with zeros.
    normalize_actions:
        If ``True`` (default), normalise the demo actions to ``[-1, 1]``
        per dimension using each dimension's maximum absolute value
        (clamped to ≥ 1.0, so dims already in ``[-1, 1]`` are unchanged).
        The resulting scale vector is stored in
        ``IsaacPklDataset.action_scale`` so a downstream consumer can
        invert the normalisation if needed (e.g. to pass raw actions to
        the env).  Set to ``False`` to store raw un-normalised actions.
    action_scale_path:
        If non-empty, load per-dimension scales from this ``.pt`` file
        (shape ``(7,)``) instead of computing max-abs from the current
        data. Use for **sequential** training on multiple data folders with
        ``normalize_actions=True``: pass the first run's
        ``action_scale.pt`` so every shard uses the same normalisation as
        the checkpoint you warm-start from. Requires ``normalize_actions``
        to be ``True``.
    """

    path: str = ""
    max_episodes: int = -1
    max_len: int = -1
    use_default_socket_pad: bool = True
    normalize_actions: bool = True
    action_scale_path: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class IsaacPklDataset:
    """Flat transition dataset loaded from one or more MarsLab ``.pkl`` files.

    Each ``.pkl`` file contains a ``List[dict]`` where each dict is one
    SARS transition::

        {
            'episode_id': int,
            'timestep':   int,
            'obs':        {'state': ndarray(1, 7), ...},
            'action':     ndarray(1, 7),
            'reward':     ndarray(1,),
            'next_obs':   {'state': ndarray(1, 7), ...},
            'done':       ndarray(1,),
            'success':    ndarray(1,),
            'timeout':    ndarray(1,),
        }

    ``cfg.path`` may be a single ``.pkl`` file or a directory; in the
    latter case every ``*.pkl`` inside the directory is loaded and merged.
    Episode IDs are re-keyed per file to avoid collisions across files.

    All transitions are stored in a flat ``_entries`` list; episode
    boundaries are only used for statistics and optional filtering.

    The observation stored in each entry is the **current** obs (not
    next_obs), padded from 7-D to 14-D.

    Parameters
    ----------
    cfg : IsaacDatasetConfig
    """

    # ── IBRL / StateBcPolicy compatibility ───────────────────────────────────
    obs_shape: tuple
    action_dim: int
    num_steps: int
    num_episodes: int
    # Per-dim scale used to normalise actions when cfg.normalize_actions=True.
    # Shape: (action_dim,).  Set to ones when normalisation is disabled so
    # callers can always compute  raw_action = stored_action * action_scale
    # without branching.
    action_scale: torch.Tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_pkl_files(path: str) -> List[str]:
        """Return a sorted list of .pkl paths from a file or directory."""
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.pkl")))
            if not files:
                raise ValueError(f"No .pkl files found in directory: {path}")
            return files
        else:
            if not os.path.isfile(path):
                raise ValueError(f"Path does not exist: {path}")
            return [path]

    @staticmethod
    def _load_raw_transitions(pkl_files: List[str]) -> List[dict]:
        """Load and merge raw transitions from one or more pkl files.

        Episode IDs are offset per file so they remain unique across
        the merged list even when individual files reuse the same IDs.
        """
        merged: List[dict] = []
        ep_id_offset = 0
        for fpath in pkl_files:
            with open(fpath, "rb") as f:
                raw: List[dict] = pickle.load(f)
            # Find the max episode_id in this file so the next file's IDs
            # start above it.
            if raw:
                max_id_in_file = max(t["episode_id"] for t in raw)
                for t in raw:
                    # Shallow copy so we don't mutate the original dict.
                    t2 = dict(t)
                    t2["episode_id"] = t["episode_id"] + ep_id_offset
                    merged.append(t2)
                ep_id_offset += max_id_in_file + 1
        return merged

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, cfg: IsaacDatasetConfig) -> None:
        self.cfg = cfg

        if not cfg.path:
            raise ValueError("IsaacDatasetConfig.path must be set.")

        pkl_files = self._resolve_pkl_files(cfg.path)
        if len(pkl_files) == 1:
            print(f"[IsaacPklDataset] loading from: {pkl_files[0]}")
        else:
            print(
                f"[IsaacPklDataset] loading {len(pkl_files)} pkl files "
                f"from directory: {cfg.path}"
            )
            for f in pkl_files:
                print(f"  {os.path.basename(f)}")

        raw = self._load_raw_transitions(pkl_files)

        # ── group by (remapped) episode_id, sort each episode by timestep ─
        eps_raw: Dict[int, List[dict]] = defaultdict(list)
        for t in raw:
            eps_raw[t["episode_id"]].append(t)

        ep_ids = sorted(eps_raw.keys())
        if cfg.max_episodes > 0:
            ep_ids = ep_ids[: cfg.max_episodes]

        # ── build flat entry list ─────────────────────────────────────────
        self._entries: List[Dict[str, torch.Tensor]] = []
        episode_lens: List[int] = []
        num_success_transitions = 0
        num_success_episodes = 0

        for ep_id in ep_ids:
            transitions = sorted(eps_raw[ep_id], key=lambda t: t["timestep"])

            if cfg.max_len > 0:
                transitions = transitions[: cfg.max_len]

            ep_len = 0
            ep_has_success = False

            for tr in transitions:
                # ── observation: squeeze off the batch dim, pad to 14-D ──
                state_np = np.asarray(tr["obs"]["state"], dtype=np.float32)
                state_np = state_np.squeeze()  # (7,)
                state_7d = torch.from_numpy(state_np)
                state_14d = self._pad_state(state_7d)

                # ── action: squeeze off the batch dim ─────────────────────
                action_np = np.asarray(tr["action"], dtype=np.float32)
                action_np = action_np.squeeze()  # (7,)
                action_t = torch.from_numpy(action_np)

                self._entries.append(
                    {
                        "state": state_14d,  # (14,)
                        "action": action_t,  # (7,)
                    }
                )
                ep_len += 1

                if bool(np.asarray(tr["success"]).squeeze()):
                    num_success_transitions += 1
                    ep_has_success = True

            if ep_len > 0:
                episode_lens.append(ep_len)
                if ep_has_success:
                    num_success_episodes += 1

        # ── shape / dim metadata ──────────────────────────────────────────
        self.obs_shape = (LIVE_STATE_DIM,)
        self.action_dim = ACTION_DIM
        self.num_steps = len(self._entries)
        self.num_episodes = len(episode_lens)
        # Default: identity scale (overwritten below when num_steps > 0)
        self.action_scale = torch.ones(ACTION_DIM, dtype=torch.float32)

        # ── diagnostic print ──────────────────────────────────────────────
        print(f"  Episodes loaded    : {self.num_episodes}")
        print(f"  Total transitions  : {self.num_steps}")
        print(f"  Success transitions: {num_success_transitions}")
        print(f"  Success episodes   : {num_success_episodes} / {self.num_episodes}")
        if episode_lens:
            print(
                f"  Episode length     : "
                f"min={min(episode_lens)}, "
                f"max={max(episode_lens)}, "
                f"mean={np.mean(episode_lens):.1f}"
            )
        print(f"  obs_shape          : {self.obs_shape}")
        print(f"  action_dim         : {self.action_dim}")

        if self.num_steps > 0:
            all_actions = torch.stack([e["action"] for e in self._entries])
            scale_path = (cfg.action_scale_path or "").strip()

            if scale_path:
                if not cfg.normalize_actions:
                    raise ValueError(
                        "action_scale_path is set but normalize_actions is False; "
                        "enable normalize_actions or clear action_scale_path."
                    )
                if not os.path.isfile(scale_path):
                    raise FileNotFoundError(
                        f"action_scale_path does not exist: {scale_path}"
                    )
                loaded = torch.load(scale_path, map_location="cpu")
                if not isinstance(loaded, torch.Tensor):
                    loaded = torch.as_tensor(loaded, dtype=torch.float32)
                else:
                    loaded = loaded.to(dtype=torch.float32).clone()
                if loaded.shape != (ACTION_DIM,):
                    raise ValueError(
                        f"action_scale must have shape ({ACTION_DIM},), "
                        f"got {tuple(loaded.shape)}"
                    )
                self.action_scale = loaded
            else:
                # Compute per-dim max-abs scale.  Clamp to ≥ 1.0 so dims that
                # are already inside [-1, 1] (e.g. position) stay unchanged.
                abs_max: torch.Tensor = all_actions.abs().max(dim=0).values.clamp(
                    min=1.0
                )
                self.action_scale = abs_max  # shape: (action_dim,)

            if cfg.normalize_actions:
                # Normalise stored actions in-place: each dim → [-1, 1]
                scale = self.action_scale  # (action_dim,)
                for entry in self._entries:
                    entry["action"] = entry["action"] / scale
                scale_tag = (
                    f"loaded from {scale_path}" if scale_path else "data max-abs"
                )
                print(
                    f"  action normalisation : ENABLED\n"
                    f"  action_scale ({scale_tag}) : "
                    f"{[f'{v:.4f}' for v in self.action_scale.tolist()]}"
                )
            else:
                print("  action normalisation : DISABLED (raw demo actions stored)")

            # Re-stack after possible in-place normalisation for range report
            all_actions_final = torch.stack([e["action"] for e in self._entries])
            for i in range(self.action_dim):
                lo = all_actions_final[:, i].min().item()
                hi = all_actions_final[:, i].max().item()
                print(f"  action dim {i}        : [{lo:.4f}, {hi:.4f}]")

        if self.num_steps == 0:
            raise RuntimeError(
                f"No transitions were loaded from {cfg.path}. "
                "Check that the file is non-empty and the path is correct."
            )

    # ──────────────────────────────────────────────────────────────────────
    # State padding
    # ──────────────────────────────────────────────────────────────────────

    def _pad_state(self, state_7d: torch.Tensor) -> torch.Tensor:
        """Extend a 7-D EE-only state to the 14-D live-env state format.

        Appends either the default socket pose or zeros depending on
        ``cfg.use_default_socket_pad``.

        Parameters
        ----------
        state_7d : Tensor of shape ``(7,)``

        Returns
        -------
        Tensor of shape ``(14,)``
        """
        if self.cfg.use_default_socket_pad:
            pad = _DEFAULT_SOCKET_PAD
        else:
            pad = torch.zeros(LIVE_STATE_DIM - DEMO_STATE_DIM, dtype=torch.float32)
        return torch.cat([state_7d, pad], dim=0)

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────

    def sample_bc(self, batchsize: int, device: str) -> Batch:
        """Sample a random mini-batch of ``(obs, action)`` pairs.

        Sampling is done **with replacement** whenever the requested
        batch size exceeds the dataset size (which can happen with very
        small demo sets).

        Parameters
        ----------
        batchsize : int
        device    : str – PyTorch device string, e.g. ``"cuda"``

        Returns
        -------
        Batch
            ``batch.obs``    – ``{"state": Tensor(B, 14), "prop": Tensor(B, 14)}``
            ``batch.action`` – ``{"action": Tensor(B, 7)}``
        """
        replace = len(self._entries) < batchsize
        indices = np.random.choice(len(self._entries), batchsize, replace=replace)

        states = torch.stack([self._entries[i]["state"] for i in indices]).to(device)
        actions = torch.stack([self._entries[i]["action"] for i in indices]).to(device)

        obs = {"state": states, "prop": states}
        action = {"action": actions}
        return Batch(obs=obs, action=action)

    # ──────────────────────────────────────────────────────────────────────
    # Dunder helpers
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.num_steps

    def __repr__(self) -> str:
        return (
            f"IsaacPklDataset("
            f"episodes={self.num_episodes}, "
            f"steps={self.num_steps}, "
            f"obs_shape={self.obs_shape}, "
            f"action_dim={self.action_dim})"
        )
