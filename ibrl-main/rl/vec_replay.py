"""
Vectorized n-step replay buffer for IBRL with N parallel environments.

Design
------
* One ``rela.Episode`` per environment tracks its in-flight episode.
* A single shared ``rela.SingleStepTransitionReplay`` stores completed
  episodes and provides uniform random sampling with n-step returns.
* When an episode terminates (``dones[i] == True``), the episode is
  finalized via ``pop_transition()``, pushed to the shared replay, and
  the tracker is immediately re-initialised with the new episode's first
  observation (which the IsaacGymBulbEnv wrapper already provides in the
  ``next_obs`` tensor for done environments).

Demo loading
------------
``add_demos_from_pkl`` reads the MarsLab ``.pkl`` transition format::

    List[{
        'episode_id': int,
        'timestep':   int,
        'obs':        {'state': ndarray(1, 7), ...},
        'action':     ndarray(1, 7),
        'reward':     ndarray(1,),
        'next_obs':   {'state': ndarray(1, 7), ...},
        'done':       ndarray(1,),
        'success':    ndarray(1,),
        'timeout':    ndarray(1,),
    }]

The demo state is 7-D (ee_pos + ee_quat).  Because the live IsaacGym env
produces a 14-D state (ee_pos + ee_quat + socket_pos + socket_quat) we
**zero-pad** the missing 7 dimensions so the demo transitions have the
same shape as live-env transitions.  The RL will learn the socket part
from live interaction.
"""

# from __future__ import annotations

import glob
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from common_utils import rela

# ---------------------------------------------------------------------------
# Dimensionality constants (must match env/isaac_gym_wrapper.py)
# ---------------------------------------------------------------------------

LIVE_STATE_DIM = 14  # ee_pos(3)+ee_quat(4)+socket_pos(3)+socket_quat(4)
DEMO_STATE_DIM = 7  # ee_pos(3)+ee_quat(4)  – what the .pkl demos carry


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _pad_state(state_7d: torch.Tensor) -> torch.Tensor:
    """Pad a 7-D demo state tensor to the 14-D live-env state.

    The extra 7 dimensions (socket_pos + socket_quat) are filled with
    zeros.  The socket quaternion identity [0,0,0,1] would be more
    semantically correct, but zero-padding is simpler and the RL can
    learn the socket information from live interaction anyway.

    Parameters
    ----------
    state_7d : Tensor of shape ``(7,)``

    Returns
    -------
    Tensor of shape ``(14,)``
    """
    padding = torch.zeros(LIVE_STATE_DIM - DEMO_STATE_DIM, dtype=state_7d.dtype)
    return torch.cat([state_7d, padding], dim=0)


# ---------------------------------------------------------------------------
# VecReplayBuffer
# ---------------------------------------------------------------------------


class VecReplayBuffer:
    """Vectorized n-step replay buffer.

    Maintains ``num_envs`` independent ``rela.Episode`` accumulators that
    share one ``rela.SingleStepTransitionReplay`` for uniform random
    sampling with n-step discount returns.

    Parameters
    ----------
    num_envs:
        Number of parallel environments (must match the training env).
    nstep:
        Number of steps for n-step return computation.
    gamma:
        Discount factor used when accumulating multi-step rewards.
    max_episode_length:
        Upper bound on episode length.  ``rela.Episode`` pre-allocates
        storage up to this size.
    replay_size:
        Maximum number of *episodes* (not transitions) stored.  Older
        episodes are evicted when the buffer is full.
    """

    def __init__(
        self,
        num_envs: int,
        nstep: int,
        gamma: float,
        max_episode_length: int,
        replay_size: int,
    ) -> None:
        self.num_envs = num_envs
        self.nstep = nstep
        self.gamma = gamma
        self.max_episode_length = max_episode_length

        # ── N independent in-flight episode accumulators ─────────────────
        self.episodes: List[rela.Episode] = [
            rela.Episode(nstep, max_episode_length, gamma) for _ in range(num_envs)
        ]

        # ── Shared storage (uniform random sampling at train time) ────────
        self.replay = rela.SingleStepTransitionReplay(
            frame_stack=1,
            n_step=nstep,
            capacity=replay_size,
            seed=1,
            prefetch=3,
            extra=0.1,
        )

        # ── Counters ──────────────────────────────────────────────────────
        self.num_episode = 0
        self.num_success = 0

        # Track which episode slots have been initialised with an obs.
        # Slots are uninitialised only before the very first call to
        # new_episodes().
        self._initialized: List[bool] = [False] * num_envs

    # ──────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def new_episodes(self, obs: Dict[str, torch.Tensor]) -> None:
        """Start fresh episodes for **all** environments.

        Must be called once after ``env.reset()`` before the first
        ``add_step()`` call, and again after evaluation episodes to
        re-synchronise the episode trackers with the newly reset env.

        Parameters
        ----------
        obs : dict with key ``"state"`` of shape ``(N, state_dim)``
            Initial observations for all N environments.
        """
        for i in range(self.num_envs):
            env_obs = {k: v[i].contiguous() for k, v in obs.items()}
            print(
                f"[ibrl] replay env {i} obs "
                + ", ".join(
                    f"{k}:shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
                    for k, v in env_obs.items()
                ),
                flush=True,
            )
            self.episodes[i].init({})
            print(f"[ibrl] replay env {i} init ok", flush=True)
            self.episodes[i].push_obs(env_obs)
            print(f"[ibrl] replay env {i} push_obs ok", flush=True)
            self._initialized[i] = True

    def reset_current_episodes(self) -> None:
        """Drop any in-flight episode trackers and mark all env slots uninitialized.

        This is used when the simulator is force-reset outside the normal
        episode termination flow (for example, after warm-up or evaluation).
        In that case the partial trajectories should not be continued, and
        calling ``new_episodes()`` on already-initialized ``rela.Episode``
        objects would trip an internal assertion.
        """
        self.episodes = [
            rela.Episode(self.nstep, self.max_episode_length, self.gamma)
            for _ in range(self.num_envs)
        ]
        self._initialized = [False] * self.num_envs

    def _start_single_episode(self, env_id: int, obs: Dict[str, torch.Tensor]) -> None:
        """Initialise the episode tracker for one env with a pre-sliced obs."""
        cpu_obs = {k: v.contiguous() for k, v in obs.items()}
        self.episodes[env_id].init({})
        self.episodes[env_id].push_obs(cpu_obs)
        self._initialized[env_id] = True

    # ──────────────────────────────────────────────────────────────────────
    # Adding transitions
    # ──────────────────────────────────────────────────────────────────────

    def add_step(
        self,
        next_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        successes: torch.Tensor,
    ) -> None:
        """Ingest one vectorised environment step into the replay.

        For each environment ``i``:

        * Push ``actions[i]``, ``rewards[i]``, ``dones[i]`` into its
          episode tracker.
        * If the episode ended (``dones[i] == True``):

          1. Finalise the episode and push it to the shared replay.
          2. Re-initialise the tracker with ``next_obs[i]``, which the
             ``IsaacGymBulbEnv`` wrapper guarantees to be the **new
             episode's initial observation** for done environments.

        * Otherwise push ``next_obs[i]`` as the continuing observation.

        Parameters
        ----------
        next_obs : dict, tensors of shape ``(N, state_dim)`` on any device.
        actions  : ``Tensor(N, action_dim)``
        rewards  : ``Tensor(N,)``
        dones    : ``Tensor(N,)`` bool
        successes: ``Tensor(N,)`` bool
        """
        for i in range(self.num_envs):
            if not self._initialized[i]:
                continue

            env_action = {"action": actions[i].detach().cpu()}
            env_reward = float(rewards[i].item())
            env_done = bool(dones[i].item())
            env_success = bool(successes[i].item())
            env_next = {k: v[i].contiguous() for k, v in next_obs.items()}

            # ── rela.Episode expects strict push order: ──────────────────
            #   push_obs → push_action → push_reward → push_terminal
            # ─────────────────────────────────────────────────────────────
            self.episodes[i].push_action(env_action)
            self.episodes[i].push_reward(env_reward)
            self.episodes[i].push_terminal(float(env_done))

            if env_done:
                # Finalise and store the completed episode
                transition = self.episodes[i].pop_transition()
                self.replay.add(transition)
                self.num_episode += 1
                if env_success:
                    self.num_success += 1

                # Start the next episode with next_obs[i] (new initial obs)
                self._start_single_episode(i, env_next)
            else:
                # Episode continues: push the next observation
                self.episodes[i].push_obs(env_next)

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────

    def sample(self, batchsize: int, device: str) -> rela.SingleStepTransition:
        """Sample a batch of n-step transitions.

        Returns a ``rela.SingleStepTransition`` whose fields are::

            .obs        – dict  {'state': Tensor(B, 14), 'prop': Tensor(B, 14)}
            .next_obs   – dict  {'state': Tensor(B, 14), 'prop': Tensor(B, 14)}
            .action     – dict  {'action': Tensor(B, 7)}
            .reward     – Tensor(B,)   (n-step accumulated, already discounted)
            .bootstrap  – Tensor(B,)   (gamma^n or 0 at episode boundary)

        These fields map directly to what ``QAgent.update()`` expects.
        """
        return self.replay.sample(batchsize, device)

    # ──────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────

    def size(self) -> int:
        """Number of complete episodes currently in the replay."""
        return self.replay.size()

    def ready(self, min_episodes: int = 1) -> bool:
        """Return True once the buffer holds at least *min_episodes*."""
        return self.size() >= min_episodes

    def __repr__(self) -> str:
        return (
            f"VecReplayBuffer("
            f"num_envs={self.num_envs}, "
            f"nstep={self.nstep}, "
            f"gamma={self.gamma}, "
            f"episodes={self.num_episode}, "
            f"successes={self.num_success})"
        )


# ---------------------------------------------------------------------------
# Demo loading from .pkl
# ---------------------------------------------------------------------------


def _resolve_pkl_files(path: str) -> List[str]:
    """Return a sorted list of .pkl paths from a single file or directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.pkl")))
        if not files:
            raise ValueError(f"No .pkl files found in directory: {path}")
        return files
    else:
        if not os.path.isfile(path):
            raise ValueError(f"Path does not exist: {path}")
        return [path]


def _load_and_merge_transitions(pkl_files: List[str]) -> List[dict]:
    """Load and merge raw transitions from one or more pkl files.

    Episode IDs are offset per file so they remain unique across the
    merged list even when individual files reuse the same IDs.
    """
    merged: List[dict] = []
    ep_id_offset = 0
    for fpath in pkl_files:
        with open(fpath, "rb") as f:
            raw: List[dict] = pickle.load(f)
        if raw:
            max_id_in_file = max(t["episode_id"] for t in raw)
            for t in raw:
                t2 = dict(t)
                t2["episode_id"] = t["episode_id"] + ep_id_offset
                merged.append(t2)
            ep_id_offset += max_id_in_file + 1
    return merged


def add_demos_from_pkl(
    replay: VecReplayBuffer,
    pkl_path: str,
    max_episodes: int = -1,
    verbose: bool = True,
    action_scale: Optional[torch.Tensor] = None,
) -> None:
    """Load offline demonstration data from MarsLab ``.pkl`` file(s) into
    the replay buffer.

    ``pkl_path`` may point to a **single .pkl file** or a **directory**.
    When a directory is given, every ``*.pkl`` inside it is loaded and
    merged (episode IDs are re-keyed to avoid collisions across files).

    The ``.pkl`` format stores a flat list of individual SARS transitions.
    This function reconstructs episode structure from the ``episode_id`` and
    ``timestep`` fields, feeds each episode through a temporary
    ``rela.Episode`` accumulator, and pushes the resulting
    ``MultiStepTransition`` directly into the replay's shared storage.

    State dimension mismatch
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Demo states are 7-D (``ee_pos`` + ``ee_quat``).  The live IsaacGym env
    produces 14-D states.  The extra 7 dimensions are **zero-padded** here
    so all tensors in the replay have a uniform shape.

    Parameters
    ----------
    replay :
        The ``VecReplayBuffer`` whose shared replay storage receives the
        demo data.
    pkl_path :
        Path to a single ``.pkl`` file **or a directory** containing
        multiple ``.pkl`` files.
    max_episodes :
        Maximum number of episodes to load in total.  ``-1`` means load all.
    verbose :
        If ``True``, print loading progress.
    action_scale : Tensor of shape ``(action_dim,)`` or ``None``
        Per-dimension scale used to normalise demo actions into the
        ``[-1, 1]`` range expected by the live env and RL actor.
        When ``None`` (default) the scale is computed automatically as
        the per-dim maximum absolute value across all demo actions
        (clamped to ≥ 1.0, so dims already in ``[-1, 1]`` are unchanged).
        Pass the ``action_scale`` attribute of an ``IsaacPklDataset``
        to share the same normalisation between the BC dataset and the
        replay buffer.
    """
    pkl_files = _resolve_pkl_files(pkl_path)

    if verbose:
        if len(pkl_files) == 1:
            print(f"Loading demos from {pkl_files[0]}")
        else:
            print(
                f"Loading demos from {len(pkl_files)} pkl files "
                f"in directory: {pkl_path}"
            )
            for f in pkl_files:
                print(f"  {os.path.basename(f)}")

    data = _load_and_merge_transitions(pkl_files)

    # ── Group transitions by episode_id, sort by timestep ────────────────
    episodes_raw: Dict[int, list] = defaultdict(list)
    for t in data:
        episodes_raw[t["episode_id"]].append(t)

    for ep_transitions in episodes_raw.values():
        ep_transitions.sort(key=lambda t: t["timestep"])

    episode_ids = sorted(episodes_raw.keys())
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]

    if verbose:
        print(f"  Episodes found: {len(episodes_raw)}, loading: {len(episode_ids)}")

    loaded = 0
    skipped = 0

    # ── Compute per-dim action scale if not explicitly provided ──────────
    # This brings demo actions into [-1, 1] so they match the live RL
    # actor's output range.  The env applies pos_action_scale / rot_action_scale
    # internally, so both demo and live actions should arrive un-scaled.
    if action_scale is None:
        _raw_actions = []
        for _ep_id in episode_ids:
            for tr in episodes_raw[_ep_id]:
                a = np.asarray(tr["action"], dtype=np.float32).squeeze()
                _raw_actions.append(a)
        if _raw_actions:
            _stacked = np.stack(_raw_actions, axis=0)  # (N, action_dim)
            _abs_max = np.abs(_stacked).max(axis=0)  # (action_dim,)
            action_scale = torch.from_numpy(
                np.maximum(_abs_max, 1.0).astype(np.float32)
            )
        else:
            action_scale = torch.ones(7, dtype=torch.float32)  # fallback

    assert action_scale is not None  # always set by the block above
    if verbose:
        print(
            f"  action_scale (demo normalisation): "
            f"{[f'{v:.4f}' for v in action_scale.tolist()]}"
        )

    for ep_id in episode_ids:
        transitions = episodes_raw[ep_id]
        ep_len = len(transitions)

        if ep_len == 0:
            skipped += 1
            continue

        # Clamp to replay's max_episode_length (rela.Episode pre-allocates)
        if ep_len > replay.max_episode_length:
            if verbose:
                print(
                    f"  Episode {ep_id}: length {ep_len} exceeds "
                    f"max_episode_length {replay.max_episode_length}. "
                    f"Truncating."
                )
            transitions = transitions[: replay.max_episode_length]
            ep_len = len(transitions)

        # ── Create a temporary episode accumulator ────────────────────────
        # We need max_seq_len >= ep_len; use ep_len + 1 as a safe margin.
        tmp_episode = rela.Episode(
            replay.nstep,
            ep_len + 1,
            replay.gamma,
        )

        # ── Helper: convert demo state array → padded 14-D tensor ────────
        def _to_state_tensor(state_arr) -> torch.Tensor:
            """Convert a numpy state array to a padded float32 CPU tensor."""
            arr = np.asarray(state_arr, dtype=np.float32)
            t = torch.from_numpy(arr).squeeze()  # squeeze batch dim if present
            if t.dim() == 0:
                t = t.unsqueeze(0)
            # Pad from DEMO_STATE_DIM to LIVE_STATE_DIM
            if t.shape[0] < LIVE_STATE_DIM:
                t = _pad_state(t)
            return t

        # ── Feed the first obs ────────────────────────────────────────────
        first_state = _to_state_tensor(transitions[0]["obs"]["state"])
        initial_obs = {"state": first_state, "prop": first_state}
        tmp_episode.init({})
        tmp_episode.push_obs(initial_obs)

        # ── Feed each transition ──────────────────────────────────────────
        for step_idx, tr in enumerate(transitions):
            action_arr = np.asarray(tr["action"], dtype=np.float32).squeeze()
            action_t = torch.from_numpy(action_arr)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)
            # Normalise to [-1, 1] so demo actions match the live RL actor's
            # output range (the env applies pos/rot scales internally).
            action_t = (action_t / action_scale).clamp(-1.0, 1.0)

            reward = float(np.asarray(tr["reward"]).squeeze())

            # Force terminal=1 at the last step of the episode so that
            # rela.Episode can finalize correctly even for truncated episodes.
            is_last = step_idx == ep_len - 1
            raw_done = bool(np.asarray(tr["done"]).squeeze())
            terminal = float(raw_done or is_last)

            tmp_episode.push_action({"action": action_t})
            tmp_episode.push_reward(reward)
            tmp_episode.push_terminal(terminal)

            if not terminal:
                # Push next obs so the episode obs sequence is complete
                next_state = _to_state_tensor(tr["next_obs"]["state"])
                next_obs_dict = {"state": next_state, "prop": next_state}
                tmp_episode.push_obs(next_obs_dict)
            else:
                # Episode ends here – do not push next obs
                break

        # ── Finalise and push to shared replay ────────────────────────────
        try:
            multi_step_transition = tmp_episode.pop_transition()
            replay.replay.add(multi_step_transition)
            replay.num_episode += 1

            # Check if the last transition was a success
            last_success = bool(
                np.asarray(
                    transitions[min(ep_len - 1, len(transitions) - 1)]["success"]
                ).squeeze()
            )
            if last_success:
                replay.num_success += 1

            loaded += 1
        except Exception as e:
            if verbose:
                print(f"  Warning: could not finalise episode {ep_id}: {e}")
            skipped += 1

    if verbose:
        print(
            f"  Loaded {loaded} demo episodes into replay "
            f"(skipped {skipped}). "
            f"Replay size: {replay.size()} episodes."
        )
