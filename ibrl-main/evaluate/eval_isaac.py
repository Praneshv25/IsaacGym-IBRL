"""
Evaluation utilities for IBRL with the IsaacGym bulb-insertion environment.

Unlike the Robosuite evaluator, which creates a fresh single-env per call,
this evaluator reuses a (potentially shared) ``IsaacGymBulbEnv`` whose N
parallel environments are stepped simultaneously until ``num_episodes``
completed episodes have been collected.

Usage
-----
::

    from evaluate.eval_isaac import run_eval_isaac

    scores = run_eval_isaac(
        env=train_env,          # IsaacGymBulbEnv instance
        agent=agent,            # QAgent (use_state=True)
        num_episodes=50,
        verbose=True,
    )
    print(f"Success rate: {sum(scores)/len(scores):.2%}")
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from common_utils import ibrl_utils as utils

# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------


def run_eval_isaac(
    env,
    agent,
    num_episodes: int,
    *,
    verbose: bool = True,
    stddev: float = 0.0,
) -> List[float]:
    """Run ``num_episodes`` evaluation episodes on a vectorised IsaacGym env.

    The function resets **all** N environments at the start, then steps
    them in parallel until ``num_episodes`` episodes have completed.  Each
    completed episode contributes one binary score (``1.0`` for success,
    ``0.0`` for failure) to the returned list.

    The function is safe to call with ``num_episodes`` that is not a
    multiple of ``env.num_envs``: it simply collects results until the
    requested count is reached and ignores the surplus.

    Parameters
    ----------
    env : IsaacGymBulbEnv
        The (possibly shared) vectorised environment.  It will be fully
        reset at the start of evaluation.
    agent : QAgent
        The policy to evaluate.  Must support ``agent.act(obs, …)``.
    num_episodes : int
        How many complete episodes to collect before returning.
    verbose : bool
        Print a per-episode summary and final statistics when ``True``.
    stddev : float
        Action noise standard deviation used during evaluation.  Defaults
        to ``0.0`` (deterministic / greedy policy).

    Returns
    -------
    List[float]
        Binary success indicators, one per completed episode.  The list
        contains exactly ``num_episodes`` values.
    """
    scores: List[float] = []
    episode_rewards: List[float] = []

    # Per-env cumulative reward for the current in-flight episode
    running_reward = torch.zeros(env.num_envs, device=env.device)

    # Reset all environments and get initial observations
    obs = env.reset()

    with torch.no_grad(), utils.eval_mode(agent):
        while len(scores) < num_episodes:
            # ── act ──────────────────────────────────────────────────────
            # obs["state"] shape: (N, 14)  → dim==2 → no unsqueeze
            actions = agent.act(obs, eval_mode=True, stddev=stddev, cpu=False)

            # ── step ─────────────────────────────────────────────────────
            obs, rewards, dones, successes = env.step(actions)

            running_reward += rewards

            # ── collect results for done envs ─────────────────────────────
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids.tolist():
                if len(scores) >= num_episodes:
                    break
                success = bool(successes[idx].item())
                scores.append(float(success))
                episode_rewards.append(float(running_reward[idx].item()))
                running_reward[idx] = 0.0

                if verbose:
                    ep_num = len(scores)
                    print(
                        f"  eval episode {ep_num:3d}/{num_episodes} | "
                        f"env {idx:3d} | "
                        f"success: {success} | "
                        f"reward: {episode_rewards[-1]:.4f}"
                    )

    if verbose:
        mean_success = float(np.mean(scores))
        mean_reward = float(np.mean(episode_rewards))
        print(
            f"Eval done: {num_episodes} episodes | "
            f"success rate: {mean_success:.2%} | "
            f"mean reward: {mean_reward:.4f}"
        )

    return scores
