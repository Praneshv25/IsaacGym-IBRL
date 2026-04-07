#!/usr/bin/env python
"""
Minimal BC evaluation in IsaacGym (e.g. Python 3.8 + IsaacGym env).

Does **not** import ``common_utils``, ``rela``, ``pyrallis``, or ``bc.bc_policy``.
Only third-party needs beyond IsaacGym/manifeel: **numpy**, **torch**, **PyYAML**.

Run from ``ibrl-main``::

    python evaluate/eval_bc_isaac_min.py \\
        --checkpoint exps/bc_isaac/run1/model0.pt \\
        --num_episodes 50

Default ``--max_episode_length`` is 1000 (set ``<=0`` for YAML horizon). For an MP4
of one episode see ``evaluate/record_bc_isaac_episode.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Repo root on sys.path (only ``env/`` is imported from the project)
# ---------------------------------------------------------------------------
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# IsaacGym must be imported before torch (also enforced in isaac_gym_wrapper).
import isaacgym  # noqa: E402, F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from env.isaac_gym_wrapper import IsaacGymBulbEnv  # noqa: E402

# Must match ``bc/isaac_dataset.py`` / live env
_OBS_DIM = 14
_ACTION_DIM = 7


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


# ---------------------------------------------------------------------------
# Inlined from ``bc/bc_policy.StateBcPolicy`` (state-only BC)
# ---------------------------------------------------------------------------


@dataclass
class StateBcPolicyConfig:
    num_layer: int = 3
    hidden_dim: int = 256
    dropout: float = 0.5
    layer_norm: int = 0


class StateBcPolicy(nn.Module):
    def __init__(
        self, obs_shape: Tuple[int, ...], action_dim: int, cfg: StateBcPolicyConfig
    ):
        super().__init__()
        assert len(obs_shape) == 1
        self.cfg = cfg
        dims = [obs_shape[0]] + [cfg.hidden_dim for _ in range(cfg.num_layer)]
        layers: List[nn.Module] = []
        for i in range(cfg.num_layer):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if cfg.layer_norm == 1:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if cfg.layer_norm == 2 and (i == cfg.num_layer - 1):
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Dropout(cfg.dropout))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(obs["state"])

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        eval_mode: bool = True,
        cpu: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        assert eval_mode
        assert not self.training
        state = obs["state"]

        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True

        greedy_action = self.net(state).detach()

        if unsqueezed:
            greedy_action = greedy_action.squeeze(0)
        if cpu:
            greedy_action = greedy_action.cpu()
        return greedy_action


# ---------------------------------------------------------------------------
# Inlined from ``common_utils/py/ibrl_utils.eval_mode``
# ---------------------------------------------------------------------------


class _EvalMode:
    def __init__(self, *models: nn.Module):
        self.models = models

    def __enter__(self) -> None:
        self.prev_states: List[bool] = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args: object) -> bool:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


# ---------------------------------------------------------------------------
# Inlined from ``evaluate/eval_isaac.run_eval_isaac``
# ---------------------------------------------------------------------------


def run_bc_eval_isaac(
    env: IsaacGymBulbEnv,
    agent: StateBcPolicy,
    num_episodes: int,
    *,
    verbose: bool = True,
    stddev: float = 0.0,
    log_every: int = 200,
) -> List[float]:
    scores: List[float] = []
    episode_rewards: List[float] = []

    running_reward = torch.zeros(env.num_envs, device=env.device)
    if verbose:
        print("[eval_bc_isaac_min] env.reset() ...", flush=True)
    obs = env.reset()
    if verbose:
        print(
            f"[eval_bc_isaac_min] env.reset() done | max_episode_length={env.max_episode_length} | "
            f"num_envs={env.num_envs} | need {num_episodes} completed episodes",
            flush=True,
        )
        if log_every > 0:
            print(
                f"[eval_bc_isaac_min] progress every {log_every} sim steps "
                f"(episode lines only when an env finishes)",
                flush=True,
            )

    sim_step = 0
    with torch.no_grad(), _EvalMode(agent):
        while len(scores) < num_episodes:
            actions = agent.act(obs, eval_mode=True, stddev=stddev, cpu=False)

            obs, rewards, dones, successes = env.step(actions)
            sim_step += 1

            if verbose and log_every > 0 and sim_step % log_every == 0:
                emin = int(env._episode_step.min().item())
                emax = int(env._episode_step.max().item())
                print(
                    f"[eval_bc_isaac_min] sim_step={sim_step} | "
                    f"episodes_done={len(scores)}/{num_episodes} | "
                    f"episode_step min/max={emin}/{emax}",
                    flush=True,
                )

            running_reward += rewards

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


def run_bc_eval_isaac_vision(
    env: IsaacGymBulbEnv,
    agent: nn.Module,
    num_episodes: int,
    *,
    policy_obs_fn,
    verbose: bool = True,
    log_every: int = 200,
) -> List[float]:
    scores: List[float] = []
    episode_rewards: List[float] = []

    running_reward = torch.zeros(env.num_envs, device=env.device)
    if verbose:
        print("[eval_bc_isaac_min] env.reset() ...", flush=True)
    env.reset()
    obs = policy_obs_fn(env, agent)
    if verbose:
        print(
            f"[eval_bc_isaac_min] env.reset() done | max_episode_length={env.max_episode_length} | "
            f"num_envs={env.num_envs} | need {num_episodes} completed episodes",
            flush=True,
        )
        if log_every > 0:
            print(
                f"[eval_bc_isaac_min] progress every {log_every} sim steps "
                f"(episode lines only when an env finishes)",
                flush=True,
            )

    sim_step = 0
    with torch.no_grad(), _EvalMode(agent):
        while len(scores) < num_episodes:
            actions = agent.act(obs, eval_mode=True, cpu=False)
            _state_obs, rewards, dones, successes = env.step(actions)
            sim_step += 1

            if verbose and log_every > 0 and sim_step % log_every == 0:
                emin = int(env._episode_step.min().item())
                emax = int(env._episode_step.max().item())
                print(
                    f"[eval_bc_isaac_min] sim_step={sim_step} | "
                    f"episodes_done={len(scores)}/{num_episodes} | "
                    f"episode_step min/max={emin}/{emax}",
                    flush=True,
                )

            running_reward += rewards
            obs = policy_obs_fn(env, agent)

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


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"Expected mapping in {path}")
    return d


def _build_policy(cfg_y: dict, weight_file: str, device: torch.device) -> StateBcPolicy:
    pol = cfg_y.get("policy") or {}
    pcfg = StateBcPolicyConfig(
        num_layer=int(pol.get("num_layer", 3)),
        hidden_dim=int(pol.get("hidden_dim", 256)),
        dropout=float(pol.get("dropout", 0.5)),
        layer_norm=int(pol.get("layer_norm", 0)),
    )
    obs_dim = int(cfg_y.get("obs_dim", _OBS_DIM))
    action_dim = int(cfg_y.get("action_dim", _ACTION_DIM))
    policy = StateBcPolicy(
        obs_shape=(obs_dim,),
        action_dim=action_dim,
        cfg=pcfg,
    )
    policy.load_state_dict(torch.load(weight_file, map_location=device), strict=True)
    policy.to(device)
    policy.train(False)
    return policy


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal BC eval (IsaacGym bulb)")
    p.add_argument("--checkpoint", required=True, help="model0.pt (cfg.yaml alongside)")
    p.add_argument("--isaacgym_envs_path", default="", help="Else from cfg.yaml")
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--num_envs", type=int, default=-1)
    p.add_argument("--sim_device", default="")
    p.add_argument("--rl_device", default="")
    p.add_argument("--graphics_device_id", type=int, default=-999)
    p.add_argument("--headless", type=int, default=-1)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument(
        "--isaac_camera_policy",
        action="append",
        default=[],
        metavar="POLICY_CAM:ISAAC_KEY",
        help="Repeatable. Map training image key to Isaac obs key (vision BC).",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="Print rollout progress every N vectorized sim steps (0 disables)",
    )
    p.add_argument(
        "--max_episode_length",
        type=int,
        default=1000,
        help="Override task.rl.max_episode_length (<=0 uses YAML default)",
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

    num_envs = args.num_envs if args.num_envs > 0 else int(cfg_y.get("num_eval_envs", 16))
    sim_dev = args.sim_device or cfg_y.get("sim_device") or "cuda:0"
    rl_dev = args.rl_device or cfg_y.get("rl_device") or sim_dev
    gdev = (
        args.graphics_device_id
        if args.graphics_device_id != -999
        else int(cfg_y.get("graphics_device_id", -1))
    )
    headless = bool(args.headless) if args.headless >= 0 else bool(cfg_y.get("headless", True))
    seed = int(cfg_y.get("seed", 0)) if args.seed < 0 else args.seed

    device = torch.device(rl_dev if torch.cuda.is_available() else "cpu")
    vision = len(_image_keys_from_cfg_yaml(cfg_y)) > 0
    extra_overrides = None
    policy_obs_fn = None

    if vision:
        from evaluate.record_bc_isaac_episode import (
            _build_policy_to_isaac_map,
            _parse_cam_alias_args,
            _policy_obs_from_ig,
            _vision_hydra_overrides,
        )
        from train_bc_isaac_vis import load_bc_policy_vis

        try:
            policy, _ = load_bc_policy_vis(ckpt, str(device))
        except Exception as e:
            sys.exit(
                f"Failed to load vision BC policy (needs train_bc_isaac_vis cfg + dataset path). "
                f"Original error: {e}"
            )

        h, w = int(policy.encoder.obs_shape[1]), int(policy.encoder.obs_shape[2])
        cli_map = _parse_cam_alias_args(list(args.isaac_camera_policy))
        pol_to_isaac = _build_policy_to_isaac_map(policy.rl_cameras, (h, w), cli_map)
        isaac_obs_dims = [(pol_to_isaac[c], h, w) for c in policy.rl_cameras]
        extra_overrides = _vision_hydra_overrides(isaac_obs_dims)
        extra_overrides.append("task.env.enableCameraSensors=true")
        policy_obs_fn = lambda env_, policy_: _policy_obs_from_ig(env_, policy_, pol_to_isaac)
        print(
            f"[eval_bc_isaac_min] vision BC | rl_cameras={policy.rl_cameras} | "
            f"obs_shape=(3,{h},{w}) | policy→isaac map={pol_to_isaac}",
            flush=True,
        )
    else:
        policy = _build_policy(cfg_y, ckpt, device)

    scale_pt = os.path.join(run_dir, "action_scale.pt")
    if os.path.isfile(scale_pt):
        print(
            f"[eval_bc_isaac_min] action_scale.pt present (same as train_bc_isaac eval; "
            f"env uses policy output directly)."
        )

    mel = (
        None
        if args.max_episode_length <= 0
        else int(args.max_episode_length)
    )
    env = IsaacGymBulbEnv(
        isaacgym_envs_path=ig_path,
        num_envs=num_envs,
        sim_device=sim_dev,
        rl_device=rl_dev,
        graphics_device_id=gdev,
        headless=headless,
        seed=seed,
        max_episode_length=mel,
        extra_overrides=extra_overrides,
    )
    print(env)

    if vision:
        assert policy_obs_fn is not None
        scores = run_bc_eval_isaac_vision(
            env,
            policy,
            args.num_episodes,
            policy_obs_fn=policy_obs_fn,
            verbose=True,
            log_every=args.log_every,
        )
    else:
        scores = run_bc_eval_isaac(
            env,
            policy,
            args.num_episodes,
            verbose=True,
            stddev=0.0,
            log_every=args.log_every,
        )
    print(f"[eval_bc_isaac_min] mean success: {float(np.mean(scores)):.4f}")


if __name__ == "__main__":
    main()
