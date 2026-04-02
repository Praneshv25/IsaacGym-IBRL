"""
train_bc_isaac.py
=================
Behavioral Cloning training for the IsaacGym bulb-insertion task.

Trains a ``StateBcPolicy`` (3-layer MLP) to clone actions from the
MarsLab ``.pkl`` offline demonstration dataset.

State representation
--------------------
Demo states are 7-D (ee_pos + ee_quat).  The ``IsaacPklDataset`` pads
them to 14-D by appending the default socket pose so the BC policy's
input format matches the live ``IsaacGymBulbEnv``.

IBRL integration
----------------
The saved checkpoint can be loaded into ``train_rl_isaac.py`` via
``--bc_policy <path/to/model_best.pt>`` to enable the IBRL
action-selection mechanism (Q-guided BC + RL action selection).

Usage
-----
::

    # minimal
    python train_bc_isaac.py \\
        --dataset.path "MarsLab Offline RL Feb Transitions.pkl" \\
        --save_dir exps/bc_isaac/run1

    # with live-env evaluation every epoch (requires IsaacGym GPU)
    python train_bc_isaac.py \\
        --dataset.path "MarsLab Offline RL Feb Transitions.pkl" \\
        --eval_with_env 1 \\
        --isaacgym_envs_path ../manifeel-isaacgymenvs-tacsl-manifeel-rl \\
        --save_dir exps/bc_isaac/run1

    # sequential training on data shards (folder of ``*.pkl`` per shard)
    # Shard 1: train and note ``model0.pt`` and ``action_scale.pt`` under save_dir.
    # Shard 2+: warm-start weights and reuse the *first* shard's action scale.
    python train_bc_isaac.py \\
        --dataset.path /path/to/shard2 \\
        --dataset.action_scale_path exps/bc_isaac/shard1/action_scale.pt \\
        --init_checkpoint exps/bc_isaac/shard1/model0.pt \\
        --save_dir exps/bc_isaac/shard2

    # Global shuffled cycling (one shard in RAM at a time; avoids sequential
    # fine-tune forgetting). Requires a global scale file, e.g. from
    # ``python tools/compute_global_action_scale.py --data_dir ... --out g.pt``::
    #
    #   python train_bc_isaac.py \\
    #       --dataset.path /path/to/all_pkls_dir \\
    #       --dataset.fixed_action_scale_path g.pt \\
    #       --val_dataset_path /path/to/val.pkl \\
    #       --shard_cycle 1 \\
    #       --save_dir exps/bc_isaac/cycle1
"""

# from __future__ import annotations

import gc
import os
import random
import sys
from dataclasses import dataclass, field, replace
from typing import List, Optional

try:
    import isaacgym  # noqa: F401 — before torch when IsaacGym is installed
except ImportError:
    pass

import numpy as np
import pyrallis
import torch
import yaml

import common_utils
from bc.bc_policy import StateBcPolicy, StateBcPolicyConfig
from bc.isaac_dataset import (
    ACTION_DIM,
    LIVE_STATE_DIM,
    IsaacDatasetConfig,
    IsaacPklDataset,
)
from common_utils import ibrl_utils as utils
from env.isaac_gym_wrapper import IsaacGymBulbEnv
from evaluate.eval_isaac import run_eval_isaac

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MainConfig(common_utils.RunConfig):
    """Top-level configuration for BC training on IsaacGym bulb-insertion."""

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset: IsaacDatasetConfig = field(default_factory=lambda: IsaacDatasetConfig())

    # ── policy architecture ───────────────────────────────────────────────────
    policy: StateBcPolicyConfig = field(default_factory=lambda: StateBcPolicyConfig())

    # ── training ──────────────────────────────────────────────────────────────
    seed: int = 1
    num_epoch: int = 100
    epoch_len: int = 1_000  # gradient steps per epoch
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 5.0
    weight_decay: float = 0.0

    # Warm-start from a prior BC run (sequential shards). Path to a ``.pt``
    # checkpoint (e.g. ``.../model0.pt``). Must match architecture: 14-D state,
    # 7-D action, same ``policy`` settings. Empty or ``none`` = train from scratch.
    init_checkpoint: str = ""

    # ── live-env evaluation (optional, requires IsaacGym + GPU) ──────────────
    # Set eval_with_env=1 to run success-rate evaluation after each epoch.
    # When disabled (default) the checkpoint is saved by *lowest training loss*.
    eval_with_env: int = 0
    num_eval_episodes: int = 20
    # IsaacGym env settings (only used when eval_with_env=1)
    isaacgym_envs_path: str = "../manifeel-isaacgymenvs-tacsl-manifeel-rl"
    num_eval_envs: int = 16
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    graphics_device_id: int = -1
    headless: bool = True

    # ── logging / saving ──────────────────────────────────────────────────────
    save_dir: str = "exps/bc_isaac/run1"
    use_wb: int = 0
    # Save an extra named checkpoint every N epochs (−1 = disabled)
    save_per: int = -1

    # Directory of ``*.pkl`` shards; each epoch shuffles shards and trains
    # ``epoch_len // num_shards`` steps per file (one shard loaded at a time).
    # With ``normalize_actions``, set ``dataset.fixed_action_scale_path`` or
    # ``dataset.action_scale_path`` to a global ``.pt`` from
    # ``tools/compute_global_action_scale.py``.
    shard_cycle: int = 0
    val_dataset_path: str = ""
    val_num_batches: int = 20

    def __post_init__(self) -> None:
        # Derived read-only fields (informational only)
        self.obs_dim: int = LIVE_STATE_DIM  # 14
        self.action_dim: int = ACTION_DIM  # 7


# ---------------------------------------------------------------------------
# Model I/O helpers  (used by train_rl_isaac.py to load a trained BC policy)
# ---------------------------------------------------------------------------


def load_bc_policy(
    weight_file: str, device: str
) -> tuple["StateBcPolicy", Optional[torch.Tensor]]:
    """Load a saved ``StateBcPolicy`` checkpoint.

    The config YAML written alongside the checkpoint is used to
    reconstruct the policy architecture, so no architecture arguments
    need to be passed explicitly.

    An ``action_scale.pt`` file is expected alongside the checkpoint.
    If found, it is loaded and returned so the caller can invert the
    action normalisation when passing actions to the environment.

    Parameters
    ----------
    weight_file : str
        Path to a ``.pt`` file saved by ``TopkSaver``.
    device : str
        PyTorch device string, e.g. ``"cuda:0"``.

    Returns
    -------
    policy : StateBcPolicy
        The loaded policy in *eval* mode on *device*.
    action_scale : Tensor of shape ``(action_dim,)`` or ``None``
        Per-dimension normalisation scale saved alongside the checkpoint,
        or ``None`` if the file does not exist (e.g. legacy checkpoints
        trained without normalisation).
    """
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, "cfg.yaml")
    cfg: MainConfig = pyrallis.load(MainConfig, open(cfg_path))  # type: ignore

    policy = StateBcPolicy(
        obs_shape=(cfg.obs_dim,),
        action_dim=cfg.action_dim,
        cfg=cfg.policy,
    )
    policy.load_state_dict(torch.load(weight_file, map_location=device))
    policy.to(device)
    policy.train(False)

    # Load optional action_scale (saved during training)
    scale_path = os.path.join(run_folder, "action_scale.pt")
    action_scale: Optional[torch.Tensor] = None
    if os.path.exists(scale_path):
        _scale: torch.Tensor = torch.load(scale_path, map_location="cpu")
        action_scale = _scale
        print(
            f"  [load_bc_policy] action_scale loaded from {scale_path}: "
            f"{[f'{v:.4f}' for v in _scale.tolist()]}"
        )
    else:
        print(
            f"  [load_bc_policy] no action_scale.pt found at {scale_path}; "
            f"assuming actions are already in [-1, 1]."
        )

    return policy, action_scale


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """Manages dataset, policy, optimiser and the training loop."""

    def __init__(self, cfg: MainConfig) -> None:
        self.cfg = cfg
        self.work_dir = cfg.save_dir
        # Populated after dataset is loaded; used to persist normalisation.
        self._action_scale: Optional[torch.Tensor] = None
        os.makedirs(self.work_dir, exist_ok=True)

        common_utils.set_all_seeds(cfg.seed)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        # ── persist config ────────────────────────────────────────────────
        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        print(common_utils.wrap_ruler("config"))
        with open(cfg.cfg_path) as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path))

        self.val_dataset: Optional[IsaacPklDataset] = None
        self._shard_paths: List[str] = []
        self._shard_val_warned: bool = False

        vdp = (cfg.val_dataset_path or "").strip()
        if vdp:
            print(common_utils.wrap_ruler("val dataset"))
            self.val_dataset = IsaacPklDataset(replace(cfg.dataset, path=vdp))

        # ── dataset ───────────────────────────────────────────────────────
        print(common_utils.wrap_ruler("dataset"))
        if cfg.shard_cycle:
            dp = cfg.dataset.path
            if not os.path.isdir(dp):
                raise ValueError(
                    "shard_cycle=1 requires dataset.path to be a directory of *.pkl files, "
                    f"not {dp!r}"
                )
            self._shard_paths = IsaacPklDataset._resolve_pkl_files(dp)
            if cfg.dataset.normalize_actions and not (cfg.dataset.action_scale_path or "").strip():
                raise ValueError(
                    "shard_cycle=1 with normalize_actions requires dataset.action_scale_path "
                    "or dataset.fixed_action_scale_path (tools/compute_global_action_scale.py)."
                )
            self.dataset = IsaacPklDataset(replace(cfg.dataset, path=self._shard_paths[0]))
        else:
            self.dataset = IsaacPklDataset(cfg.dataset)
        self._action_scale = self.dataset.action_scale

        # ── policy ────────────────────────────────────────────────────────
        print(common_utils.wrap_ruler("policy"))
        self.policy = StateBcPolicy(
            obs_shape=self.dataset.obs_shape,  # (14,)
            action_dim=self.dataset.action_dim,  # 7
            cfg=cfg.policy,
        )
        self.policy = self.policy.to("cuda" if torch.cuda.is_available() else "cpu")
        print(self.policy)
        common_utils.count_parameters(self.policy)

        ic = (cfg.init_checkpoint or "").strip()
        if ic and ic.lower() != "none":
            if not os.path.isfile(ic):
                raise FileNotFoundError(f"init_checkpoint not found: {ic}")
            device = next(self.policy.parameters()).device
            state = torch.load(ic, map_location=device)
            self.policy.load_state_dict(state, strict=True)
            print(common_utils.wrap_ruler(f"loaded init_checkpoint: {ic}"))

        # ── optimiser ─────────────────────────────────────────────────────
        if cfg.weight_decay > 0:
            self.optim = torch.optim.AdamW(
                self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            self.optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        # ── optional live-env for evaluation ─────────────────────────────
        self.eval_env: Optional[IsaacGymBulbEnv] = None
        if cfg.eval_with_env:
            print(common_utils.wrap_ruler("building eval env"))
            self.eval_env = IsaacGymBulbEnv(
                isaacgym_envs_path=cfg.isaacgym_envs_path,
                num_envs=cfg.num_eval_envs,
                sim_device=cfg.sim_device,
                rl_device=cfg.rl_device,
                graphics_device_id=cfg.graphics_device_id,
                headless=cfg.headless,
                seed=cfg.seed,
            )
            print(self.eval_env)

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def _eval_loss(self, num_batches: int = 20) -> float:
        """Estimate validation loss on randomly sampled mini-batches.

        Uses ``val_dataset_path`` when set; otherwise the current training
        dataset (merged directory or single shard).
        """
        ds = self.val_dataset if self.val_dataset is not None else self.dataset
        if ds is None:
            raise RuntimeError("_eval_loss: no dataset loaded.")
        device = next(self.policy.parameters()).device
        losses = []
        with torch.no_grad(), utils.eval_mode(self.policy):
            for _ in range(num_batches):
                batch = ds.sample_bc(self.cfg.batch_size, str(device))
                loss = self.policy.loss(batch)
                losses.append(loss.item())
        return float(np.mean(losses))

    def _eval_env_success(self) -> float:
        """Run live-env rollouts and return mean success rate."""
        assert self.eval_env is not None
        scores = run_eval_isaac(
            env=self.eval_env,
            agent=self.policy,  # type: ignore[arg-type]
            num_episodes=self.cfg.num_eval_episodes,
            verbose=False,
        )
        return float(np.mean(scores))

    # ──────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────

    def _save_action_scale(self) -> None:
        """Persist ``action_scale`` so ``load_bc_policy`` can reload it."""
        if self._action_scale is not None:
            scale_path = os.path.join(self.work_dir, "action_scale.pt")
            torch.save(self._action_scale.cpu(), scale_path)

    def train(self) -> None:
        """Run the full BC training loop.

        Each epoch consists of ``epoch_len`` gradient steps.  After each
        epoch the policy is evaluated (either via live env or training
        loss), and the best checkpoint is saved.

        Checkpoint scoring
        ~~~~~~~~~~~~~~~~~~
        * ``eval_with_env=1`` → scored by *success rate* (higher = better).
        * ``eval_with_env=0`` → scored by *−training loss* (higher = better,
          i.e. lower loss).  This means ``TopkSaver`` will keep the
          checkpoint with the lowest observed training loss.
        """
        cfg = self.cfg
        device = str(next(self.policy.parameters()).device)

        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(cfg.use_wb),
            wb_exp_name=cfg.wb_exp,
            wb_run_name=cfg.wb_run,
            wb_group_name=cfg.wb_group,
            config=self.cfg_dict,
        )
        # topk=1: keep only the single best checkpoint
        saver = common_utils.TopkSaver(self.work_dir, topk=1)
        stopwatch = common_utils.Stopwatch()

        best_score = -float("inf")
        optim_step = 0

        print(common_utils.wrap_ruler("training"))

        for epoch in range(cfg.num_epoch):
            stopwatch.reset()

            # ── gradient steps ────────────────────────────────────────────
            self.policy.train(True)
            epoch_losses = []

            if cfg.shard_cycle:
                paths = list(self._shard_paths)
                rng = random.Random(int(cfg.seed) + epoch)
                rng.shuffle(paths)
                n_shards = len(paths)
                steps_per = max(1, cfg.epoch_len // n_shards)

                for sp in paths:
                    shard_cfg = replace(cfg.dataset, path=sp)
                    ds = IsaacPklDataset(shard_cfg)
                    self.dataset = ds
                    self._action_scale = ds.action_scale

                    for _ in range(steps_per):
                        with stopwatch.time("sample"):
                            batch = ds.sample_bc(cfg.batch_size, device)

                        with stopwatch.time("train"):
                            loss = self.policy.loss(batch)

                            self.optim.zero_grad()
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.policy.parameters(), max_norm=cfg.grad_clip
                            )
                            self.optim.step()

                        epoch_losses.append(loss.item())
                        stat["train/loss"].append(loss.item())
                        stat["train/grad_norm"].append(grad_norm.item())
                        optim_step += 1

                    del ds
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if self.val_dataset is None:
                    if not self._shard_val_warned:
                        print(
                            "[train_bc_isaac] shard_cycle: val_dataset_path is empty — "
                            "val_loss uses alphabetically-first shard only. "
                            "Set val_dataset_path for a held-out validation set."
                        )
                        self._shard_val_warned = True
                    self.dataset = IsaacPklDataset(
                        replace(cfg.dataset, path=self._shard_paths[0])
                    )
            else:
                for _ in range(cfg.epoch_len):
                    with stopwatch.time("sample"):
                        batch = self.dataset.sample_bc(cfg.batch_size, device)

                    with stopwatch.time("train"):
                        loss = self.policy.loss(batch)

                        self.optim.zero_grad()
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), max_norm=cfg.grad_clip
                        )
                        self.optim.step()

                    epoch_losses.append(loss.item())
                    stat["train/loss"].append(loss.item())
                    stat["train/grad_norm"].append(grad_norm.item())
                    optim_step += 1

            mean_loss = float(np.mean(epoch_losses))
            epoch_time = stopwatch.elapsed_time_since_reset
            stat["other/epoch_time"].append(epoch_time)
            stat["other/speed"].append(cfg.epoch_len / max(epoch_time, 1e-6))

            # ── evaluation & scoring ───────────────────────────────────────
            self.policy.train(False)

            if cfg.eval_with_env:
                with stopwatch.time("eval"):
                    score = self._eval_env_success()
                stat["score/success_rate"].append(score)
                score_label = f"success_rate={score:.4f}"
            else:
                # Use negative loss so TopkSaver (higher = better) saves the
                # checkpoint with the lowest training loss.
                val_loss = self._eval_loss(num_batches=cfg.val_num_batches)
                score = -val_loss
                stat["score/neg_loss"].append(score)
                score_label = f"val_loss={val_loss:.6f}"

            best_score = max(best_score, score)
            stat["score/best"].append(best_score)

            # ── save ──────────────────────────────────────────────────────
            saved = saver.save(self.policy.state_dict(), score, save_latest=True)
            # Persist action_scale alongside every checkpoint save.
            # (Idempotent – writes the same file each epoch.)
            self._save_action_scale()

            # Periodic named checkpoint
            if cfg.save_per > 0 and (epoch + 1) % cfg.save_per == 0:
                saver.save(
                    self.policy.state_dict(),
                    score,
                    force_save_name=f"epoch{epoch + 1}",
                )

            # ── summary ───────────────────────────────────────────────────
            stat.summary(epoch, reset=True)
            stopwatch.summary(reset=True)
            shard_info = f" | shards={len(self._shard_paths)}" if cfg.shard_cycle else ""
            print(
                f"epoch {epoch + 1:4d}/{cfg.num_epoch} | "
                f"loss={mean_loss:.6f} | {score_label} | "
                f"best={best_score:.4f} | saved={saved}{shard_info}"
            )
            print(common_utils.get_mem_usage())

        # ── final evaluation on best checkpoint ───────────────────────────
        print(common_utils.wrap_ruler("final evaluation (best checkpoint)"))
        best_ckpt = saver.get_best_model()
        if best_ckpt is not None:
            self.policy.load_state_dict(torch.load(best_ckpt, map_location=device))
            self.policy.train(False)

            if cfg.eval_with_env:
                final_score = self._eval_env_success()
                print(f"Final success rate (best ckpt): {final_score:.4f}")
                stat["score/final_success_rate"].append(final_score)
            else:
                final_loss = self._eval_loss(num_batches=50)
                print(f"Final val loss (best ckpt): {final_loss:.6f}")
                stat["score/final_val_loss"].append(final_loss)

            stat.summary(cfg.num_epoch, reset=True)
        else:
            print("Warning: no checkpoint found – saver may have been empty.")

        print(f"Training complete.  Checkpoints in: {self.work_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import rich.traceback

    rich.traceback.install()
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    cfg: MainConfig = pyrallis.parse(config_class=MainConfig)  # type: ignore

    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
