"""
train_bc_isaac_vis.py
=====================
Behavioral cloning on MarsLab Isaac ``.pkl`` demos using **RGB + proprio**
(``BcPolicy``: MultiViewEncoder + MLP), mirroring ``train_bc.py`` / robomimic
style training on the TacSL bulb task.

**Recommended setup (no tactile):** pass a single RGB camera key, typically
``wrist`` on MarsLab bulb pickles. Tactile tensors (e.g.
``right_tactile_camera_taxim``, ``tactile_force_field_right``) stay in the
``obs`` dict but are **ignored** unless you list them in ``image_keys``.

``dataset.image_keys`` must name keys present under each transition's
``obs``. Images are converted to uint8 ``(C, H, W)``; proprio is still the
14-D padded state (same as ``train_bc_isaac.py``).

**Live IsaacGym evaluation** (``eval_with_env=1``) is **not** supported here:
``IsaacGymBulbEnv`` is state-only. Use training / val loss, or add a
camera-enabled env wrapper separately.

IBRL: ``train_rl_isaac.py`` ``--bc_policy`` expects ``StateBcPolicy`` by
default; vision BC checkpoints are **not** drop-in unless you adapt RL for
images.

Schema reference (``MarsLab Offline RL Feb Transitions.pkl``): ``list`` of
transitions; ``obs`` includes ``state`` ``(1,7)`` float32, ``wrist``
``(1,256,256,3)`` float32 (~0–1 RGB), plus tactile fields you can omit.
``BcPolicy`` needs every listed ``image_keys`` view to share the same
``H×W`` (single ``wrist`` satisfies that).

Usage (wrist only)::

    python train_bc_isaac_vis.py \\
        --dataset.path "../MarsLab Offline RL Feb Transitions.pkl" \\
        --dataset.image_keys wrist \\
        --save_dir exps/bc_isaac_vis/run1

Optional: multiple **RGB** cameras with identical ``H×W`` (pyrallis list)::

    python train_bc_isaac_vis.py \\
        --dataset.path /path/to/data \\
        --dataset.image_keys[0] cam_a \\
        --dataset.image_keys[1] cam_b \\
        --save_dir exps/bc_isaac_vis/run1
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pyrallis
import torch
import yaml

import common_utils
from bc.bc_policy import BcPolicy, BcPolicyConfig
from bc.isaac_dataset import ACTION_DIM, LIVE_STATE_DIM, IsaacDatasetConfig, IsaacPklDataset
from common_utils import ibrl_utils as utils


@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: IsaacDatasetConfig = field(default_factory=lambda: IsaacDatasetConfig())
    policy: BcPolicyConfig = field(default_factory=lambda: BcPolicyConfig())

    seed: int = 1
    num_epoch: int = 100
    epoch_len: int = 1_000
    batch_size: int = 64
    lr: float = 1e-4
    grad_clip: float = 5.0
    weight_decay: float = 0.0

    init_checkpoint: str = ""

    # Reserved: IsaacGymBulbEnv has no camera obs for BC-style eval.
    eval_with_env: int = 0
    num_eval_episodes: int = 20
    isaacgym_envs_path: str = "../manifeel-isaacgymenvs-tacsl-manifeel-rl"
    num_eval_envs: int = 16
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    graphics_device_id: int = -1
    headless: bool = True

    save_dir: str = "exps/bc_isaac_vis/run1"
    use_wb: int = 0
    save_per: int = -1

    def __post_init__(self) -> None:
        self.obs_dim: int = LIVE_STATE_DIM
        self.action_dim: int = ACTION_DIM


def load_bc_policy_vis(
    weight_file: str, device: str
) -> Tuple[BcPolicy, Optional[torch.Tensor]]:
    """Load a ``BcPolicy`` checkpoint saved by this script."""
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, "cfg.yaml")
    cfg: MainConfig = pyrallis.load(MainConfig, open(cfg_path))  # type: ignore

    dataset = IsaacPklDataset(cfg.dataset)
    policy = BcPolicy(
        obs_shape=dataset.obs_shape,
        prop_shape=dataset.prop_shape,
        action_dim=dataset.action_dim,
        rl_cameras=dataset.rl_cameras,
        cfg=cfg.policy,
    )
    policy.load_state_dict(torch.load(weight_file, map_location=device))
    policy.to(device)
    policy.train(False)

    scale_path = os.path.join(run_folder, "action_scale.pt")
    action_scale: Optional[torch.Tensor] = None
    if os.path.exists(scale_path):
        action_scale = torch.load(scale_path, map_location="cpu")
        print(f"  [load_bc_policy_vis] action_scale from {scale_path}")

    return policy, action_scale


class Workspace:
    def __init__(self, cfg: MainConfig) -> None:
        self.cfg = cfg
        self.work_dir = cfg.save_dir
        self._action_scale: Optional[torch.Tensor] = None
        os.makedirs(self.work_dir, exist_ok=True)

        if cfg.eval_with_env:
            raise ValueError(
                "train_bc_isaac_vis: eval_with_env is not supported (IsaacGymBulbEnv "
                "has no camera observations). Set eval_with_env=0."
            )

        keys = [str(k).strip() for k in cfg.dataset.image_keys if str(k).strip()]
        if not keys:
            raise ValueError(
                "train_bc_isaac_vis: set dataset.image_keys to non-empty list of "
                "obs keys (e.g. --dataset.image_keys rgb). "
                "For state-only BC use train_bc_isaac.py."
            )

        common_utils.set_all_seeds(cfg.seed)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        print(common_utils.wrap_ruler("config"))
        with open(cfg.cfg_path) as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path))

        print(common_utils.wrap_ruler("dataset"))
        self.dataset = IsaacPklDataset(cfg.dataset)
        if not self.dataset.use_images:
            raise RuntimeError("internal: dataset.use_images expected True")

        self._action_scale = self.dataset.action_scale

        print(common_utils.wrap_ruler("policy"))
        self.policy = BcPolicy(
            obs_shape=self.dataset.obs_shape,
            prop_shape=self.dataset.prop_shape,
            action_dim=self.dataset.action_dim,
            rl_cameras=self.dataset.rl_cameras,
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
            self.policy.load_state_dict(torch.load(ic, map_location=device), strict=True)
            print(common_utils.wrap_ruler(f"loaded init_checkpoint: {ic}"))

        if cfg.weight_decay > 0:
            self.optim = torch.optim.AdamW(
                self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            self.optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def _eval_loss(self, num_batches: int = 20) -> float:
        device = next(self.policy.parameters()).device
        losses = []
        with torch.no_grad(), utils.eval_mode(self.policy):
            for _ in range(num_batches):
                batch = self.dataset.sample_bc(self.cfg.batch_size, str(device))
                loss = self.policy.loss(batch)
                losses.append(loss.item())
        return float(np.mean(losses))

    def _save_action_scale(self) -> None:
        if self._action_scale is not None:
            torch.save(self._action_scale.cpu(), os.path.join(self.work_dir, "action_scale.pt"))

    def train(self) -> None:
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
        saver = common_utils.TopkSaver(self.work_dir, topk=1)
        stopwatch = common_utils.Stopwatch()

        best_score = -float("inf")
        print(common_utils.wrap_ruler("training"))

        for epoch in range(cfg.num_epoch):
            stopwatch.reset()
            self.policy.train(True)
            epoch_losses = []

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

            mean_loss = float(np.mean(epoch_losses))
            epoch_time = stopwatch.elapsed_time_since_reset
            stat["other/epoch_time"].append(epoch_time)
            stat["other/speed"].append(cfg.epoch_len / max(epoch_time, 1e-6))

            self.policy.train(False)
            val_loss = self._eval_loss()
            score = -val_loss
            stat["score/neg_loss"].append(score)
            best_score = max(best_score, score)
            stat["score/best"].append(best_score)

            saved = saver.save(self.policy.state_dict(), score, save_latest=True)
            self._save_action_scale()

            if cfg.save_per > 0 and (epoch + 1) % cfg.save_per == 0:
                saver.save(
                    self.policy.state_dict(),
                    score,
                    force_save_name=f"epoch{epoch + 1}",
                )

            stat.summary(epoch, reset=True)
            stopwatch.summary(reset=True)
            print(
                f"epoch {epoch + 1:4d}/{cfg.num_epoch} | "
                f"loss={mean_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best={best_score:.4f} | saved={saved}"
            )
            print(common_utils.get_mem_usage())

        print(common_utils.wrap_ruler("final evaluation (best checkpoint)"))
        best_ckpt = saver.get_best_model()
        if best_ckpt is not None:
            self.policy.load_state_dict(torch.load(best_ckpt, map_location=device))
            self.policy.train(False)
            final_loss = self._eval_loss(num_batches=50)
            print(f"Final val loss (best ckpt): {final_loss:.6f}")
            stat["score/final_val_loss"].append(final_loss)
            stat.summary(cfg.num_epoch, reset=True)
        else:
            print("Warning: no checkpoint found.")

        print(f"Training complete.  Checkpoints in: {self.work_dir}")


def main() -> None:
    import rich.traceback

    rich.traceback.install()
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    cfg: MainConfig = pyrallis.parse(config_class=MainConfig)  # type: ignore
    Workspace(cfg).train()


if __name__ == "__main__":
    main()
