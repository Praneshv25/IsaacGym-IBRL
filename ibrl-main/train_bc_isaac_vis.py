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

Camera keys must match each transition's ``obs``. Prefer
``--dataset.image_keys_csv wrist`` on the CLI — pyrallis often fails to parse
``--dataset.image_keys wrist`` for ``List[str]``. Alternatively use
``--dataset.image_keys[0] wrist`` or a YAML config. Images are uint8
``(C, H, W)``; with ``--policy.use_prop 1``, proprio defaults to the same
legacy 14-D padded state as ``train_bc_isaac.py``. For cleaner vision BC on
datasets that only store EE state, prefer ``--dataset.image_prop_mode ee7``.

**Live IsaacGym evaluation** (``eval_with_env=1``) is **not** supported here:
``IsaacGymBulbEnv`` is state-only. Use training / val loss, or add a
camera-enabled env wrapper separately.

IBRL: ``train_rl_isaac.py`` ``--bc_policy`` expects ``StateBcPolicy`` by
default; vision BC checkpoints are **not** drop-in unless you adapt RL for
images.

Schema reference (``MarsLab Offline RL Feb Transitions.pkl``): ``list`` of
transitions; ``obs`` includes ``state`` ``(1,7)`` float32, ``wrist``
``(1,256,256,3)`` float32 (~0–1 RGB), plus tactile fields you can omit.
``BcPolicy`` needs every camera view to share the same ``H×W`` (single
``wrist`` satisfies that).

Examples::

    # wrist + 7-D EE proprio (recommended)
    python train_bc_isaac_vis.py \\
        --dataset.path "/path/to/data.pkl_or_folder" \\
        --dataset.image_keys_csv wrist \\
        --dataset.image_prop_mode ee7 \\
        --policy.use_prop 1 \\
        --save_dir exps/bc_isaac_vis/run1

    # shard 3+ (reuse shard1 action scale; warm-start from previous shard)
    python train_bc_isaac_vis.py \\
        --dataset.path "/path/to/shard3" \\
        --dataset.image_keys_csv wrist \\
        --policy.use_prop 1 \\
        --dataset.action_scale_path exps/bc_isaac_vis/shard1/action_scale.pt \\
        --init_checkpoint exps/bc_isaac_vis/shard2/model0.pt \\
        --save_dir exps/bc_isaac_vis/shard3

    # multiple RGB cameras (comma-separated or indexed)
    python train_bc_isaac_vis.py \\
        --dataset.path /path/to/data \\
        --dataset.image_keys_csv cam_a,cam_b \\
        --save_dir exps/bc_isaac_vis/run1

    # Shuffled shard cycling (global scale + optional held-out val pkl)::
    #
    #   python tools/compute_global_action_scale.py --data_dir /path/to/shards --out g.pt
    #   python train_bc_isaac_vis.py \\
    #       --dataset.path /path/to/shards \\
    #       --dataset.image_keys_csv wrist \\
    #       --policy.use_prop 1 \\
    #       --dataset.fixed_action_scale_path g.pt \\
    #       --val_dataset_path /path/to/val.pkl \\
    #       --shard_cycle 1 \\
    #       --save_dir exps/bc_isaac_vis/cycle1
"""

import gc
import os
import random
import sys
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyrallis
import torch
import yaml

import common_utils
from bc.bc_policy import BcPolicy, BcPolicyConfig
from bc.isaac_dataset import ACTION_DIM, LIVE_STATE_DIM, IsaacDatasetConfig, IsaacPklDataset
from bc.multiview_encoder import FuseMethod, MultiViewEncoderConfig
from networks.encoder import ResNetEncoderConfig
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

    # ``dataset.path`` = directory of ``*.pkl`` shards; each epoch shuffles shards and
    # loads one shard at a time (~``epoch_len // num_shards`` steps each). Requires
    # ``dataset.action_scale_path`` or ``dataset.fixed_action_scale_path`` when
    # ``dataset.normalize_actions`` is True (see ``tools/compute_global_action_scale.py``).
    shard_cycle: int = 0
    # Single ``.pkl`` or directory for held-out ``_eval_loss`` (same schema as training).
    val_dataset_path: str = ""
    val_num_batches: int = 20

    def __post_init__(self) -> None:
        self.obs_dim: int = LIVE_STATE_DIM
        self.action_dim: int = ACTION_DIM


def _pick_keys(d: Dict[str, Any], cls) -> Dict[str, Any]:
    """Subset *d* to dataclass *cls* field names (ignore unknown YAML keys)."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


def _resnet_encoder_cfg_from_dict(d: object) -> ResNetEncoderConfig:
    if not isinstance(d, dict):
        return ResNetEncoderConfig()
    return ResNetEncoderConfig(**_pick_keys(d, ResNetEncoderConfig))


def _multiview_encoder_cfg_from_dict(d: object) -> MultiViewEncoderConfig:
    if not isinstance(d, dict):
        return MultiViewEncoderConfig()
    d = dict(d)
    resnet_raw = d.pop("resnet", None)
    resnet = _resnet_encoder_cfg_from_dict(resnet_raw)
    fuse_raw = d.pop("fuse_method", "cat")
    if isinstance(fuse_raw, FuseMethod):
        fuse = fuse_raw
    else:
        fuse = FuseMethod(str(fuse_raw))
    rest = _pick_keys(d, MultiViewEncoderConfig)
    rest.pop("resnet", None)
    rest.pop("fuse_method", None)
    return MultiViewEncoderConfig(fuse_method=fuse, resnet=resnet, **rest)


def _bc_policy_cfg_from_dict(d: object) -> BcPolicyConfig:
    if not isinstance(d, dict):
        return BcPolicyConfig()
    d = dict(d)
    enc_raw = d.pop("encoder", None)
    encoder = (
        _multiview_encoder_cfg_from_dict(enc_raw)
        if enc_raw is not None
        else MultiViewEncoderConfig()
    )
    rest = _pick_keys(d, BcPolicyConfig)
    rest.pop("encoder", None)
    return BcPolicyConfig(encoder=encoder, **rest)


def _isaac_dataset_cfg_from_dict(d: object) -> IsaacDatasetConfig:
    if not isinstance(d, dict):
        return IsaacDatasetConfig()
    return IsaacDatasetConfig(**_pick_keys(d, IsaacDatasetConfig))


def load_main_config_from_cfg_yaml(cfg_path: str) -> MainConfig:
    """Instantiate ``MainConfig`` from ``cfg.yaml`` using PyYAML only.

    Avoids ``pyrallis.load``, which can fail on Python 3.8 with
    ``typing-inspect`` (e.g. weakref errors on ``str`` / nested fields).
    """
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: expected a YAML mapping at the top level")

    raw = dict(raw)
    dataset_raw = raw.pop("dataset", None) or {}
    policy_raw = raw.pop("policy", None) or {}
    dataset_cfg = _isaac_dataset_cfg_from_dict(dataset_raw)
    policy_cfg = _bc_policy_cfg_from_dict(policy_raw)
    top = _pick_keys(raw, MainConfig)
    return MainConfig(dataset=dataset_cfg, policy=policy_cfg, **top)


def load_bc_policy_vis(
    weight_file: str, device: str
) -> Tuple[BcPolicy, Optional[torch.Tensor]]:
    """Load a ``BcPolicy`` checkpoint saved by this script."""
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, "cfg.yaml")
    cfg = load_main_config_from_cfg_yaml(cfg_path)

    # Only need image/proprio shapes for ``BcPolicy`` construction. Loading every
    # ``*.pkl`` in a multi-shard directory can OOM or get SIGKILL; use one file
    # and one transition. Action scaling comes from ``action_scale.pt`` below,
    # not from scanning the dataset.
    lite_dataset_cfg = replace(
        cfg.dataset,
        max_pkl_files=1,
        max_episodes=1,
        max_len=1,
        normalize_actions=False,
        action_scale_path="",
        fixed_action_scale_path="",
    )
    dataset = IsaacPklDataset(lite_dataset_cfg)
    policy = BcPolicy(
        obs_shape=dataset.obs_shape,
        prop_shape=dataset.prop_shape,
        action_dim=dataset.action_dim,
        rl_cameras=dataset.rl_cameras,
        cfg=cfg.policy,
    )
    policy.prop_shape = dataset.prop_shape
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
                "train_bc_isaac_vis: set dataset.image_keys_csv (e.g. wrist) or "
                "dataset.image_keys / dataset.image_keys[0]. "
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

        self.val_dataset: Optional[IsaacPklDataset] = None
        self._shard_paths: List[str] = []
        self._shard_val_warned: bool = False

        vdp = (cfg.val_dataset_path or "").strip()
        if vdp:
            print(common_utils.wrap_ruler("val dataset"))
            self.val_dataset = IsaacPklDataset(replace(cfg.dataset, path=vdp))

        print(common_utils.wrap_ruler("dataset"))
        if cfg.shard_cycle:
            dp = cfg.dataset.path
            if not os.path.isdir(dp):
                raise ValueError(
                    "shard_cycle=1 requires dataset.path to be a directory of *.pkl files, "
                    f"not {dp!r}"
                )
            self._shard_paths = IsaacPklDataset._resolve_pkl_files(dp)
            scale_set = bool((cfg.dataset.action_scale_path or "").strip())
            if cfg.dataset.normalize_actions and not scale_set:
                raise ValueError(
                    "shard_cycle=1 with normalize_actions requires dataset.action_scale_path "
                    "or dataset.fixed_action_scale_path (run tools/compute_global_action_scale.py)."
                )
            self.dataset = IsaacPklDataset(replace(cfg.dataset, path=self._shard_paths[0]))
        else:
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
        ds = self.val_dataset if self.val_dataset is not None else self.dataset
        if ds is None:
            raise RuntimeError("_eval_loss: no dataset (train shard) loaded.")
        device = next(self.policy.parameters()).device
        losses = []
        with torch.no_grad(), utils.eval_mode(self.policy):
            for _ in range(num_batches):
                batch = ds.sample_bc(self.cfg.batch_size, str(device))
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

                    del ds
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if self.val_dataset is None:
                    if not self._shard_val_warned:
                        print(
                            "[train_bc_isaac_vis] shard_cycle: val_dataset_path is empty — "
                            "val_loss will use alphabetically-first shard only. "
                            "Set val_dataset_path to a held-out pkl for global validation."
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

            mean_loss = float(np.mean(epoch_losses))
            epoch_time = stopwatch.elapsed_time_since_reset
            stat["other/epoch_time"].append(epoch_time)
            stat["other/speed"].append(cfg.epoch_len / max(epoch_time, 1e-6))

            self.policy.train(False)
            val_loss = self._eval_loss(num_batches=cfg.val_num_batches)
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
            shard_info = f" | shards={len(self._shard_paths)}" if cfg.shard_cycle else ""
            print(
                f"epoch {epoch + 1:4d}/{cfg.num_epoch} | "
                f"loss={mean_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best={best_score:.4f} | saved={saved}{shard_info}"
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
