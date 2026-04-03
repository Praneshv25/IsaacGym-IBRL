#!/usr/bin/env python
"""
PyBullet + OpenCV rollout of the bulb-insertion **state BC** policy (Franka Panda).

Visual sanity check that a checkpoint moves the arm; physics / frames differ from
IsaacGym ``TacSLTaskBulb``, but action scaling matches ``TacSLTaskBulb.yaml`` /
``_apply_actions_as_ctrl_targets``. Dim 7 (gripper) drives PyBullet ``panda_finger*``
joints (revolute or prismatic) within each joint’s URDF limits; arm IK uses
``jointIndices`` so fingers are not part of the IK solve.

Dependencies::

    pip install pybullet

Run from ``ibrl-main``::

    python evaluate/viz_bc_pybullet_franka.py \\
        --checkpoint exps/bc_isaac/shard9/model0.pt

Replay a headless Isaac rollout (``evaluate/dump_isaac_state_rollout.py``) without loading a policy::

    python evaluate/viz_bc_pybullet_franka.py --replay_npz rollout.npz

Uses ``--renderer tiny`` by default: hardware OpenGL in ``DIRECT`` mode often shows a half
white / half gray image on macOS. The camera buffer may also be **width×height** or **flat**;
the script reshapes to **height×width** for OpenCV (wrong layout looks like vertical bands).

The first policy step can pause several seconds on CPU inverse kinematics.

If ``model0.pt`` is missing locally, copy it next to ``cfg.yaml`` from your training machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

# ---------------------------------------------------------------------------
# Inlined from ``evaluate/eval_bc_isaac_min`` (state-only BC, no Isaac import)
# ---------------------------------------------------------------------------

_OBS_DIM = 14
_ACTION_DIM = 7

# TacSLTaskBulb.yaml (same as live Isaac env scaling for pose deltas)
_POS_ACTION_SCALE = np.array([0.01, 0.01, 0.01], dtype=np.float64)
_ROT_ACTION_SCALE = np.array([0.05, 0.05, 0.05], dtype=np.float64)
_CLAMP_ROT_THRESH = 1e-6

# Default socket pad from ``bc/isaac_dataset.py`` / TacSLTaskBulb.yaml
_SOCKET_POS = np.array([0.5, 0.0, 0.02], dtype=np.float64)
_SOCKET_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # xyzw

# Approximate Isaac ``randomize.franka_arm_initial_dof_pos``
_INITIAL_ARM_Q = np.array(
    [0.0002, -0.1591, 0.0005, -2.0966, -0.0002, 1.9335, 0.7855], dtype=np.float64
)


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


# ---------------------------------------------------------------------------
# Math (Isaac-style quaternions: xyzw)
# ---------------------------------------------------------------------------


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    half = 0.5 * float(angle)
    s = np.sin(half)
    a = axis.astype(np.float64)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(half)], dtype=np.float64)


def _apply_isaac_style_delta(
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match ``TacSLTaskBulb._apply_actions_as_ctrl_targets`` pose part (do_scale=True)."""
    pos_actions = action[0:3] * _POS_ACTION_SCALE
    target_pos = ee_pos + pos_actions

    rot_actions = action[3:6] * _ROT_ACTION_SCALE
    ang = float(np.linalg.norm(rot_actions))
    if ang <= _CLAMP_ROT_THRESH:
        dq = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = rot_actions / ang
        dq = _quat_from_angle_axis(ang, axis)
    target_quat = _quat_mul(dq, ee_quat)
    return target_pos, target_quat


def _policy_a6_to_finger_command(
    a6: float,
    lower: float,
    upper: float,
) -> float:
    """Map BC policy dim 6 in [-1, 1] to a finger joint position.

    ``(a6 + 1) / 2`` is *closedness* in [0, 1] (matches demos / tanh output).
    PyBullet Franka fingers are usually [lower, upper] with upper = more open (e.g. 0.04).
    """
    t = float(np.clip((a6 + 1.0) * 0.5, 0.0, 1.0))
    return float(lower + (upper - lower) * (1.0 - t))


def _pybullet_camera_to_bgr_u8(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Turn PyBullet ``getCameraImage`` RGB(A) into H×W BGR uint8 for OpenCV.

    PyBullet often returns either a **flat** buffer (``width * height * 4``) or a
    **width-major** array with shape ``(width, height, 4)``. OpenCV expects
    ``(height, width, channels)``; getting this wrong produces vertical bands
    (e.g. white strip + gray).
    """
    a = np.asarray(rgb)
    if a.ndim == 1:
        if a.size == width * height * 4:
            a = a.reshape((height, width, 4))
        elif a.size == width * height * 3:
            a = a.reshape((height, width, 3))
        else:
            raise ValueError(
                f"Unexpected flat camera buffer size {a.size} for {width}x{height}"
            )
    elif a.ndim == 3:
        r0, r1 = int(a.shape[0]), int(a.shape[1])
        # PyBullet often uses first dim = width, second = height (W×H layout).
        # OpenCV expects H×W. Only swap when non-square (square W×H == H×W is ambiguous).
        if width != height and (r0, r1) == (width, height):
            a = np.transpose(a, (1, 0, 2))
        elif (r0, r1) != (height, width):
            raise ValueError(
                f"Unexpected camera shape {a.shape} for requested {width}x{height}"
            )
    else:
        raise ValueError(f"Unexpected camera ndim={a.ndim}")

    if a.shape[2] == 4:
        x = a[..., :4]
    else:
        x = a[..., :3]

    if x.dtype != np.uint8:
        xf = x.astype(np.float64)
        if xf.max() <= 1.0 + 1e-6:
            xf = np.clip(xf, 0.0, 1.0) * 255.0
        else:
            xf = np.clip(xf, 0.0, 255.0)
        x = xf.astype(np.uint8)

    if x.shape[2] == 4:
        return cv2.cvtColor(x, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# PyBullet env
# ---------------------------------------------------------------------------


def _pb_apply_action_one_step(
    bullet,
    robot: int,
    ee_link: int,
    arm_indices: List[int],
    finger_indices: List[int],
    finger_limits: List[Tuple[float, float]],
    a: np.ndarray,
    *,
    phys_steps: int,
    ik_max_iter: int,
    ik_resid: float,
    gripper_force: float,
) -> None:
    """Apply one Isaac-style BC action (7-D) and step PyBullet ``phys_steps`` times."""
    ls = bullet.getLinkState(robot, ee_link, computeForwardKinematics=True)
    ee_pos = np.array(ls[4], dtype=np.float64)
    ee_quat = np.array(ls[5], dtype=np.float64)
    tgt_pos, tgt_quat = _apply_isaac_style_delta(ee_pos, ee_quat, a)

    try:
        q_arm = bullet.calculateInverseKinematics(
            robot,
            ee_link,
            targetPosition=tgt_pos.tolist(),
            targetOrientation=tgt_quat.tolist(),
            jointIndices=arm_indices,
            maxNumIterations=int(ik_max_iter),
            residualThreshold=float(ik_resid),
        )
        q_arm = [float(x) for x in q_arm]
        if len(q_arm) != len(arm_indices):
            raise ValueError("IK length mismatch")
    except (TypeError, ValueError):
        q_full = bullet.calculateInverseKinematics(
            robot,
            ee_link,
            targetPosition=tgt_pos.tolist(),
            targetOrientation=tgt_quat.tolist(),
            maxNumIterations=int(ik_max_iter),
            residualThreshold=float(ik_resid),
        )
        q_arm = [float(q_full[j]) for j in arm_indices]

    for k, j in enumerate(arm_indices):
        bullet.setJointMotorControl2(
            robot,
            j,
            bullet.POSITION_CONTROL,
            targetPosition=float(q_arm[k]),
            force=500.0,
        )
    for j, (lo, hi) in zip(finger_indices, finger_limits):
        finger_target = _policy_a6_to_finger_command(a[6], lo, hi)
        bullet.setJointMotorControl2(
            robot,
            j,
            bullet.POSITION_CONTROL,
            targetPosition=finger_target,
            force=float(gripper_force),
        )
    for _ in range(phys_steps):
        bullet.stepSimulation()


def _pb_seed_pose_from_state0(
    bullet,
    robot: int,
    ee_link: int,
    arm_indices: List[int],
    state0: np.ndarray,
    *,
    finger_indices: List[int],
    finger_limits: List[Tuple[float, float]],
    ik_max_iter: int,
    ik_resid: float,
    phys_steps: int,
) -> None:
    """Align arm EE to first Isaac observation (pos+quat from 14-D state)."""
    s = state0.astype(np.float64)
    tgt_pos = s[0:3]
    tgt_quat = s[3:7]
    try:
        q_arm = bullet.calculateInverseKinematics(
            robot,
            ee_link,
            targetPosition=tgt_pos.tolist(),
            targetOrientation=tgt_quat.tolist(),
            jointIndices=arm_indices,
            maxNumIterations=int(ik_max_iter),
            residualThreshold=float(ik_resid),
        )
        q_arm = [float(x) for x in q_arm]
    except (TypeError, ValueError):
        q_full = bullet.calculateInverseKinematics(
            robot,
            ee_link,
            targetPosition=tgt_pos.tolist(),
            targetOrientation=tgt_quat.tolist(),
            maxNumIterations=int(ik_max_iter),
            residualThreshold=float(ik_resid),
        )
        q_arm = [float(q_full[j]) for j in arm_indices]
    for k, j in enumerate(arm_indices):
        bullet.resetJointState(robot, j, float(q_arm[k]))
    for j, (lo, hi) in zip(finger_indices, finger_limits):
        bullet.resetJointState(robot, j, hi)
    for _ in range(max(phys_steps * 4, 8)):
        bullet.stepSimulation()


def _find_link_index(body: int, name: str) -> int:
    """Joint index whose *child* link matches ``name`` (``getLinkState`` uses this index)."""
    import pybullet as p

    want = name.encode()
    n = p.getNumJoints(body)
    for j in range(n):
        info = p.getJointInfo(body, j)
        if info[12] == want:
            return j
    raise RuntimeError(f"Child link {name!r} not found on body {body}")


def main() -> None:
    import pybullet as p
    import pybullet_data

    ap = argparse.ArgumentParser(description="BC policy PyBullet + OpenCV (Franka)")
    ap.add_argument(
        "--checkpoint",
        default="",
        help="Path to model0.pt (default: exps/bc_isaac/shard9/model0.pt under ibrl-main)",
    )
    ap.add_argument("--steps", type=int, default=10_000, help="Max policy steps")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument(
        "--phys_steps",
        type=int,
        default=4,
        help="PyBullet substeps per policy step (cf. controlFrequencyInv)",
    )
    ap.add_argument(
        "--renderer",
        choices=("tiny", "opengl"),
        default="tiny",
        help="Camera renderer. Use 'tiny' in DIRECT mode (default): OpenGL often draws "
        "garbage white/gray on macOS and does not raise errors.",
    )
    ap.add_argument(
        "--ik_max_iter",
        type=int,
        default=25,
        help="IK iterations per step (lower = faster, less precise).",
    )
    ap.add_argument(
        "--ik_resid",
        type=float,
        default=1e-3,
        help="IK residual threshold (looser = faster).",
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Use PyBullet GUI window in addition to OpenCV (helps debug if camera buffer looks wrong).",
    )
    ap.add_argument(
        "--gripper_force",
        type=float,
        default=120.0,
        help="Max force for each finger joint (position control).",
    )
    ap.add_argument(
        "--replay_npz",
        default="",
        help="Replay ``states``/``actions`` from dump_isaac_state_rollout.py (no policy).",
    )
    args = ap.parse_args()

    replay_path = (args.replay_npz or "").strip()
    policy = None
    device = torch.device("cpu")

    if replay_path:
        replay_path = os.path.abspath(os.path.expanduser(replay_path))
        if not os.path.isfile(replay_path):
            sys.exit(f"--replay_npz not found: {replay_path}")
    else:
        ckpt = args.checkpoint.strip()
        if not ckpt:
            ckpt = os.path.join(_REPO_ROOT, "exps/bc_isaac/shard9/model0.pt")
        ckpt = os.path.abspath(os.path.expanduser(ckpt))
        if not os.path.isfile(ckpt):
            sys.exit(f"Checkpoint not found: {ckpt}")

        run_dir = os.path.dirname(ckpt)
        cfg_path = os.path.join(run_dir, "cfg.yaml")
        if not os.path.isfile(cfg_path):
            sys.exit(f"cfg.yaml not found next to checkpoint: {cfg_path}")

        cfg_y = _load_cfg(cfg_path)
        device = torch.device(
            args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
        )
        policy = _build_policy(cfg_y, ckpt, device)

    replay_states: Optional[np.ndarray] = None
    replay_actions: Optional[np.ndarray] = None
    if replay_path:
        data = np.load(replay_path, allow_pickle=True)
        replay_states = np.asarray(data["states"], dtype=np.float32)
        replay_actions = np.asarray(data["actions"], dtype=np.float32)
        if replay_states.ndim != 2 or replay_states.shape[1] != _OBS_DIM:
            sys.exit(
                f"replay states expected (T+1, {_OBS_DIM}), got {replay_states.shape}"
            )
        if replay_actions.ndim != 2 or replay_actions.shape[1] != _ACTION_DIM:
            sys.exit(
                f"replay actions expected (T, {_ACTION_DIM}), got {replay_actions.shape}"
            )
        if replay_states.shape[0] != replay_actions.shape[0] + 1:
            sys.exit(
                f"expected states.shape[0] == actions.shape[0] + 1, got "
                f"{replay_states.shape[0]} vs {replay_actions.shape[0]}"
            )

    # DIRECT + software camera works headless; GUI optional for debugging.
    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    if args.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
        basePosition=[0.0, 0.0, 0.0],
    )

    # Visual “socket” (thin box) at default pad pose
    half_ext = [0.04, 0.04, 0.005]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=[0.9, 0.85, 0.2, 1.0])
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=_SOCKET_POS.tolist(),
    )

    n_j = p.getNumJoints(robot)
    arm_indices: List[int] = []
    finger_entries: List[Tuple[str, int]] = []
    for j in range(n_j):
        info = p.getJointInfo(robot, j)
        name = info[1].decode("utf-8")
        jtype = info[2]
        if "panda_finger" in name and jtype in (
            p.JOINT_REVOLUTE,
            p.JOINT_PRISMATIC,
        ):
            finger_entries.append((name, j))
        elif name.startswith("panda_joint") and jtype == p.JOINT_REVOLUTE:
            arm_indices.append(j)
    arm_indices = sorted(arm_indices)[:7]
    if len(arm_indices) != 7:
        sys.exit(f"Expected 7 arm joints, got {arm_indices}")

    finger_entries.sort(key=lambda x: x[0])
    finger_indices = [j for _, j in finger_entries]
    finger_limits: List[Tuple[float, float]] = []
    for j in finger_indices:
        info = p.getJointInfo(robot, j)
        lo, hi = float(info[8]), float(info[9])
        if lo > hi:
            lo, hi = hi, lo
        finger_limits.append((lo, hi))

    if not finger_indices:
        print(
            "[viz_bc_pybullet_franka] WARNING: no panda_finger joints found "
            "(revolute/prismatic); gripper commands are ignored.",
            flush=True,
        )

    try:
        ee_link = _find_link_index(robot, "panda_hand")
    except RuntimeError:
        ee_link = _find_link_index(robot, "panda_link8")

    if replay_states is not None:
        _pb_seed_pose_from_state0(
            p,
            robot,
            ee_link,
            arm_indices,
            replay_states[0],
            finger_indices=finger_indices,
            finger_limits=finger_limits,
            ik_max_iter=args.ik_max_iter,
            ik_resid=args.ik_resid,
            phys_steps=args.phys_steps,
        )
    else:
        for k, j in enumerate(arm_indices):
            p.resetJointState(robot, j, float(_INITIAL_ARM_Q[k]))
        # Start gripper open (pybullet_data Panda: upper limit ≈ open)
        for j, (lo, hi) in zip(finger_indices, finger_limits):
            p.resetJointState(robot, j, hi)

    # Camera pose similar to TacSLTaskBulb viewer-ish (looking at table center)
    cam_target = [0.5, 0.0, 0.1]
    cam_up = [0.0, 0.0, 1.0]
    cam_dist = 1.2
    cam_yaw = 0.0
    cam_pitch = -25.0
    width, height = 640, 480

    def _cam_pos() -> List[float]:
        yaw_r = np.radians(cam_yaw)
        pitch_r = np.radians(cam_pitch)
        dx = cam_dist * np.cos(pitch_r) * np.cos(yaw_r)
        dy = cam_dist * np.cos(pitch_r) * np.sin(yaw_r)
        dz = cam_dist * np.sin(pitch_r)
        return [cam_target[0] + dx, cam_target[1] + dy, cam_target[2] + dz]

    renderer_flag = (
        p.ER_TINY_RENDERER if args.renderer == "tiny" else p.ER_BULLET_HARDWARE_OPENGL
    )

    if replay_states is not None:
        print(
            f"[viz_bc_pybullet_franka] REPLAY {replay_path}\n"
            f"  T={len(replay_actions)} | renderer={args.renderer} | ik_iter={args.ik_max_iter}",
            flush=True,
        )
    else:
        print(
            f"[viz_bc_pybullet_franka] checkpoint={ckpt}\n"
            f"  device={device} | ee_link={ee_link} | steps={args.steps} | "
            f"renderer={args.renderer} | ik_iter={args.ik_max_iter}",
            flush=True,
        )
    print(
        f"  finger joints: {finger_indices} | limits={finger_limits} | "
        f"gripper_force={args.gripper_force}",
        flush=True,
    )
    print(
        "Starting loop (first inverse-kinematics solve can take a few seconds on CPU)…",
        flush=True,
    )

    cv2.namedWindow("bc_pybullet_franka (q to quit)", cv2.WINDOW_AUTOSIZE)
    try:
        cv2.startWindowThread()
    except Exception:
        pass

    max_steps = len(replay_actions) if replay_actions is not None else args.steps
    step_i = 0
    while step_i < max_steps:
        if replay_actions is not None:
            a = replay_actions[step_i].astype(np.float64)
        else:
            ls = p.getLinkState(robot, ee_link, computeForwardKinematics=True)
            ee_pos = np.array(ls[4], dtype=np.float64)
            ee_quat = np.array(ls[5], dtype=np.float64)

            state_np = np.concatenate(
                [ee_pos, ee_quat, _SOCKET_POS, _SOCKET_QUAT]
            ).astype(np.float32)
            obs = {"state": torch.from_numpy(state_np).to(device)}
            assert policy is not None
            with torch.no_grad():
                act_t = policy.act(obs, eval_mode=True, cpu=True)
            a = act_t.numpy().astype(np.float64)

        _pb_apply_action_one_step(
            p,
            robot,
            ee_link,
            arm_indices,
            finger_indices,
            finger_limits,
            a,
            phys_steps=args.phys_steps,
            ik_max_iter=args.ik_max_iter,
            ik_resid=args.ik_resid,
            gripper_force=args.gripper_force,
        )

        cam = _cam_pos()
        vm = p.computeViewMatrix(cam, cam_target, cam_up)
        pm = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width) / float(height), nearVal=0.05, farVal=10.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=vm,
            projectionMatrix=pm,
            renderer=renderer_flag,
        )
        bgr = _pybullet_camera_to_bgr_u8(rgb, width, height)
        if finger_indices:
            f0 = _policy_a6_to_finger_command(
                a[6], finger_limits[0][0], finger_limits[0][1]
            )
            hud = (
                f"step {step_i} | |a|={float(np.linalg.norm(a)):.3f} | finger_cmd={f0:.4f}"
            )
        else:
            hud = f"step {step_i} | |a|={float(np.linalg.norm(a)):.3f}"
        cv2.putText(
            bgr,
            hud,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("bc_pybullet_franka (q to quit)", bgr)
        if step_i == 0:
            raw = np.asarray(rgb)
            print(
                f"First frame rendered. Raw camera: shape={raw.shape} dtype={raw.dtype}",
                flush=True,
            )
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        step_i += 1

    cv2.destroyAllWindows()
    p.disconnect(cid)


if __name__ == "__main__":
    main()
