"""
Pattern C: End-to-end RL (no FSM) + Curriculum for Nut-Bolt Screwing in Isaac Gym
- Vectorized envs
- PPO in pure PyTorch
- Wrist limit aware control (panda_joint7 limit)

Requires:
  - Isaac Gym (gymapi, gymtorch, gymutil)
  - torch, numpy

Run:
  python train_nut_bolt_patternC_curriculum.py --num_envs 64 --headless True
"""

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import (
    to_torch,
    quat_mul, quat_conjugate, quat_from_angle_axis,
)

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Math helpers
# -------------------------
@torch.jit.script
def orientation_error_batched(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    # desired/current: [N,4] (xyzw)
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    sign = torch.sign(q_r[:, 3:4])
    return q_r[:, 0:3] * sign

def control_ik_batched(dpose: torch.Tensor, j_eef: torch.Tensor, damping: float) -> torch.Tensor:
    """
    dpose: [N,6] or [N,6,1]
    j_eef: [N,6,7]
    returns: dq: [N,7]
    """
    if dpose.ndim == 2:
        dpose = dpose.unsqueeze(-1)  # [N,6,1]
    N = j_eef.shape[0]
    device = j_eef.device
    jT = j_eef.transpose(1, 2)  # [N,7,6]
    A = j_eef @ jT              # [N,6,6]
    I = torch.eye(6, device=device, dtype=A.dtype).unsqueeze(0).expand(N, -1, -1)
    A = A + (damping ** 2) * I
    x = torch.linalg.solve(A, dpose)       # [N,6,1]
    dq = (jT @ x).squeeze(-1)              # [N,7]
    return dq

def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """
    yaw: [K] tensor
    returns quat [K,4] in xyzw
    """
    K = yaw.shape[0]
    axis = torch.zeros((K, 3), device=yaw.device, dtype=yaw.dtype)
    axis[:, 2] = 1.0  # z-axis
    return quat_from_angle_axis(yaw, axis)


@torch.jit.script
def quat_to_yaw_xyzw(q: torch.Tensor) -> torch.Tensor:
    # q: [N,4] xyzw
    # yaw from quaternion
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # yaw = atan2(2(wz+xy), 1-2(y^2+z^2))
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(t0, t1)

def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi

# -------------------------
# PPO Network
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        # self.log_std = nn.Parameter(torch.zeros(act_dim))  # global logstd
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.5)  # std ~ 0.22
        self.v = nn.Linear(last, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        z = self.backbone(obs)
        mu = self.mu(z)
        v = self.v(z).squeeze(-1)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return mu, std, v

def gaussian_logprob(actions, mu, std):
    var = std * std
    return -0.5 * (((actions - mu) ** 2) / (var + 1e-8) + 2.0 * torch.log(std + 1e-8) + math.log(2 * math.pi)).sum(-1)

# -------------------------
# Curriculum
# -------------------------
class Curriculum:
    """
    Stage progression based on moving success rate.
      0: stabilize, nut pre-aligned, no yaw target
      1: align+press, small noise
      2: small rotation target (e.g., -30deg), nut pre-engaged
      3: large rotation (e.g., -180deg)
      4: full task (e.g., -2*pi), random friction/mass/noise
    """
    def __init__(self, device, stage=0):
        self.device = device
        self.stage = stage
        self.succ_ema = 0.0
        self.ema_alpha = 0.02
        self.promote_thresh = [0.95, 0.90, 0.85, 0.75]  # thresholds to go 0->1->2->3->4
        self.max_stage = 4

    def update(self, success_rate: float):
        self.succ_ema = (1 - self.ema_alpha) * self.succ_ema + self.ema_alpha * success_rate
        if self.stage < self.max_stage:
            if self.succ_ema > self.promote_thresh[self.stage]:
                self.stage += 1
                print(f"[CURRICULUM] promoted to stage {self.stage} | succ_ema={self.succ_ema:.3f}")

    def params(self):
        # Return stage-dependent parameters
        # if self.stage == 0:
        #     return dict(
        #         max_steps=80,
        #         yaw_target=0.0, yaw_tol=math.radians(60),
        #         init_preengaged=True,
        #         init_noise_xy=0.000,
        #         init_noise_yaw=0.0,
        #         action_scale_trans=0.008,
        #         action_scale_rot=0.05,
        #         obs_noise=0.0,
        #         need_rotation=False,
        #     )
        # if self.stage == 1:
        #     return dict(
        #         max_steps=120,
        #         yaw_target=0.0, yaw_tol=math.radians(60),
        #         init_preengaged=False,
        #         init_noise_xy=0.010,
        #         init_noise_yaw=math.radians(10),
        #         action_scale_trans=0.002,
        #         action_scale_rot=0.02,
        #         obs_noise=0.0,
        #         need_rotation=False,
        #     )
        if self.stage == 0:
            return dict(
                max_steps=250,
                yaw_target=0.0, yaw_tol=math.radians(60),
                init_preengaged=False,          # start with reaching skill
                init_noise_xy=0.02,
                init_noise_yaw=math.radians(10),
                action_scale_trans=0.01,        # 1 cm/step
                action_scale_rot=0.02,
                obs_noise=0.0,
                need_rotation=False,
            )
        if self.stage == 1:
            return dict(
                max_steps=250,
                yaw_target=0.0, yaw_tol=math.radians(60),
                init_preengaged=False,
                init_noise_xy=0.02,
                init_noise_yaw=math.radians(15),
                action_scale_trans=0.008,
                action_scale_rot=0.02,
                obs_noise=0.0,
                need_rotation=False,
            )

        if self.stage == 2:
            return dict(
                max_steps=200,
                yaw_target=-math.radians(30), yaw_tol=math.radians(10),
                init_preengaged=True,
                init_noise_xy=0.008,
                init_noise_yaw=math.radians(15),
                action_scale_trans=0.002,
                action_scale_rot=0.03,
                obs_noise=0.0,
                need_rotation=True,
            )
        if self.stage == 3:
            return dict(
                max_steps=300,
                yaw_target=-math.pi, yaw_tol=math.radians(12),
                init_preengaged=True,
                init_noise_xy=0.010,
                init_noise_yaw=math.radians(20),
                action_scale_trans=0.002,
                action_scale_rot=0.03,
                obs_noise=0.005,
                need_rotation=True,
            )
        # stage 4
        return dict(
            max_steps=450,
            yaw_target=-2.0 * math.pi, yaw_tol=math.radians(12),
            init_preengaged=False,
            init_noise_xy=0.020,
            init_noise_yaw=math.radians(30),
            action_scale_trans=0.002,
            action_scale_rot=0.03,
            obs_noise=0.010,
            need_rotation=True,
        )

# -------------------------
# Env
# -------------------------
class NutBoltScrewEnv:
    """
    Pattern C RL env:
      - policy outputs Cartesian deltas (dpose) + gripper delta
      - IK converts dpose -> joint delta
      - Wrist limit safe filter for joint7
      - Dense reward for alignment/press/rotation progress
      - Curriculum controls initialization + target yaw
    """
    def __init__(self, args):
        self.args = args

        # --------- Gym init ----------
        self.gym = gymapi.acquire_gym()
        # custom_parameters = [{"name": "--num_envs", "type": int, "default": 64, "help": "Number of environments"}]
        custom_parameters = [
            {"name": "--num_envs", "type": int, "default": 64, "help": "Number of environments"},
            {"name": "--headless", "action": "store_true", "help": "Run without viewer"},
        ]
        self.args = gymutil.parse_arguments(
            description="Pattern-C RL Nut-Bolt Screwing + Curriculum",
            custom_parameters=custom_parameters
        )

        # Force GPU pipeline if using SDF assets
        if not self.args.use_gpu or self.args.use_gpu_pipeline:
            self.args.use_gpu = True
            self.args.use_gpu_pipeline = True

        self.device = torch.device(self.args.sim_device)

        # sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 32
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise RuntimeError("This example requires PhysX")

        self.sim_params = sim_params
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        # viewer (optional)
        self.viewer = None
        if not self.args.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("Failed to create viewer")

        # assets
        asset_root = "../../assets"
        table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        self.table_dims = table_dims

        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_opts)

        bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
        bolt_opts = gymapi.AssetOptions()
        bolt_opts.fix_base_link = True
        bolt_opts.enable_gyroscopic_forces = True
        self.bolt_asset = self.gym.load_asset(self.sim, asset_root, bolt_file, bolt_opts)

        nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
        nut_opts = gymapi.AssetOptions()
        nut_opts.fix_base_link = False
        nut_opts.enable_gyroscopic_forces = True
        self.nut_asset = self.gym.load_asset(self.sim, asset_root, nut_file, nut_opts)

        franka_file = "urdf/franka_description/robots/franka_panda.urdf"
        franka_opts = gymapi.AssetOptions()
        franka_opts.armature = 0.01
        franka_opts.fix_base_link = True
        franka_opts.disable_gravity = True
        franka_opts.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(self.sim, asset_root, franka_file, franka_opts)

        # dof props
        franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_lower = to_torch(franka_dof_props["lower"], device=self.device)
        self.franka_upper = to_torch(franka_dof_props["upper"], device=self.device)

        # Position control
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)
        self.franka_dof_props = franka_dof_props

        self.num_envs = self.args.num_envs
        self.num_dofs = self.gym.get_asset_dof_count(self.franka_asset)  # 9

        # default pose
        mids = 0.3 * (self.franka_upper + self.franka_lower)
        default = torch.zeros(self.num_dofs, device=self.device)
        default[:7] = mids[:7]
        default[7:] = self.franka_upper[7:]
        self.default_dof_pos = default

        # indices for wrist joint 7 (0-based index = 6)
        self.wrist_idx = 6
        self.wrist_low = float(self.franka_lower[self.wrist_idx].item())
        self.wrist_high = float(self.franka_upper[self.wrist_idx].item())

        # link index
        link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
        self.hand_body_index_asset = link_dict["panda_hand"]

        # create envs
        self.envs = []
        self.nut_rb_idxs = []
        self.bolt_rb_idxs = []
        self.hand_rb_idxs = []

        self.nut_actor_idxs = []
        self.bolt_actor_idxs = []
        self.franka_actor_idxs = []

        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        bolt_pose = gymapi.Transform()
        nut_pose = gymapi.Transform()

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # table
            self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 0)

            # bolt
            bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
            bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.0)
            bolt_pose.p.z = table_dims.z
            bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            bolt_handle = self.gym.create_actor(env, self.bolt_asset, bolt_pose, "bolt", i, 0)
            self.bolt_actor_idxs.append(self.gym.get_actor_index(env, bolt_handle, gymapi.DOMAIN_SIM))
            self.bolt_rb_idxs.append(self.gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM))

            # nut
            nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
            nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
            nut_pose.p.z = table_dims.z + 0.02
            nut_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            nut_handle = self.gym.create_actor(env, self.nut_asset, nut_pose, "nut", i, 0)
            self.nut_actor_idxs.append(self.gym.get_actor_index(env, nut_handle, gymapi.DOMAIN_SIM))
            self.nut_rb_idxs.append(self.gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM))

            # franka
            franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 0)
            self.franka_actor_idxs.append(self.gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM))
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_rb_idxs.append(hand_idx)

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(1, 0, 0.6)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

        # tensor API
        self.gym.prepare_sim(self.sim)

        # root state
        _actor_root = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_root = gymtorch.wrap_tensor(_actor_root)

        # rigid bodies
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # dofs
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_states_view = self.dof_states.view(self.num_envs, self.num_dofs, 2)
        self.dof_pos = self.dof_states_view[:, :, 0]  # [N,9]
        self.dof_vel = self.dof_states_view[:, :, 1]

        # jacobian
        _jac = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jac = gymtorch.wrap_tensor(_jac)
        self.j_eef = self.jac[:, self.hand_body_index_asset - 1, :, :7]  # [N,6,7]

        # indices tensors
        self.nut_rb_idxs_t = torch.tensor(self.nut_rb_idxs, device=self.device, dtype=torch.long)
        self.bolt_rb_idxs_t = torch.tensor(self.bolt_rb_idxs, device=self.device, dtype=torch.long)
        self.hand_rb_idxs_t = torch.tensor(self.hand_rb_idxs, device=self.device, dtype=torch.long)

        self.nut_actor_idxs_t = torch.tensor(self.nut_actor_idxs, device=self.device, dtype=torch.long)
        self.bolt_actor_idxs_t = torch.tensor(self.bolt_actor_idxs, device=self.device, dtype=torch.long)

        # buffers
        self.progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # RL interface dims
        self.obs_dim = 25
        self.act_dim = 7

        # action limits (base; curriculum can scale)
        self.base_max_trans = 0.002
        self.base_max_rot = 0.03
        self.base_max_grip = 0.01

        # control
        self.damping = 0.15

        # cached target dofs
        self.pos_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.target_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        # curriculum + targets
        self.curriculum = Curriculum(self.device, stage=0)
        self.stage_params = self.curriculum.params()

        # initial reset
        self.reset(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        # refresh actor root
        self.gym.refresh_actor_root_state_tensor(self.sim)

        p = self.curriculum.params()

        # reset dofs
        self.dof_states_view[env_ids, :, 0] = self.default_dof_pos.unsqueeze(0)
        self.dof_states_view[env_ids, :, 1] = 0.0
        self.target_dof_pos[env_ids, :] = self.default_dof_pos.unsqueeze(0)

        # reset nut pose near bolt with curriculum noise
        nut_actor_ids = self.nut_actor_idxs_t[env_ids]
        bolt_actor_ids = self.bolt_actor_idxs_t[env_ids]
        bolt_root = self.actor_root[bolt_actor_ids]  # [K,13]
        bolt_p = bolt_root[:, 0:3]
        bolt_q = bolt_root[:, 3:7]

        K = env_ids.numel()
        xy_noise = (torch.rand((K, 2), device=self.device) - 0.5) * (2.0 * p["init_noise_xy"])
        yaw_noise = (torch.rand((K,), device=self.device) - 0.5) * (2.0 * p["init_noise_yaw"])

        nut_p = bolt_p.clone()
        if p["init_preengaged"]:
            # start already above bolt (pre-engaged) -> easier early stages
            nut_p[:, 0:2] += xy_noise
            nut_p[:, 2] = bolt_p[:, 2] + 0.020  # tune for your assets
        else:
            # start offset -> requires alignment then insertion
            nut_p[:, 0] += xy_noise[:, 0]
            nut_p[:, 1] += 0.20 + xy_noise[:, 1]
            nut_p[:, 2] = self.table_dims.z + 0.02

        # yaw = bolt_yaw + noise
        bolt_yaw = quat_to_yaw_xyzw(bolt_q)
        nut_yaw = bolt_yaw + yaw_noise
        # axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand(K, 3)
        # nut_q = quat_from_angle_axis(nut_yaw.view(K, 1), axis).squeeze(1)
        nut_q = quat_from_yaw(nut_yaw)



        # write root state
        self.actor_root[nut_actor_ids, 0:3] = nut_p
        self.actor_root[nut_actor_ids, 3:7] = nut_q
        self.actor_root[nut_actor_ids, 7:13] = 0.0

        # commit
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.actor_root))

        self.progress[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.success_buf[env_ids] = False

    # -------------------------
    # Observation
    # -------------------------
    def compute_obs(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        nut = self.rb_states[self.nut_rb_idxs_t, :13]   # [N,13]
        bolt = self.rb_states[self.bolt_rb_idxs_t, :13]
        hand = self.rb_states[self.hand_rb_idxs_t, :13]

        nut_p = nut[:, 0:3];  nut_q = nut[:, 3:7]
        bolt_p = bolt[:, 0:3]; bolt_q = bolt[:, 3:7]
        hand_p = hand[:, 0:3]; hand_q = hand[:, 3:7]
        hand_lv = hand[:, 7:10]
        hand_av = hand[:, 10:13]

        rel_nut_bolt = nut_p - bolt_p
        rel_hand_nut = hand_p - nut_p

        rel_yaw = wrap_to_pi(
            quat_to_yaw_xyzw(nut_q) - quat_to_yaw_xyzw(bolt_q)
        ).unsqueeze(-1)


        # gripper sep (two finger joints)
        grip_sep = (self.dof_pos[:, 7] + self.dof_pos[:, 8]).unsqueeze(-1)  # [N,1]

        # nut height above bolt
        nut_height = (nut_p[:, 2] - bolt_p[:, 2]).unsqueeze(-1)

        # contact proxy (distance based)
        xy_dist = torch.linalg.norm(rel_nut_bolt[:, 0:2], dim=1, keepdim=True)
        contact = (xy_dist < 0.010).float()  # proxy; tune or replace with force threshold if available

        # normalized step
        p = self.curriculum.params()
        t = (self.progress.float() / float(p["max_steps"])).clamp(0, 1).unsqueeze(-1)

        obs = torch.cat([
            rel_nut_bolt,             # 3
            nut_q,                    # 4
            rel_hand_nut,             # 3
            hand_q,                   # 4
            hand_lv,                  # 3
            hand_av,                  # 3
            grip_sep,                 # 1
            nut_height,               # 1
            contact,                  # 1
            t,                        # 1
            rel_yaw,    # +1 dim
        ], dim=1)

        # optional observation noise (late curriculum)
        if p["obs_noise"] > 0:
            obs = obs + torch.randn_like(obs) * p["obs_noise"]

        return obs

    # -------------------------
    # Reward + Done
    # -------------------------
    def compute_reward_done(self):
        nut = self.rb_states[self.nut_rb_idxs_t, :13]
        bolt = self.rb_states[self.bolt_rb_idxs_t, :13]
        hand = self.rb_states[self.hand_rb_idxs_t, :13]

        nut_p = nut[:, 0:3]; nut_q = nut[:, 3:7]
        bolt_p = bolt[:, 0:3]; bolt_q = bolt[:, 3:7]
        hand_p = hand[:, 0:3]

        # geometry
        rel = nut_p - bolt_p
        xy_dist = torch.linalg.norm(rel[:, 0:2], dim=1)
        # target z offset for "screwed down" proxy (tune!)
        TARGET_Z_OFFSET = 0.020
        z_err = torch.abs(rel[:, 2] - TARGET_Z_OFFSET)

        # rotation progress: nut yaw relative to bolt yaw
        nut_yaw = quat_to_yaw_xyzw(nut_q)
        bolt_yaw = quat_to_yaw_xyzw(bolt_q)
        rel_yaw = wrap_to_pi(nut_yaw - bolt_yaw)  # [-pi,pi]

        p = self.curriculum.params()
        yaw_target = p["yaw_target"]
        # For large targets like -2*pi, measure progress by accumulated rotation is hard without unwrapped yaw.
        # We'll use a shaped "move toward target modulo 2pi" proxy. For stage 4, success uses z+xy mainly + "enough rotation".
        yaw_err = torch.abs(wrap_to_pi(rel_yaw - wrap_to_pi(torch.tensor(yaw_target, device=self.device))))

        # contact proxy
        contact = (xy_dist < 0.010).float()

        # wrist limit barrier (learn to avoid limits)
        wrist = self.dof_pos[:, self.wrist_idx]
        # normalized distance to nearest limit
        dist_low = (wrist - self.wrist_low)
        dist_high = (self.wrist_high - wrist)
        dist_near = torch.minimum(dist_low, dist_high)
        # barrier grows when within ~10% of range
        wrist_range = (self.wrist_high - self.wrist_low)
        margin = 0.10 * wrist_range
        wrist_barrier = torch.clamp((margin - dist_near) / (margin + 1e-6), 0.0, 1.0)

        # reward terms
        r_xy = -xy_dist
        r_z = -z_err
        r_contact = 0.2 * contact

        # rotation reward only in stages that need rotation
        if p["need_rotation"]:
            r_yaw = -0.5 * yaw_err
        else:
            r_yaw = torch.zeros_like(r_xy)

        # smoothness penalty (proxy: hand distance to nut)
        # r_hand = -0.1 * torch.linalg.norm(hand_p - nut_p, dim=1)

        hand_dist = torch.linalg.norm(hand_p - nut_p, dim=1)
        # stronger reach shaping (especially before rotation curriculum)
        reach_w = 1.0 if (not p["need_rotation"]) else 0.2
        r_hand = -reach_w * hand_dist


        # wrist penalty
        r_wrist = -0.5 * wrist_barrier

        # success
        xy_ok = xy_dist < 0.003
        z_ok = z_err < 0.002

        if p["need_rotation"]:
            yaw_ok = yaw_err < p["yaw_tol"]
            success = xy_ok & z_ok & yaw_ok
        else:
            success = xy_ok & z_ok

        # timeout / fail
        timeout = self.progress >= p["max_steps"]

        # nut fallen (below table)
        fallen = nut_p[:, 2] < (self.table_dims.z * 0.7)

        done = success | timeout | fallen

        # sparse bonuses
        r_success = success.float() * 10.0
        r_fail = fallen.float() * -5.0

        reward = r_xy + r_z + r_yaw + r_contact + r_hand + r_wrist + r_success + r_fail

        r_reach_bonus = (hand_dist < 0.03).float() * 0.5
        reward = reward + r_reach_bonus


        # store success
        self.success_buf = success

        return reward, done

    # -------------------------
    # Wrist-limit aware action application
    # -------------------------
    def apply_actions(self, actions: torch.Tensor):
        """
        actions: [N,7] in [-1,1] (PPO outputs)
          [dx,dy,dz,droll,dpitch,dyaw,dgrip]
        """
        p = self.curriculum.params()
        # max_trans = self.base_max_trans * (p["action_scale_trans"] / self.base_max_trans)
        # max_rot = self.base_max_rot * (p["action_scale_rot"] / self.base_max_rot)
        # max_grip = self.base_max_grip
        max_trans = p["action_scale_trans"]
        max_rot   = p["action_scale_rot"]
        max_grip  = self.base_max_grip


        # scale and clamp
        a = torch.clamp(actions, -1.0, 1.0)
        dpos = a[:, 0:3] * max_trans
        drot = a[:, 3:6] * max_rot
        dgrip = a[:, 6] * max_grip


        # prevent vanishing motion
        min_trans = 0.001
        min_rot   = 0.01

        dpos = torch.where(
            dpos.abs() < min_trans,
            min_trans * torch.sign(dpos + 1e-6),
            dpos
        )

        drot = torch.where(
            drot.abs() < min_rot,
            min_rot * torch.sign(drot + 1e-6),
            drot
        )

        # ---- wrist limit safety filter (no FSM) ----
        # When wrist joint near limits, attenuate commanded yaw (which often maps heavily to wrist7).
        wrist = self.dof_pos[:, self.wrist_idx]
        wrist_range = (self.wrist_high - self.wrist_low)
        margin = 0.12 * wrist_range
        dist_low = wrist - self.wrist_low
        dist_high = self.wrist_high - wrist
        dist_near = torch.minimum(dist_low, dist_high)
        # scale in [0,1]: 1 far from limits, 0 at limit
        # yaw_scale = torch.clamp(dist_near / (margin + 1e-6), 0.0, 1.0)
        # drot[:, 2] = drot[:, 2] * yaw_scale  # attenuate yaw
        yaw_scale = 0.2 + 0.8 * torch.clamp(dist_near / (margin + 1e-6), 0.0, 1.0)
        drot[:, 2] *= yaw_scale


        # desired dpose
        dpose = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        dpose[:, 0:3] = dpos
        dpose[:, 3:6] = drot

        # IK -> dq
        self.gym.refresh_jacobian_tensors(self.sim)
        dq = control_ik_batched(dpose, self.j_eef, damping=self.damping)  # [N,7]

        # joint position target update (position control)
        q = self.dof_pos[:, :7]  # [N,7]
        q_tgt = q + dq

        # hard clamp joint targets to limits (including wrist)
        q_tgt = torch.max(torch.min(q_tgt, self.franka_upper[:7].unsqueeze(0)), self.franka_lower[:7].unsqueeze(0))

        # gripper target: integrate delta, clamp
        grip = (self.dof_pos[:, 7] + self.dof_pos[:, 8])  # [N]
        grip_tgt = torch.clamp(grip + dgrip, float(self.franka_lower[7].item())*2.0, float(self.franka_upper[7].item())*2.0)
        finger = 0.5 * grip_tgt

        # write targets
        self.pos_action[:, :7] = q_tgt
        self.pos_action[:, 7] = finger
        self.pos_action[:, 8] = finger
        

        # if torch.rand(1).item() < 0.001:
        #     print("dq norm:", dq.norm(dim=1).mean().item(),
        #         "dpose norm:", dpose.norm(dim=1).mean().item())

        if torch.rand(1).item() < 0.002:
            print(
                "dq mean:", dq.abs().mean().item(),
                "q7:", self.dof_pos[:, self.wrist_idx].mean().item(),
                "grip:", (self.dof_pos[:,7]+self.dof_pos[:,8]).mean().item()
            )


        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    # -------------------------
    # Step
    # -------------------------
    def step(self, actions: torch.Tensor):
        # apply action targets
        self.apply_actions(actions)

        # simulate
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh state
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # progress
        self.progress += 1

        # reward/done
        reward, done = self.compute_reward_done()

        # reset done envs
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.reset(done_ids)

        # obs
        obs = self.compute_obs()

        # viewer
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        info = {
            "success": self.success_buf.clone(),
            "done": done.clone()
        }
        return obs, reward, done, info

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

# -------------------------
# PPO Trainer
# -------------------------
class PPO:
    def __init__(self, env: NutBoltScrewEnv):
        self.env = env
        self.device = env.device

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.net = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

        # PPO hyperparams
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 1.0

        # rollout settings
        self.horizon = 256
        self.mini_epochs = 4
        self.minibatch = 2048

    @torch.no_grad()
    def rollout(self):
        N = self.env.num_envs
        obs = self.env.compute_obs()

        obs_buf = torch.zeros((self.horizon, N, self.obs_dim), device=self.device)
        act_buf = torch.zeros((self.horizon, N, self.act_dim), device=self.device)
        logp_buf = torch.zeros((self.horizon, N), device=self.device)
        rew_buf = torch.zeros((self.horizon, N), device=self.device)
        done_buf = torch.zeros((self.horizon, N), device=self.device)
        val_buf = torch.zeros((self.horizon, N), device=self.device)

        success_count = 0
        done_count = 0

        for t in range(self.horizon):
            mu, std, v = self.net(obs)
            eps = torch.randn_like(mu)
            act = mu + std * eps
            logp = gaussian_logprob(act, mu, std)

            obs2, rew, done, info = self.env.step(act)

            obs_buf[t] = obs
            act_buf[t] = act
            logp_buf[t] = logp
            rew_buf[t] = rew
            done_buf[t] = done.float()
            val_buf[t] = v

            success_count += int(info["success"].sum().item())
            done_count += int(done.sum().item())

            obs = obs2

        # bootstrap
        mu, std, v_last = self.net(obs)

        # GAE
        adv = torch.zeros((self.horizon, N), device=self.device)
        lastgaelam = torch.zeros((N,), device=self.device)
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                nextnonterminal = 1.0 - done_buf[t]
                nextvalues = v_last
            else:
                nextnonterminal = 1.0 - done_buf[t + 1]
                nextvalues = val_buf[t + 1]
            delta = rew_buf[t] + self.gamma * nextvalues * nextnonterminal - val_buf[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam

        ret = adv + val_buf
        # flatten
        b_obs = obs_buf.reshape(-1, self.obs_dim)
        b_act = act_buf.reshape(-1, self.act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)

        # normalize adv
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        stats = {
            "success_per_rollout": success_count,
            "done_per_rollout": done_count
        }
        return b_obs, b_act, b_logp, b_adv, b_ret, b_val, stats

    def update(self, b_obs, b_act, b_logp, b_adv, b_ret, b_val):
        total = b_obs.shape[0]
        idx = torch.arange(total, device=self.device)

        for _ in range(self.mini_epochs):
            perm = idx[torch.randperm(total)]
            for start in range(0, total, self.minibatch):
                mb = perm[start:start + self.minibatch]
                obs = b_obs[mb]
                act = b_act[mb]
                old_logp = b_logp[mb]
                adv = b_adv[mb]
                ret = b_ret[mb]

                mu, std, v = self.net(obs)
                logp = gaussian_logprob(act, mu, std)
                ratio = torch.exp(logp - old_logp)

                # PPO loss
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                # value loss
                v_loss = 0.5 * (ret - v).pow(2).mean()

                # entropy
                ent = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std + 1e-8)).sum(-1).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.entropy_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

        return {"pi_loss": float(pi_loss.item()), "v_loss": float(v_loss.item()), "ent": float(ent.item())}

# -------------------------
# Main
# -------------------------
def main():
    env = NutBoltScrewEnv(args=None)
    algo = PPO(env)

    # # --- sanity motion test ---
    # with torch.no_grad():
    #     test_action = torch.zeros((env.num_envs, env.act_dim), device=env.device)
    #     test_action[:, 2] = 1.0   # move down
    #     test_action[:, 5] = 0.5   # yaw
    #     for _ in range(100):
    #         env.step(test_action)


    # training loop
    print("[TRAIN] starting...")
    t0 = time.time()

    for it in range(1, 20001):
        b_obs, b_act, b_logp, b_adv, b_ret, b_val, stats = algo.rollout()
        losses = algo.update(b_obs, b_act, b_logp, b_adv, b_ret, b_val)

        # estimate success rate from rollout stats (rough)
        # "success_per_rollout" counts successes at termination instants; normalize by num_envs*horizon scale
        succ_rate = stats["success_per_rollout"] / max(1.0, stats["done_per_rollout"])

        # curriculum update
        env.curriculum.update(succ_rate)

        if it % 20 == 0:
            dt = time.time() - t0
            p = env.curriculum.params()
            print(
                f"[it {it:05d}] stage={env.curriculum.stage} "
                f"succ_rate={succ_rate:.3f} ema={env.curriculum.succ_ema:.3f} "
                f"pi={losses['pi_loss']:.3f} v={losses['v_loss']:.3f} ent={losses['ent']:.3f} "
                f"max_steps={p['max_steps']} yaw_target={p['yaw_target']:.2f} dt={dt:.1f}s"
            )
            t0 = time.time()

    env.close()

if __name__ == "__main__":
    main()
