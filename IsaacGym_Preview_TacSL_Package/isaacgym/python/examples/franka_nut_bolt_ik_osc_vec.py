# """
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Franka Cube Pick
# ----------------
# Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
# Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
# """

# from inspect import Attribute
# from isaacgym import gymapi
# from isaacgym import gymutil
# from isaacgym import gymtorch
# from isaacgym.torch_utils import *

# import math
# import numpy as np
# import torch


# import torch
# from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_from_angle_axis

# @torch.jit.script
# def orientation_error_batched(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
#     # desired/current: [N,4] (xyzw)
#     cc = quat_conjugate(current)
#     q_r = quat_mul(desired, cc)
#     # q_r: [N,4]
#     sign = torch.sign(q_r[:, 3:4])
#     return q_r[:, 0:3] * sign

# def control_ik_batched(dpose: torch.Tensor, j_eef: torch.Tensor, damping: float) -> torch.Tensor:
#     """
#     dpose: [N,6] or [N,6,1]
#     j_eef: [N,6,7]
#     returns: dq: [N,7]
#     """
#     if dpose.ndim == 2:
#         dpose = dpose.unsqueeze(-1)  # [N,6,1]

#     N = j_eef.shape[0]
#     device = j_eef.device
#     jT = j_eef.transpose(1, 2)  # [N,7,6]
#     # A = J J^T + λ^2 I
#     A = j_eef @ jT  # [N,6,6]
#     I = torch.eye(6, device=device, dtype=A.dtype).unsqueeze(0).expand(N, -1, -1)
#     A = A + (damping ** 2) * I

#     # Solve A x = dpose for x, then dq = J^T x
#     # x: [N,6,1]
#     x = torch.linalg.solve(A, dpose)
#     dq = (jT @ x).squeeze(-1)  # [N,7]
#     return dq

# @torch.jit.script
# def quat_mul_const(a: torch.Tensor, q_const: torch.Tensor) -> torch.Tensor:
#     # a: [K,4], q_const: [4]
#     b = q_const.unsqueeze(0).expand_as(a)
#     return quat_mul(a, b)

# class ScrewFSMVec:
#     """
#     Vectorized nut-bolt screwing FSM.
#     Works on CPU or GPU. No python loop over envs.
#     """

#     # state ids
#     GO_ABOVE_NUT      = 0
#     PREP_GRIP         = 1
#     GRIP              = 2
#     LIFT              = 3
#     GO_ABOVE_BOLT     = 4
#     GO_ON_BOLT        = 5
#     LOOSEN_GRIP       = 6
#     SCREW_MOTION      = 7
#     UNGRIP_SCREW      = 8
#     ROTATE_BACK       = 9
#     BACK_TO_SCREW_GRIP= 10

#     def __init__(self, num_envs, sim_dt, nut_height, bolt_height, screw_speed,
#                  screw_limit_angle, device):
#         self.N = num_envs
#         self.dt = float(sim_dt)
#         self.nut_h = float(nut_height)
#         self.bolt_h = float(bolt_height)
#         self.screw_speed = float(screw_speed)
#         self.screw_limit = float(screw_limit_angle)
#         self.device = torch.device(device)

#         # per-env state
#         self.state = torch.full((self.N,), self.GO_ABOVE_NUT, dtype=torch.int64, device=self.device)
#         self.screw_angle = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

#         # outputs
#         self.dpose = torch.zeros((self.N, 6), dtype=torch.float32, device=self.device)
#         self.grip_sep = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

#         # constants (broadcast)
#         self.above_offset = torch.tensor([0.0, 0.0, 0.08 + self.bolt_h], device=self.device)
#         self.grip_offset  = torch.tensor([0.0, 0.0, 0.12 + self.nut_h], device=self.device)
#         self.lift_offset  = torch.tensor([0.0, 0.0, 0.15 + self.bolt_h], device=self.device)

#         self.above_bolt_offset = (torch.tensor([0.0, 0.0, self.bolt_h], device=self.device) + self.grip_offset)
#         self.on_bolt_offset    = (torch.tensor([0.0, 0.0, 0.8 * self.bolt_h], device=self.device) + self.grip_offset)

#         # hand-down quaternion (xyzw)
#         self.hand_down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

#         grab_angle = torch.tensor([torch.pi / 6.0], device=self.device)
#         grab_axis  = torch.tensor([0.0, 0.0, 1.0], device=self.device)
#         grab_q = quat_from_angle_axis(grab_angle, grab_axis).squeeze(0)  # [4]
#         self.nut_grab_q = quat_mul(grab_q, self.hand_down_q)  # [4]

#         self.screw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)

#     def reset(self, env_ids: torch.Tensor):
#         # env_ids: [K]
#         self.state[env_ids] = self.GO_ABOVE_NUT
#         self.screw_angle[env_ids] = 0.0
#         self.dpose[env_ids] = 0.0
#         self.grip_sep[env_ids] = 0.08

#     def update(self, nut_pose, bolt_pose, hand_pose, current_grip_sep):
#         """
#         nut_pose, bolt_pose, hand_pose: [N,7] -> (x,y,z, qx,qy,qz,qw)
#         current_grip_sep: [N] (dof7 + dof8)
#         outputs:
#           self.dpose: [N,6]
#           self.grip_sep: [N]
#           self.state: [N]
#         """

#         N = self.N
#         dev = self.device

#         nut_pos  = nut_pose[:, 0:3]
#         nut_q    = nut_pose[:, 3:7]
#         bolt_pos = bolt_pose[:, 0:3]
#         hand_pos = hand_pose[:, 0:3]
#         hand_q   = hand_pose[:, 3:7]

#         # allocate targets
#         target_pos  = torch.empty((N, 3), dtype=torch.float32, device=dev)
#         target_q    = torch.empty((N, 4), dtype=torch.float32, device=dev)
#         target_sep  = torch.empty((N,),    dtype=torch.float32, device=dev)

#         # default (will be overwritten by masks)
#         target_pos[:] = hand_pos
#         target_q[:]   = self.hand_down_q
#         target_sep[:] = 0.08

#         s = self.state

#         # ---------- state masks ----------
#         m_go_above_nut   = (s == self.GO_ABOVE_NUT)
#         m_prep_grip      = (s == self.PREP_GRIP)
#         m_grip           = (s == self.GRIP)
#         m_lift           = (s == self.LIFT)
#         m_go_above_bolt  = (s == self.GO_ABOVE_BOLT)
#         m_go_on_bolt     = (s == self.GO_ON_BOLT)
#         m_loosen_grip    = (s == self.LOOSEN_GRIP)
#         m_screw_motion   = (s == self.SCREW_MOTION)
#         m_ungrip_screw   = (s == self.UNGRIP_SCREW)
#         m_rotate_back    = (s == self.ROTATE_BACK)
#         m_back_to_grip   = (s == self.BACK_TO_SCREW_GRIP)

#         # ---------- GO_ABOVE_NUT ----------
#         if m_go_above_nut.any():
#             target_sep[m_go_above_nut] = 0.08
#             target_pos[m_go_above_nut] = nut_pos[m_go_above_nut] + self.above_offset
#             target_q[m_go_above_nut]   = self.hand_down_q

#         # ---------- PREP_GRIP ----------
#         if m_prep_grip.any():
#             target_sep[m_prep_grip] = 0.08
#             target_pos[m_prep_grip] = nut_pos[m_prep_grip] + self.grip_offset
#             # targetQ = nut_q * nut_grab_q
#             target_q[m_prep_grip] = quat_mul_const(nut_q[m_prep_grip], self.nut_grab_q)

#         # ---------- GRIP ----------
#         if m_grip.any():
#             target_sep[m_grip] = 0.0
#             target_pos[m_grip] = nut_pos[m_grip] + self.grip_offset
#             target_q[m_grip]   = quat_mul_const(nut_q[m_grip], self.nut_grab_q)

#         # ---------- LIFT ----------
#         if m_lift.any():
#             target_sep[m_lift] = 0.0
#             pos = nut_pos[m_lift].clone()
#             # keep nut aligned above bolt z + 0.004, then lift
#             pos[:, 2] = bolt_pos[m_lift, 2] + 0.004
#             target_pos[m_lift] = pos + self.lift_offset
#             target_q[m_lift]   = self.hand_down_q

#         # ---------- GO_ABOVE_BOLT ----------
#         if m_go_above_bolt.any():
#             target_sep[m_go_above_bolt] = 0.0
#             target_pos[m_go_above_bolt] = bolt_pos[m_go_above_bolt] + self.above_bolt_offset
#             target_q[m_go_above_bolt]   = self.hand_down_q

#         # ---------- GO_ON_BOLT ----------
#         if m_go_on_bolt.any():
#             target_sep[m_go_on_bolt] = 0.0
#             pos = bolt_pos[m_go_on_bolt].clone()
#             pos[:, 2] = bolt_pos[m_go_on_bolt, 2]
#             target_pos[m_go_on_bolt] = pos + self.on_bolt_offset
#             target_q[m_go_on_bolt]   = self.hand_down_q

#         # ---------- LOOSEN_GRIP ----------
#         if m_loosen_grip.any():
#             sep = 0.037
#             target_sep[m_loosen_grip] = sep
#             target_pos[m_loosen_grip] = bolt_pos[m_loosen_grip] + self.on_bolt_offset
#             target_q[m_loosen_grip]   = self.hand_down_q

#         # ---------- SCREW_MOTION ----------
#         if m_screw_motion.any():
#             sep = 0.037
#             target_sep[m_screw_motion] = sep

#             pos = bolt_pos[m_screw_motion].clone()
#             pos[:, 2] = nut_pos[m_screw_motion, 2]
#             target_pos[m_screw_motion] = pos + self.grip_offset

#             # screw_angle -= dt * screw_speed
#             self.screw_angle[m_screw_motion] -= self.dt * self.screw_speed

#             ang = self.screw_angle[m_screw_motion].unsqueeze(-1)  # [K,1]
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)  # [K,4]
#             target_q[m_screw_motion] = quat_mul_const(screw_q, self.hand_down_q)

#         # ---------- UNGRIP_SCREW ----------
#         if m_ungrip_screw.any():
#             sep = 0.06
#             target_sep[m_ungrip_screw] = sep

#             pos = bolt_pos[m_ungrip_screw].clone()
#             pos[:, 2] = nut_pos[m_ungrip_screw, 2]
#             target_pos[m_ungrip_screw] = pos + self.grip_offset

#             ang = self.screw_angle[m_ungrip_screw].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_ungrip_screw] = quat_mul_const(screw_q, self.hand_down_q)

#         # ---------- ROTATE_BACK ----------
#         if m_rotate_back.any():
#             sep = 0.06
#             target_sep[m_rotate_back] = sep

#             pos = bolt_pos[m_rotate_back].clone()
#             pos[:, 2] = nut_pos[m_rotate_back, 2]
#             target_pos[m_rotate_back] = pos + self.grip_offset

#             # screw_angle += dt * 2*screw_speed
#             self.screw_angle[m_rotate_back] += self.dt * (2.0 * self.screw_speed)

#             ang = self.screw_angle[m_rotate_back].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_rotate_back] = quat_mul_const(screw_q, self.hand_down_q)

#         # ---------- BACK_TO_SCREW_GRIP ----------
#         if m_back_to_grip.any():
#             sep = 0.037
#             target_sep[m_back_to_grip] = sep

#             pos = bolt_pos[m_back_to_grip].clone()
#             pos[:, 2] = nut_pos[m_back_to_grip, 2]
#             target_pos[m_back_to_grip] = pos + self.grip_offset

#             ang = self.screw_angle[m_back_to_grip].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_back_to_grip] = quat_mul_const(screw_q, self.hand_down_q)

#         # ---------- compute dpose ----------
#         pos_err = (target_pos - hand_pos)                         # [N,3]
#         rot_err = orientation_error_batched(target_q, hand_q)     # [N,3]
#         self.dpose[:, 0:3] = pos_err
#         self.dpose[:, 3:6] = rot_err
#         self.grip_sep[:] = target_sep

#         # ---------- transitions ----------
#         err_norm = torch.linalg.norm(self.dpose, dim=1)  # [N]
#         # convenience grip tests
#         gripped_0035 = (current_grip_sep < 0.035)
#         # for states with target sep:
#         ungripped_target = (current_grip_sep > (target_sep * 0.98))

#         # GO_ABOVE_NUT -> PREP_GRIP
#         go_above_nut_done = m_go_above_nut & (err_norm < 2e-3)
#         self.state[go_above_nut_done] = self.PREP_GRIP

#         # PREP_GRIP -> GRIP
#         prep_grip_done = m_prep_grip & (err_norm < 2e-3)
#         self.state[prep_grip_done] = self.GRIP

#         # GRIP -> LIFT
#         grip_done = m_grip & (err_norm < 1e-2) & gripped_0035
#         self.state[grip_done] = self.LIFT

#         # LIFT -> GO_ABOVE_BOLT
#         lift_done = m_lift & (err_norm < 2e-3)
#         self.state[lift_done] = self.GO_ABOVE_BOLT

#         # GO_ABOVE_BOLT -> GO_ON_BOLT
#         above_bolt_done = m_go_above_bolt & (err_norm < 2e-3)
#         self.state[above_bolt_done] = self.GO_ON_BOLT

#         # GO_ON_BOLT -> LOOSEN_GRIP
#         on_bolt_done = m_go_on_bolt & (err_norm < 2e-3)
#         self.state[on_bolt_done] = self.LOOSEN_GRIP

#         # LOOSEN_GRIP -> SCREW_MOTION (and reset screw angle)
#         loosen_done = m_loosen_grip & (err_norm < 2e-3) & ungripped_target
#         self.screw_angle[loosen_done] = 0.0
#         self.state[loosen_done] = self.SCREW_MOTION

#         # SCREW_MOTION -> UNGRIP_SCREW
#         screw_done = m_screw_motion & (self.screw_angle < -self.screw_limit)
#         self.state[screw_done] = self.UNGRIP_SCREW

#         # UNGRIP_SCREW -> ROTATE_BACK
#         ungrip_done = m_ungrip_screw & ungripped_target
#         self.state[ungrip_done] = self.ROTATE_BACK

#         # ROTATE_BACK -> BACK_TO_SCREW_GRIP
#         back_done = m_rotate_back & (self.screw_angle > 0.99 * self.screw_limit)
#         self.state[back_done] = self.BACK_TO_SCREW_GRIP

#         # BACK_TO_SCREW_GRIP -> SCREW_MOTION (and set to limit)
#         # (Your original logic sets screw_angle to screw_limit_angle then continues)
#         gripped_near_target = (current_grip_sep < (target_sep * 1.01))
#         back_to_grip_done = m_back_to_grip & (err_norm < 2e-3) & gripped_near_target
#         self.screw_angle[back_to_grip_done] = self.screw_limit
#         self.state[back_to_grip_done] = self.SCREW_MOTION

#         return self.dpose, self.grip_sep, self.state


#     @property
#     def d_pose(self):
#         return self.dpose

#     @property
#     def gripper_separation(self):
#         return self.grip_sep


# # set random seed
# np.random.seed(42)

# torch.set_printoptions(precision=4, sci_mode=False)

# # acquire gym interface
# gym = gymapi.acquire_gym()

# # parse arguments

# # Add custom arguments
# custom_parameters = [
#     {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
# ]
# args = gymutil.parse_arguments(
#     description="Franka Jacobian Inverse Kinematics (IK) Nut-Bolt Screwing",
#     custom_parameters=custom_parameters,
# )

# # Force GPU:
# if not args.use_gpu or args.use_gpu_pipeline:
#     print("Forcing GPU sim - CPU sim not supported by SDF")
#     args.use_gpu = True
#     args.use_gpu_pipeline = True

# # set torch device
# device = args.sim_device

# # configure sim
# sim_params = gymapi.SimParams()
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.dt = 1.0 / 60.0
# sim_params.substeps = 2
# sim_params.use_gpu_pipeline = args.use_gpu_pipeline
# if args.physics_engine == gymapi.SIM_PHYSX:
#     sim_params.physx.solver_type = 1
#     sim_params.physx.num_position_iterations = 32
#     sim_params.physx.num_velocity_iterations = 1
#     sim_params.physx.rest_offset = 0.0
#     sim_params.physx.contact_offset = 0.005
#     sim_params.physx.friction_offset_threshold = 0.01
#     sim_params.physx.friction_correlation_distance = 0.0005
#     sim_params.physx.num_threads = args.num_threads
#     sim_params.physx.use_gpu = args.use_gpu
# else:
#     raise Exception("This example can only be used with PhysX")

# # Set controller parameters
# # IK params
# damping = 0.15

# # create sim
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
# if sim is None:
#     raise Exception("Failed to create sim")

# # create viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise Exception("Failed to create viewer")

# asset_root = "../../assets"

# # create table asset
# table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# # create bolt asset
# bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
# bolt_options = gymapi.AssetOptions()
# bolt_options.flip_visual_attachments = False  # default = False
# bolt_options.fix_base_link = True
# bolt_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
# bolt_options.density = 800.0  # 7850.0
# bolt_options.armature = 0.0  # default = 0.0
# bolt_options.linear_damping = 0.0  # default = 0.0
# bolt_options.max_linear_velocity = 1000.0  # default = 1000.0
# bolt_options.angular_damping = 0.0  # default = 0.5
# bolt_options.max_angular_velocity = 1000.0  # default = 64.0
# bolt_options.disable_gravity = False  # default = False
# bolt_options.enable_gyroscopic_forces = True  # default = True
# bolt_asset = gym.load_asset(sim, asset_root, bolt_file, bolt_options)

# # create nut asset
# nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
# nut_options = gymapi.AssetOptions()
# nut_options.flip_visual_attachments = False  # default = False
# nut_options.fix_base_link = False
# nut_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
# nut_options.density = 800  # 7850.0  # default = 1000
# nut_options.armature = 0.0  # default = 0.0
# nut_options.linear_damping = 0.0  # default = 0.0
# nut_options.max_linear_velocity = 1000.0  # default = 1000.0
# nut_options.angular_damping = 0.0  # default = 0.5
# nut_options.max_angular_velocity = 1000.0  # default = 64.0
# nut_options.disable_gravity = False  # default = False
# nut_options.enable_gyroscopic_forces = True  # default = True
# nut_asset = gym.load_asset(sim, asset_root, nut_file, nut_options)

# # create box asset

# # asset_options = gymapi.AssetOptions()
# # box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# # load franka asset
# franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
# asset_options = gymapi.AssetOptions()
# asset_options.armature = 0.01
# asset_options.fix_base_link = True
# asset_options.disable_gravity = True
# asset_options.flip_visual_attachments = True
# franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# # configure franka dofs
# franka_dof_props = gym.get_asset_dof_properties(franka_asset)
# franka_lower_limits = franka_dof_props["lower"]
# franka_upper_limits = franka_dof_props["upper"]
# franka_ranges = franka_upper_limits - franka_lower_limits
# franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# # use position drive for all dofs
# franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
# franka_dof_props["stiffness"][:7].fill(400.0)
# franka_dof_props["damping"][:7].fill(40.0)
# # grippers
# franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
# franka_dof_props["stiffness"][7:].fill(800.0)
# franka_dof_props["damping"][7:].fill(40.0)

# # default dof states and position targets
# franka_num_dofs = gym.get_asset_dof_count(franka_asset)
# default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
# default_dof_pos[:7] = franka_mids[:7]
# # grippers open
# default_dof_pos[7:] = franka_upper_limits[7:]

# # send to torch
# default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# # get link index of panda hand, which we will use as end effector
# franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
# franka_hand_index = franka_link_dict["panda_hand"]

# # configure env grid
# num_envs = args.num_envs
# num_per_row = int(math.sqrt(num_envs))
# spacing = 1.0
# env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
# env_upper = gymapi.Vec3(spacing, spacing, spacing)
# print("Creating %d environments" % num_envs)

# franka_pose = gymapi.Transform()
# franka_pose.p = gymapi.Vec3(0, 0, 0)

# table_pose = gymapi.Transform()
# table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
# bolt_pose = gymapi.Transform()
# nut_pose = gymapi.Transform()

# # fsm parameters:
# fsm_device = device  # IMPORTANT: put FSM on same device as sim tensors for speed

# envs = []
# nut_idxs = []
# bolt_idxs = []
# hand_idxs = []
# # fsms = []

# franka_handles = []
# nut_handles = []
# bolt_handles = []

# franka_actor_idxs = []
# nut_actor_idxs = []
# bolt_actor_idxs = []


# # add ground plane
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1)
# gym.add_ground(sim, plane_params)

# for i in range(num_envs):

#     # create env
#     env = gym.create_env(sim, env_lower, env_upper, num_per_row)
#     envs.append(env)

#     # add table
#     table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

#     # add bolt
#     bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
#     bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.0)
#     bolt_pose.p.z = table_dims.z
#     bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
#     bolt_handle = gym.create_actor(env, bolt_asset, bolt_pose, "bolt", i, 0)
#     bolt_handles.append(bolt_handle)
#     bolt_actor_idxs.append(gym.get_actor_index(env, bolt_handle, gymapi.DOMAIN_SIM))
#     bolt_props = gym.get_actor_rigid_shape_properties(env, bolt_handle)

#     # bolt_props[0].filter = imesh
#     bolt_props[0].friction = 0.0  # default = ?
#     bolt_props[0].rolling_friction = 0.0  # default = 0.0
#     bolt_props[0].torsion_friction = 0.0  # default = 0.0
#     bolt_props[0].restitution = 0.0  # default = ?
#     bolt_props[0].compliance = 0.0  # default = 0.0
#     bolt_props[0].thickness = 0.0  # default = 0.0
#     gym.set_actor_rigid_shape_properties(env, bolt_handle, bolt_props)

#     # get global index of box in rigid body state tensor
#     bolt_idx = gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM)
#     bolt_idxs.append(bolt_idx)

#     # add nut
#     nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
#     nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
#     nut_pose.p.z = table_dims.z + 0.02
#     nut_handle = gym.create_actor(env, nut_asset, nut_pose, "nut", i, 0)
#     nut_handles.append(nut_handle)
#     nut_actor_idxs.append(gym.get_actor_index(env, nut_handle, gymapi.DOMAIN_SIM))
#     nut_props = gym.get_actor_rigid_shape_properties(env, nut_handle)
#     # nut_props[0].filter = i
#     nut_props[0].friction = 0.2  # default = ?
#     nut_props[0].rolling_friction = 0.0  # default = 0.0
#     nut_props[0].torsion_friction = 0.0  # default = 0.0
#     nut_props[0].restitution = 0.0  # default = ?
#     nut_props[0].compliance = 0.0  # default = 0.0
#     nut_props[0].thickness = 0.0  # default = 0.0
#     gym.set_actor_rigid_shape_properties(env, nut_handle, nut_props)

#     # get global index of box in rigid body state tensor
#     nut_idx = gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM)
#     nut_idxs.append(nut_idx)

#     # add franka
#     franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)
#     franka_handles.append(franka_handle)
#     franka_actor_idxs.append(gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM))

#     # set dof properties
#     gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

#     # get global index of hand in rigid body state tensor
#     hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
#     hand_idxs.append(hand_idx)

#     # create env's fsm - run them on CPU
#     # fsms.append(ScrewFSMVec(sim_params.dt, 0.016, 0.1, 30.0 / 180.0 * np.pi, 60.0/180.0 * np.pi, fsm_device, i))

# # point camera at middle env
# cam_pos = gymapi.Vec3(1, 0, 0.6)
# cam_target = gymapi.Vec3(-1, 0, 0.5)
# middle_env = envs[0]
# gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# # ==== prepare tensors =====
# # from now on, we will use the tensor API that can run on CPU or GPU
# gym.prepare_sim(sim)

# # actor root state tensor: [num_actors, 13] (pos(3), quat(4), linvel(3), angvel(3))
# _actor_root_state = gym.acquire_actor_root_state_tensor(sim)
# actor_root_state = gymtorch.wrap_tensor(_actor_root_state)

# # put actor index lists onto torch (same device as sim tensors)
# nut_actor_idxs_t  = torch.tensor(nut_actor_idxs,  device=device, dtype=torch.int64)   # [num_envs]
# bolt_actor_idxs_t = torch.tensor(bolt_actor_idxs, device=device, dtype=torch.int64)   # [num_envs]


# # get jacobian tensor
# # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
# _jacobian = gym.acquire_jacobian_tensor(sim, "franka")
# jacobian = gymtorch.wrap_tensor(_jacobian)

# # jacobian entries corresponding to franka hand
# j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# # get rigid body state tensor
# _rb_states = gym.acquire_rigid_body_state_tensor(sim)
# rb_states = gymtorch.wrap_tensor(_rb_states)

# # get dof state tensor
# _dof_states = gym.acquire_dof_state_tensor(sim)
# dof_states = gymtorch.wrap_tensor(_dof_states)
# dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

# # Set action tensors
# pos_action = torch.zeros_like(dof_pos).squeeze(-1)

# # dp and gripper sep tensors:
# d_pose = torch.zeros((num_envs, 6), dtype=torch.float32, device=fsm_device)
# grip_sep = torch.zeros((num_envs, 1), dtype=torch.float32, device=fsm_device)

# # Set Franka initial dof position
# dof_pos[:, :, 0] = torch.tensor(default_dof_pos, dtype=torch.float32, device=device)
# target_dof_pos = dof_pos[:, :, 0].clone()
# gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))
# gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_dof_pos))


# fsm = ScrewFSMVec(num_envs, sim_params.dt, 0.016, 0.1,
#                 30.0/180.0*math.pi, 60.0/180.0*math.pi,
#                 device=fsm_device)
# # fsm.reset(torch.arange(num_envs, device=device))
# # simulation loop
# while not gym.query_viewer_has_closed(viewer):

#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # refresh tensors
#     gym.refresh_rigid_body_state_tensor(sim)
#     gym.refresh_dof_state_tensor(sim)
#     gym.refresh_jacobian_tensors(sim)

#     # rb_states_fsm = rb_states.to(fsm_device)
#     # nut_poses = rb_states_fsm[nut_idxs, :7]
#     # bolt_poses = rb_states_fsm[bolt_idxs, :7]
#     # hand_poses = rb_states_fsm[hand_idxs, :7]
#     # dof_pos_fsm = dof_pos.to(fsm_device)
#     # cur_grip_sep_fsm = dof_pos_fsm[:, 7] + dof_pos_fsm[:, 8]
#     # for env_idx in range(num_envs):
#     #     fsms[env_idx].update(nut_poses[env_idx, :], bolt_poses[env_idx, :], hand_poses[env_idx, :], cur_grip_sep_fsm[env_idx])
#     #     d_pose[env_idx, :] = fsms[env_idx].d_pose
#     #     grip_sep[env_idx] = fsms[env_idx].gripper_separation

#     # rb_states_fsm = rb_states if rb_states.device.type == fsm_device else rb_states.to(fsm_device)
#     # nut_poses  = rb_states_fsm[nut_idxs,  :7]
#     # bolt_poses = rb_states_fsm[bolt_idxs, :7]
#     # hand_poses = rb_states_fsm[hand_idxs, :7]

#     # dof_pos_fsm = dof_pos.squeeze(-1) if dof_pos.device.type == fsm_device else dof_pos.squeeze(-1).to(fsm_device)
#     # cur_grip_sep = dof_pos_fsm[:, 7] + dof_pos_fsm[:, 8]  # [N]

#     # d_pose, grip_sep, fsm_state = fsm.update(nut_poses, bolt_poses, hand_poses, cur_grip_sep)

#     # # IK control (on sim device)
#     # d_pose_sim = d_pose.to(device)
#     # dq = control_ik_batched(d_pose_sim, j_eef, damping)  # [N,7]

#     # pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + dq
#     # grip_acts = torch.stack((0.5 * grip_sep, 0.5 * grip_sep), dim=1).to(device)  # [N,2]
#     # pos_action[:, 7:9] = grip_acts

#     rb_states_fsm = rb_states
#     nut_poses  = rb_states_fsm[nut_idxs,  :7]
#     bolt_poses = rb_states_fsm[bolt_idxs, :7]
#     hand_poses = rb_states_fsm[hand_idxs, :7]

#     dof_pos_fsm = dof_pos.squeeze(-1)
#     cur_grip_sep = dof_pos_fsm[:, 7] + dof_pos_fsm[:, 8]

#     d_pose, grip_sep, fsm_state = fsm.update(
#         nut_poses, bolt_poses, hand_poses, cur_grip_sep
#     )

#     dq = control_ik_batched(d_pose, j_eef, damping)

#     pos_action[:, :7] = dof_pos_fsm[:, :7] + dq
#     pos_action[:, 7]  = 0.5 * grip_sep
#     pos_action[:, 8]  = 0.5 * grip_sep

#     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))


#     # pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik_batched(d_pose.unsqueeze(-1).to(device), damping, j_eef, num_envs)
#     # # gripper actions depend on distance between hand and box

#     # grip_acts = torch.cat((0.5 * grip_sep, 0.5 * grip_sep), 1).to(device)
#     # pos_action[:, 7:9] = grip_acts

#     # # Deploy actions
#     # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

#     # update viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, False)
#     gym.sync_frame_time(sim)

# # cleanup
# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)


"""
Franka Nut-Bolt Screwing (Vectorized FSM) with IK or OSC control + Release/Move-away
----------------------------------------------------------------------------------

- Vectorized FSM (no per-env python loops)
- Supports --controller {ik, osc}
- Completion criterion: once nut has moved DOWN by SCREW_Z_TRAVEL meters along WORLD Z
  relative to the z position recorded at the start of SCREW_MOTION.
- After completion: open gripper and move away (lift + retreat), then DONE.

NOTE: For M4 nut/bolt, 0.4m is extremely large. If you meant 4mm, set SCREW_Z_TRAVEL=0.004.

Usage:
  python screw_fsm_ik_osc_release.py --num_envs 4 --controller ik
  python screw_fsm_ik_osc_release.py --num_envs 4 --controller osc
"""


from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import (
    to_torch,
    quat_mul,
    quat_conjugate,
    quat_from_angle_axis,
)
import math
import numpy as np
import torch

# -------------------------
# Quaternion / pose helpers
# -------------------------

@torch.jit.script
def orientation_error_batched(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """
    desired/current: [N,4] (xyzw)
    returns: [N,3]
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    sign = torch.sign(q_r[:, 3:4])
    return q_r[:, 0:3] * sign


@torch.jit.script
def quat_mul_const(a: torch.Tensor, q_const: torch.Tensor) -> torch.Tensor:
    """
    a: [K,4], q_const: [4]
    returns: [K,4]
    """
    b = q_const.unsqueeze(0).expand_as(a)
    return quat_mul(a, b)


# -------------------------
# Controllers
# -------------------------

def control_ik_batched(dpose: torch.Tensor, j_eef: torch.Tensor, damping: float) -> torch.Tensor:
    """
    Damped least-squares IK (batched).

    dpose: [N,6] or [N,6,1]
    j_eef: [N,6,7]
    returns dq: [N,7]
    """
    if dpose.ndim == 2:
        dpose = dpose.unsqueeze(-1)  # [N,6,1]

    N = j_eef.shape[0]
    device = j_eef.device
    jT = j_eef.transpose(1, 2)  # [N,7,6]
    A = j_eef @ jT              # [N,6,6]
    I = torch.eye(6, device=device, dtype=A.dtype).unsqueeze(0).expand(N, -1, -1)
    A = A + (damping ** 2) * I

    x = torch.linalg.solve(A, dpose)      # [N,6,1]
    dq = (jT @ x).squeeze(-1)             # [N,7]
    return dq


def control_osc_batched(
    dpose: torch.Tensor,               # [N,6] or [N,6,1]
    j_eef: torch.Tensor,               # [N,6,7]
    mm_7: torch.Tensor,                # [N,7,7]
    q_7: torch.Tensor,                 # [N,7,1]
    qd_7: torch.Tensor,                # [N,7,1]
    hand_vel_6: torch.Tensor,          # [N,6]
    q_des_7: torch.Tensor,             # [7] or [1,7,1]
    kp: float,
    kd: float,
    kp_null: float,
    kd_null: float,
) -> torch.Tensor:
    """
    Operational Space Control with dynamically consistent nullspace (batched).
    Returns joint torques tau: [N,7]
    """
    if dpose.ndim == 2:
        dpose = dpose.unsqueeze(-1)  # [N,6,1]

    N = j_eef.shape[0]
    device = j_eef.device
    dtype = j_eef.dtype

    mm_inv = torch.inverse(mm_7)      # [N,7,7]
    jT = j_eef.transpose(1, 2)        # [N,7,6]

    m_eef_inv = j_eef @ mm_inv @ jT   # [N,6,6]
    m_eef = torch.inverse(m_eef_inv)  # [N,6,6]

    # task-space wrench -> joint torque
    hand_vel = hand_vel_6.unsqueeze(-1)  # [N,6,1]
    u_task = jT @ m_eef @ (kp * dpose - kd * hand_vel)  # [N,7,1]

    # nullspace posture
    if q_des_7.ndim == 1:
        q_des = q_des_7.view(1, 7, 1).to(device=device, dtype=dtype)
    else:
        q_des = q_des_7.to(device=device, dtype=dtype)

    q_err = (q_des - q_7 + math.pi) % (2 * math.pi) - math.pi  # [N,7,1]
    u_null = kd_null * (-qd_7) + kp_null * q_err               # [N,7,1]
    u_null = mm_7 @ u_null                                     # [N,7,1]

    # dynamically consistent nullspace projector
    j_eef_inv = m_eef @ j_eef @ mm_inv                         # [N,6,7]
    I7 = torch.eye(7, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1)
    proj = I7 - (jT @ j_eef_inv)                                # [N,7,7]

    u = u_task + proj @ u_null                                  # [N,7,1]
    return u.squeeze(-1)                                        # [N,7]


# -------------------------
# Vectorized FSM
# -------------------------

class ScrewFSMVec:
    """
    Vectorized nut-bolt screwing FSM.
    Adds terminal behavior: RELEASE + MOVE_AWAY + DONE
    triggered when nut has moved down by SCREW_Z_TRAVEL along world Z since start of SCREW_MOTION.
    """

    # original state ids
    GO_ABOVE_NUT       = 0
    PREP_GRIP          = 1
    GRIP               = 2
    LIFT               = 3
    GO_ABOVE_BOLT      = 4
    GO_ON_BOLT         = 5
    LOOSEN_GRIP        = 6
    SCREW_MOTION       = 7
    UNGRIP_SCREW       = 8
    ROTATE_BACK        = 9
    BACK_TO_SCREW_GRIP = 10

    # new terminal states
    RELEASE_DONE       = 11
    MOVE_AWAY          = 12
    DONE               = 13

    def __init__(
        self,
        num_envs: int,
        sim_dt: float,
        nut_height: float,
        bolt_height: float,
        screw_speed: float,
        screw_limit_angle: float,
        device: str,
        screw_z_travel: float = 0.44,
    ):
        self.N = int(num_envs)
        self.dt = float(sim_dt)
        self.nut_h = float(nut_height)
        self.bolt_h = float(bolt_height)
        self.screw_speed = float(screw_speed)
        self.screw_limit = float(screw_limit_angle)
        self.device = torch.device(device)

        # completion threshold (world-z travel)
        self.screw_z_travel = float(screw_z_travel)

        # per-env state
        self.state = torch.full((self.N,), self.GO_ABOVE_NUT, dtype=torch.int64, device=self.device)
        self.screw_angle = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

        # record z at start of screw motion
        self.screw_start_z = torch.full((self.N,), float("nan"), dtype=torch.float32, device=self.device)

        # outputs
        self.dpose = torch.zeros((self.N, 6), dtype=torch.float32, device=self.device)
        self.grip_sep = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

        # constants (broadcast)
        self.above_offset = torch.tensor([0.0, 0.0, 0.08 + self.bolt_h], device=self.device)
        self.grip_offset  = torch.tensor([0.0, 0.0, 0.12 + self.nut_h], device=self.device)
        self.lift_offset  = torch.tensor([0.0, 0.0, 0.15 + self.bolt_h], device=self.device)

        self.above_bolt_offset = (torch.tensor([0.0, 0.0, self.bolt_h], device=self.device) + self.grip_offset)
        self.on_bolt_offset    = (torch.tensor([0.0, 0.0, 0.8 * self.bolt_h], device=self.device) + self.grip_offset)

        # terminal offsets
        self.release_offset = torch.tensor([0.0, 0.0, 0.18 + self.bolt_h], device=self.device)  # lift a bit
        self.away_offset    = torch.tensor([0.25, 0.0, 0.22 + self.bolt_h], device=self.device)  # retreat + lift

        # hand-down quaternion (xyzw)
        self.hand_down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        grab_angle = torch.tensor([torch.pi / 6.0], device=self.device)
        grab_axis  = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        grab_q = quat_from_angle_axis(grab_angle, grab_axis).squeeze(0)  # [4]
        self.nut_grab_q = quat_mul(grab_q, self.hand_down_q)  # [4]

        self.screw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.state[env_ids] = self.GO_ABOVE_NUT
        self.screw_angle[env_ids] = 0.0
        self.screw_start_z[env_ids] = float("nan")
        self.dpose[env_ids] = 0.0
        self.grip_sep[env_ids] = 0.08

    def update(self, nut_pose, bolt_pose, hand_pose, current_grip_sep):
        """
        nut_pose, bolt_pose, hand_pose: [N,7] (x,y,z,qx,qy,qz,qw)
        current_grip_sep: [N]
        Returns:
          dpose: [N,6]
          grip_sep: [N]
          state: [N]
        """
        N = self.N
        dev = self.device

        nut_pos  = nut_pose[:, 0:3]
        nut_q    = nut_pose[:, 3:7]
        bolt_pos = bolt_pose[:, 0:3]
        hand_pos = hand_pose[:, 0:3]
        hand_q   = hand_pose[:, 3:7]

        target_pos  = torch.empty((N, 3), dtype=torch.float32, device=dev)
        target_q    = torch.empty((N, 4), dtype=torch.float32, device=dev)
        target_sep  = torch.empty((N,),    dtype=torch.float32, device=dev)

        # defaults
        target_pos[:] = hand_pos
        target_q[:]   = self.hand_down_q
        target_sep[:] = 0.08

        s = self.state

        # masks
        m_go_above_nut   = (s == self.GO_ABOVE_NUT)
        m_prep_grip      = (s == self.PREP_GRIP)
        m_grip           = (s == self.GRIP)
        m_lift           = (s == self.LIFT)
        m_go_above_bolt  = (s == self.GO_ABOVE_BOLT)
        m_go_on_bolt     = (s == self.GO_ON_BOLT)
        m_loosen_grip    = (s == self.LOOSEN_GRIP)
        m_screw_motion   = (s == self.SCREW_MOTION)
        m_ungrip_screw   = (s == self.UNGRIP_SCREW)
        m_rotate_back    = (s == self.ROTATE_BACK)
        m_back_to_grip   = (s == self.BACK_TO_SCREW_GRIP)

        m_release_done   = (s == self.RELEASE_DONE)
        m_move_away      = (s == self.MOVE_AWAY)
        m_done           = (s == self.DONE)

        # GO_ABOVE_NUT
        if m_go_above_nut.any():
            target_sep[m_go_above_nut] = 0.08
            target_pos[m_go_above_nut] = nut_pos[m_go_above_nut] + self.above_offset
            target_q[m_go_above_nut]   = self.hand_down_q

        # PREP_GRIP
        if m_prep_grip.any():
            target_sep[m_prep_grip] = 0.08
            target_pos[m_prep_grip] = nut_pos[m_prep_grip] + self.grip_offset
            target_q[m_prep_grip]   = quat_mul_const(nut_q[m_prep_grip], self.nut_grab_q)

        # GRIP
        if m_grip.any():
            target_sep[m_grip] = 0.0
            target_pos[m_grip] = nut_pos[m_grip] + self.grip_offset
            target_q[m_grip]   = quat_mul_const(nut_q[m_grip], self.nut_grab_q)

        # LIFT
        if m_lift.any():
            target_sep[m_lift] = 0.0
            pos = nut_pos[m_lift].clone()
            pos[:, 2] = bolt_pos[m_lift, 2] + 0.004
            target_pos[m_lift] = pos + self.lift_offset
            target_q[m_lift]   = self.hand_down_q

        # GO_ABOVE_BOLT
        if m_go_above_bolt.any():
            target_sep[m_go_above_bolt] = 0.0
            target_pos[m_go_above_bolt] = bolt_pos[m_go_above_bolt] + self.above_bolt_offset
            target_q[m_go_above_bolt]   = self.hand_down_q

        # GO_ON_BOLT
        if m_go_on_bolt.any():
            target_sep[m_go_on_bolt] = 0.0
            pos = bolt_pos[m_go_on_bolt].clone()
            pos[:, 2] = bolt_pos[m_go_on_bolt, 2]
            target_pos[m_go_on_bolt] = pos + self.on_bolt_offset
            target_q[m_go_on_bolt]   = self.hand_down_q

        # LOOSEN_GRIP
        if m_loosen_grip.any():
            target_sep[m_loosen_grip] = 0.037
            target_pos[m_loosen_grip] = bolt_pos[m_loosen_grip] + self.on_bolt_offset
            target_q[m_loosen_grip]   = self.hand_down_q

        # SCREW_MOTION
        if m_screw_motion.any():
            target_sep[m_screw_motion] = 0.037
            pos = bolt_pos[m_screw_motion].clone()
            pos[:, 2] = nut_pos[m_screw_motion, 2]
            target_pos[m_screw_motion] = pos + self.grip_offset

            # keep rotating while screwing
            self.screw_angle[m_screw_motion] -= self.dt * self.screw_speed
            ang = self.screw_angle[m_screw_motion].unsqueeze(-1)
            screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
            target_q[m_screw_motion] = quat_mul_const(screw_q, self.hand_down_q)

        # UNGRIP_SCREW (kept for compatibility; may never be reached if z-travel triggers completion)
        if m_ungrip_screw.any():
            target_sep[m_ungrip_screw] = 0.06
            pos = bolt_pos[m_ungrip_screw].clone()
            pos[:, 2] = nut_pos[m_ungrip_screw, 2]
            target_pos[m_ungrip_screw] = pos + self.grip_offset
            ang = self.screw_angle[m_ungrip_screw].unsqueeze(-1)
            screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
            target_q[m_ungrip_screw] = quat_mul_const(screw_q, self.hand_down_q)

        # ROTATE_BACK
        if m_rotate_back.any():
            target_sep[m_rotate_back] = 0.06
            pos = bolt_pos[m_rotate_back].clone()
            pos[:, 2] = nut_pos[m_rotate_back, 2]
            target_pos[m_rotate_back] = pos + self.grip_offset
            self.screw_angle[m_rotate_back] += self.dt * (2.0 * self.screw_speed)
            ang = self.screw_angle[m_rotate_back].unsqueeze(-1)
            screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
            target_q[m_rotate_back] = quat_mul_const(screw_q, self.hand_down_q)

        # BACK_TO_SCREW_GRIP
        if m_back_to_grip.any():
            target_sep[m_back_to_grip] = 0.037
            pos = bolt_pos[m_back_to_grip].clone()
            pos[:, 2] = nut_pos[m_back_to_grip, 2]
            target_pos[m_back_to_grip] = pos + self.grip_offset
            ang = self.screw_angle[m_back_to_grip].unsqueeze(-1)
            screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
            target_q[m_back_to_grip] = quat_mul_const(screw_q, self.hand_down_q)

        # RELEASE_DONE
        if m_release_done.any():
            target_sep[m_release_done] = 0.08
            target_pos[m_release_done] = bolt_pos[m_release_done] + self.release_offset
            target_q[m_release_done]   = self.hand_down_q

        # MOVE_AWAY
        if m_move_away.any():
            target_sep[m_move_away] = 0.08
            target_pos[m_move_away] = bolt_pos[m_move_away] + self.away_offset
            target_q[m_move_away]   = self.hand_down_q

        # DONE
        if m_done.any():
            target_sep[m_done] = 0.08
            target_pos[m_done] = bolt_pos[m_done] + self.away_offset
            target_q[m_done]   = self.hand_down_q

        # compute dpose
        pos_err = target_pos - hand_pos
        rot_err = orientation_error_batched(target_q, hand_q)
        self.dpose[:, 0:3] = pos_err
        self.dpose[:, 3:6] = rot_err
        self.grip_sep[:] = target_sep

        # transitions
        err_norm = torch.linalg.norm(self.dpose, dim=1)

        gripped_0035 = (current_grip_sep < 0.035)
        ungripped_target = (current_grip_sep > (target_sep * 0.98))

        # GO_ABOVE_NUT -> PREP_GRIP
        go_above_nut_done = m_go_above_nut & (err_norm < 2e-3)
        self.state[go_above_nut_done] = self.PREP_GRIP

        # PREP_GRIP -> GRIP
        prep_grip_done = m_prep_grip & (err_norm < 2e-3)
        self.state[prep_grip_done] = self.GRIP

        # GRIP -> LIFT
        grip_done = m_grip & (err_norm < 1e-2) & gripped_0035
        self.state[grip_done] = self.LIFT

        # LIFT -> GO_ABOVE_BOLT
        lift_done = m_lift & (err_norm < 2e-3)
        self.state[lift_done] = self.GO_ABOVE_BOLT

        # GO_ABOVE_BOLT -> GO_ON_BOLT
        above_bolt_done = m_go_above_bolt & (err_norm < 2e-3)
        self.state[above_bolt_done] = self.GO_ON_BOLT

        # GO_ON_BOLT -> LOOSEN_GRIP
        on_bolt_done = m_go_on_bolt & (err_norm < 2e-3)
        self.state[on_bolt_done] = self.LOOSEN_GRIP

        # LOOSEN_GRIP -> SCREW_MOTION (record start z)
        loosen_done = m_loosen_grip & (err_norm < 2e-3) & ungripped_target
        self.screw_angle[loosen_done] = 0.0
        self.screw_start_z[loosen_done] = nut_pos[loosen_done, 2]
        print(f"the screw_start_z: {self.screw_start_z[loosen_done]}")
        self.state[loosen_done] = self.SCREW_MOTION

        # SCREW_MOTION -> RELEASE_DONE when z travel reached
        valid_start = torch.isfinite(self.screw_start_z)
        z_travel = torch.zeros((N,), device=dev, dtype=torch.float32)
        z_travel[valid_start] = self.screw_start_z[valid_start] - nut_pos[valid_start, 2]
        screw_done_by_z = m_screw_motion & valid_start & (z_travel >= self.screw_z_travel)
        self.state[screw_done_by_z] = self.RELEASE_DONE






        # (Optional legacy) SCREW_MOTION -> UNGRIP_SCREW by angle (only if you never trigger z completion)
        screw_done_by_angle = m_screw_motion & (~screw_done_by_z) & (self.screw_angle < -self.screw_limit)
        self.state[screw_done_by_angle] = self.UNGRIP_SCREW

        # UNGRIP_SCREW -> ROTATE_BACK
        ungrip_done = m_ungrip_screw & ungripped_target
        self.state[ungrip_done] = self.ROTATE_BACK

        # ROTATE_BACK -> BACK_TO_SCREW_GRIP
        back_done = m_rotate_back & (self.screw_angle > 0.99 * self.screw_limit)
        self.state[back_done] = self.BACK_TO_SCREW_GRIP

        # BACK_TO_SCREW_GRIP -> SCREW_MOTION
        gripped_near_target = (current_grip_sep < (target_sep * 1.01))
        back_to_grip_done = m_back_to_grip & (err_norm < 2e-3) & gripped_near_target
        self.screw_angle[back_to_grip_done] = self.screw_limit
        self.state[back_to_grip_done] = self.SCREW_MOTION

        # RELEASE_DONE -> MOVE_AWAY (wait until gripper is open enough)
        release_done = m_release_done & (err_norm < 2e-3) & (current_grip_sep > 0.075)
        self.state[release_done] = self.MOVE_AWAY

        # MOVE_AWAY -> DONE
        away_done = m_move_away & (err_norm < 2e-3)
        self.state[away_done] = self.DONE

        return self.dpose, self.grip_sep, self.state


# -------------------------
# Main
# -------------------------

def main():
    np.random.seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # acquire gym
    gym = gymapi.acquire_gym()

    # args
    custom_parameters = [
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments"},
        {"name": "--controller", "type": str, "default": "ik", "help": "Controller: ik or osc"},
        {"name": "--screw_z_travel", "type": float, "default": 0.4, "help": "Nut z travel (m) to trigger release"},
    ]
    args = gymutil.parse_arguments(
        description="Franka IK/OSC Nut-Bolt Screwing + Release on z-travel",
        custom_parameters=custom_parameters,
    )

    # Force GPU sim
    if not args.use_gpu or args.use_gpu_pipeline:
        print("Forcing GPU sim - CPU sim not supported by SDF")
        args.use_gpu = True
        args.use_gpu_pipeline = True

    device = args.sim_device
    use_osc = (args.controller.lower() == "osc")

    # sim params
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 32
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.005
        sim_params.physx.friction_offset_threshold = 0.01
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    else:
        raise RuntimeError("This example requires PhysX")

    # create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    asset_root = "../../assets"

    # ground
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # table
    table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_opts)

    # bolt
    bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
    bolt_opts = gymapi.AssetOptions()
    bolt_opts.fix_base_link = True
    bolt_opts.thickness = 0.0
    bolt_opts.density = 800.0
    bolt_opts.disable_gravity = False
    bolt_opts.enable_gyroscopic_forces = True
    bolt_asset = gym.load_asset(sim, asset_root, bolt_file, bolt_opts)

    # nut
    nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
    nut_opts = gymapi.AssetOptions()
    nut_opts.fix_base_link = False
    nut_opts.thickness = 0.0
    nut_opts.density = 800.0
    nut_opts.disable_gravity = False
    nut_opts.enable_gyroscopic_forces = True
    nut_asset = gym.load_asset(sim, asset_root, nut_file, nut_opts)

    # franka
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    franka_opts = gymapi.AssetOptions()
    franka_opts.armature = 0.01
    franka_opts.fix_base_link = True
    franka_opts.disable_gravity = True
    franka_opts.flip_visual_attachments = True
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, franka_opts)

    # dof props
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

    # controller-specific arm mode
    if use_osc:
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        franka_dof_props["stiffness"][:7].fill(0.0)
        franka_dof_props["damping"][:7].fill(0.0)
    else:
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)

    # grippers always position
    franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][7:].fill(800.0)
    franka_dof_props["damping"][7:].fill(40.0)

    franka_num_dofs = gym.get_asset_dof_count(franka_asset)
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
    default_dof_pos[:7] = franka_mids[:7]
    default_dof_pos[7:] = franka_upper_limits[7:]  # open grippers
    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

    # hand index
    franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
    franka_hand_index = franka_link_dict["panda_hand"]

    # env layout
    num_envs = int(args.num_envs)
    num_per_row = int(math.sqrt(num_envs))
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    print(f"Creating {num_envs} environments | controller={args.controller}")

    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(0, 0, 0)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

    bolt_pose = gymapi.Transform()
    nut_pose = gymapi.Transform()

    envs = []
    nut_idxs = []
    bolt_idxs = []
    hand_idxs = []

    bolt_actor_idxs = []
    nut_actor_idxs = []

    franka_handles = []
    nut_handles = []
    bolt_handles = []

    # create envs/actors
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # table
        gym.create_actor(env, table_asset, table_pose, "table", i, 0)

        # bolt placement
        bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
        bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.0)
        bolt_pose.p.z = table_dims.z
        bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        bolt_handle = gym.create_actor(env, bolt_asset, bolt_pose, "bolt", i, 0)
        bolt_handles.append(bolt_handle)
        bolt_actor_idxs.append(gym.get_actor_index(env, bolt_handle, gymapi.DOMAIN_SIM))

        # tweak bolt shape friction
        bolt_props = gym.get_actor_rigid_shape_properties(env, bolt_handle)
        bolt_props[0].friction = 0.0
        bolt_props[0].rolling_friction = 0.0
        bolt_props[0].torsion_friction = 0.0
        bolt_props[0].restitution = 0.0
        bolt_props[0].compliance = 0.0
        bolt_props[0].thickness = 0.0
        gym.set_actor_rigid_shape_properties(env, bolt_handle, bolt_props)

        # bolt rigid body index (for rb_states)
        bolt_idx = gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM)
        bolt_idxs.append(bolt_idx)

        # nut placement
        nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
        nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
        nut_pose.p.z = table_dims.z + 0.02
        nut_handle = gym.create_actor(env, nut_asset, nut_pose, "nut", i, 0)
        nut_handles.append(nut_handle)
        nut_actor_idxs.append(gym.get_actor_index(env, nut_handle, gymapi.DOMAIN_SIM))

        nut_props = gym.get_actor_rigid_shape_properties(env, nut_handle)
        nut_props[0].friction = 0.2
        nut_props[0].rolling_friction = 0.0
        nut_props[0].torsion_friction = 0.0
        nut_props[0].restitution = 0.0
        nut_props[0].compliance = 0.0
        nut_props[0].thickness = 0.0
        gym.set_actor_rigid_shape_properties(env, nut_handle, nut_props)

        # nut rigid body index
        nut_idx = gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM)
        nut_idxs.append(nut_idx)

        # franka
        franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)
        franka_handles.append(franka_handle)
        gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)

    # camera
    cam_pos = gymapi.Vec3(1, 0, 0.6)
    cam_target = gymapi.Vec3(-1, 0, 0.5)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

    # prepare sim tensors
    gym.prepare_sim(sim)

    # tensors
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)  # [num_bodies,13]

    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)  # [num_envs*9,2]
    dof_states = dof_states.view(num_envs, 9, 2)
    dof_pos = dof_states[:, :, 0:1]  # [N,9,1]
    dof_vel = dof_states[:, :, 1:2]  # [N,9,1]

    _jacobian = gym.acquire_jacobian_tensor(sim, "franka")
    jacobian = gymtorch.wrap_tensor(_jacobian)  # [N, 10, 6, 9]
    j_eef = jacobian[:, franka_hand_index - 1, :, :7]  # [N,6,7]

    # mass matrix for OSC
    _massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
    massmatrix = gymtorch.wrap_tensor(_massmatrix)  # typically [N,9,9]

    # actions
    pos_action = torch.zeros((num_envs, 9), dtype=torch.float32, device=device)
    effort_action = torch.zeros((num_envs, 9), dtype=torch.float32, device=device)

    # initialize dof targets/states
    dof_pos[:, :, 0] = torch.tensor(default_dof_pos, dtype=torch.float32, device=device)
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states.view(-1, 2)))
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_pos[:, :, 0].contiguous()))

    # FSM
    fsm = ScrewFSMVec(
        num_envs=num_envs,
        sim_dt=sim_params.dt,
        nut_height=0.016,
        bolt_height=0.1,
        screw_speed=30.0/180.0*math.pi,
        screw_limit_angle=60.0/180.0*math.pi,
        device=device,
        screw_z_travel=args.screw_z_travel,
    )

    # controller params
    damping = 0.15  # IK
    # OSC gains
    osc_kp = 150.0
    osc_kd = 2.0 * math.sqrt(osc_kp)
    osc_kp_null = 10.0
    osc_kd_null = 2.0 * math.sqrt(osc_kp_null)

    # sim loop
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        # Some builds require explicit refresh; try if available:
        try:
            gym.refresh_mass_matrix_tensors(sim)
        except Exception:
            pass

        # gather poses
        nut_poses = rb_states[nut_idxs, :7]
        bolt_poses = rb_states[bolt_idxs, :7]
        hand_poses = rb_states[hand_idxs, :7]

        # current grip separation (sum of finger joints)
        dof_pos_f = dof_pos[:, :, 0]  # [N,9]
        dof_vel_f = dof_vel[:, :, 0]  # [N,9]
        cur_grip_sep = dof_pos_f[:, 7] + dof_pos_f[:, 8]  # [N]

        d_pose, grip_sep, fsm_state = fsm.update(nut_poses, bolt_poses, hand_poses, cur_grip_sep)

        # gripper targets always position
        pos_action[:, 7] = 0.5 * grip_sep
        pos_action[:, 8] = 0.5 * grip_sep

        if not use_osc:
            # IK -> position targets for arm
            dq = control_ik_batched(d_pose, j_eef, damping)
            pos_action[:, :7] = dof_pos_f[:, :7] + dq
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action.contiguous()))
        else:
            # OSC -> torque for arm, position for grippers
            hand_linvel = rb_states[hand_idxs, 7:10]
            hand_angvel = rb_states[hand_idxs, 10:13]
            hand_vel_6 = torch.cat([hand_linvel, hand_angvel], dim=1)  # [N,6]

            mm_7 = massmatrix[:, :7, :7]   # [N,7,7]
            q_7 = dof_pos[:, :7, :]        # [N,7,1]
            qd_7 = dof_vel[:, :7, :]       # [N,7,1]

            tau = control_osc_batched(
                dpose=d_pose,
                j_eef=j_eef,
                mm_7=mm_7,
                q_7=q_7,
                qd_7=qd_7,
                hand_vel_6=hand_vel_6,
                q_des_7=default_dof_pos_tensor[:7],
                kp=osc_kp,
                kd=osc_kd,
                kp_null=osc_kp_null,
                kd_null=osc_kd_null,
            )

            effort_action.zero_()
            effort_action[:, :7] = tau

            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action.contiguous()))
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action.contiguous()))

        # render
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
