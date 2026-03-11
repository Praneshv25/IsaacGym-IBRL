# """
# Franka Nut-Bolt Screwing (Vectorized FSM) with IK or OSC control + Release/Move-away
# ----------------------------------------------------------------------------------

# - Vectorized FSM (no per-env python loops)
# - Supports --controller {ik, osc}
# - Completion criterion: once nut has moved DOWN by SCREW_Z_TRAVEL meters along WORLD Z
#   relative to the z position recorded at the start of SCREW_MOTION.
# - After completion: open gripper and move away (lift + retreat), then DONE.

# NOTE: For M4 nut/bolt, 0.4m is extremely large. If you meant 4mm, set SCREW_Z_TRAVEL=0.004.

# Usage:
#   python screw_fsm_ik_osc_release.py --num_envs 4 --controller ik
#   python screw_fsm_ik_osc_release.py --num_envs 4 --controller osc
# """


# from isaacgym import gymapi, gymutil, gymtorch
# from isaacgym.torch_utils import (
#     to_torch,
#     quat_mul,
#     quat_conjugate,
#     quat_from_angle_axis,
# )
# import math
# import numpy as np
# import torch
# from collections import defaultdict

# # -------------------------
# # Quaternion / pose helpers
# # -------------------------

# @torch.jit.script
# def orientation_error_batched(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
#     """
#     desired/current: [N,4] (xyzw)
#     returns: [N,3]
#     """
#     cc = quat_conjugate(current)
#     q_r = quat_mul(desired, cc)
#     sign = torch.sign(q_r[:, 3:4])
#     return q_r[:, 0:3] * sign


# @torch.jit.script
# def quat_mul_const(a: torch.Tensor, q_const: torch.Tensor) -> torch.Tensor:
#     """
#     a: [K,4], q_const: [4]
#     returns: [K,4]
#     """
#     b = q_const.unsqueeze(0).expand_as(a)
#     return quat_mul(a, b)


# # -------------------------
# # Controllers
# # -------------------------

# def control_ik_batched(dpose: torch.Tensor, j_eef: torch.Tensor, damping: float) -> torch.Tensor:
#     """
#     Damped least-squares IK (batched).

#     dpose: [N,6] or [N,6,1]
#     j_eef: [N,6,7]
#     returns dq: [N,7]
#     """
#     if dpose.ndim == 2:
#         dpose = dpose.unsqueeze(-1)  # [N,6,1]

#     N = j_eef.shape[0]
#     device = j_eef.device
#     jT = j_eef.transpose(1, 2)  # [N,7,6]
#     A = j_eef @ jT              # [N,6,6]
#     I = torch.eye(6, device=device, dtype=A.dtype).unsqueeze(0).expand(N, -1, -1)
#     A = A + (damping ** 2) * I

#     x = torch.linalg.solve(A, dpose)      # [N,6,1]
#     dq = (jT @ x).squeeze(-1)             # [N,7]
#     return dq


# def control_osc_batched(
#     dpose: torch.Tensor,               # [N,6] or [N,6,1]
#     j_eef: torch.Tensor,               # [N,6,7]
#     mm_7: torch.Tensor,                # [N,7,7]
#     q_7: torch.Tensor,                 # [N,7,1]
#     qd_7: torch.Tensor,                # [N,7,1]
#     hand_vel_6: torch.Tensor,          # [N,6]
#     q_des_7: torch.Tensor,             # [7] or [1,7,1]
#     kp: float,
#     kd: float,
#     kp_null: float,
#     kd_null: float,
# ) -> torch.Tensor:
#     """
#     Operational Space Control with dynamically consistent nullspace (batched).
#     Returns joint torques tau: [N,7]
#     """
#     if dpose.ndim == 2:
#         dpose = dpose.unsqueeze(-1)  # [N,6,1]

#     N = j_eef.shape[0]
#     device = j_eef.device
#     dtype = j_eef.dtype

#     mm_inv = torch.inverse(mm_7)      # [N,7,7]
#     jT = j_eef.transpose(1, 2)        # [N,7,6]

#     m_eef_inv = j_eef @ mm_inv @ jT   # [N,6,6]
#     m_eef = torch.inverse(m_eef_inv)  # [N,6,6]

#     # task-space wrench -> joint torque
#     hand_vel = hand_vel_6.unsqueeze(-1)  # [N,6,1]
#     u_task = jT @ m_eef @ (kp * dpose - kd * hand_vel)  # [N,7,1]

#     # nullspace posture
#     if q_des_7.ndim == 1:
#         q_des = q_des_7.view(1, 7, 1).to(device=device, dtype=dtype)
#     else:
#         q_des = q_des_7.to(device=device, dtype=dtype)

#     q_err = (q_des - q_7 + math.pi) % (2 * math.pi) - math.pi  # [N,7,1]
#     u_null = kd_null * (-qd_7) + kp_null * q_err               # [N,7,1]
#     u_null = mm_7 @ u_null                                     # [N,7,1]

#     # dynamically consistent nullspace projector
#     j_eef_inv = m_eef @ j_eef @ mm_inv                         # [N,6,7]
#     I7 = torch.eye(7, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1)
#     proj = I7 - (jT @ j_eef_inv)                                # [N,7,7]

#     u = u_task + proj @ u_null                                  # [N,7,1]
#     return u.squeeze(-1)                                        # [N,7]



# class DemoCollector:
#     def __init__(self):
#         self.data = defaultdict(list)

#     def push(self, **kwargs):
#         for k, v in kwargs.items():
#             self.data[k].append(v.detach().cpu())

#     def save(self, path):
#         import h5py
#         with h5py.File(path, "w") as f:
#             for k, v in self.data.items():
#                 f.create_dataset(k, data=torch.stack(v).numpy())


# # -------------------------
# # Vectorized FSM
# # -------------------------

# class ScrewFSMVec:
#     """
#     Vectorized nut-bolt screwing FSM.
#     Adds terminal behavior: RELEASE + MOVE_AWAY + DONE
#     triggered when nut has moved down by SCREW_Z_TRAVEL along world Z since start of SCREW_MOTION.
#     """

#     # original state ids
#     GO_ABOVE_NUT       = 0
#     PREP_GRIP          = 1
#     GRIP               = 2
#     LIFT               = 3
#     GO_ABOVE_BOLT      = 4
#     GO_ON_BOLT         = 5
#     LOOSEN_GRIP        = 6
#     SCREW_MOTION       = 7
#     UNGRIP_SCREW       = 8
#     ROTATE_BACK        = 9
#     BACK_TO_SCREW_GRIP = 10

#     # new terminal states
#     RELEASE_DONE       = 11
#     MOVE_AWAY          = 12
#     DONE               = 13

#     def __init__(
#         self,
#         num_envs: int,
#         sim_dt: float,
#         nut_height: float,
#         bolt_height: float,
#         screw_speed: float,
#         screw_limit_angle: float,
#         device: str,
#         screw_z_travel: float = 0.44,
#     ):
#         self.N = int(num_envs)
#         self.dt = float(sim_dt)
#         self.nut_h = float(nut_height)
#         self.bolt_h = float(bolt_height)
#         self.screw_speed = float(screw_speed)
#         self.screw_limit = float(screw_limit_angle)
#         self.device = torch.device(device)

#         # completion threshold (world-z travel)
#         self.screw_z_travel = float(screw_z_travel)

#         # per-env state
#         self.state = torch.full((self.N,), self.GO_ABOVE_NUT, dtype=torch.int64, device=self.device)
#         self.screw_angle = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

#         # record z at start of screw motion
#         self.screw_start_z = torch.full((self.N,), float("nan"), dtype=torch.float32, device=self.device)

#         # outputs
#         self.dpose = torch.zeros((self.N, 6), dtype=torch.float32, device=self.device)
#         self.grip_sep = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

#         # constants (broadcast)
#         self.above_offset = torch.tensor([0.0, 0.0, 0.08 + self.bolt_h], device=self.device)
#         self.grip_offset  = torch.tensor([0.0, 0.0, 0.12 + self.nut_h], device=self.device)
#         self.lift_offset  = torch.tensor([0.0, 0.0, 0.15 + self.bolt_h], device=self.device)

#         self.above_bolt_offset = (torch.tensor([0.0, 0.0, self.bolt_h], device=self.device) + self.grip_offset)
#         self.on_bolt_offset    = (torch.tensor([0.0, 0.0, 0.8 * self.bolt_h], device=self.device) + self.grip_offset)

#         # terminal offsets
#         self.release_offset = torch.tensor([0.0, 0.0, 0.18 + self.bolt_h], device=self.device)  # lift a bit
#         self.away_offset    = torch.tensor([0.25, 0.0, 0.22 + self.bolt_h], device=self.device)  # retreat + lift

#         # hand-down quaternion (xyzw)
#         self.hand_down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

#         grab_angle = torch.tensor([torch.pi / 6.0], device=self.device)
#         grab_axis  = torch.tensor([0.0, 0.0, 1.0], device=self.device)
#         grab_q = quat_from_angle_axis(grab_angle, grab_axis).squeeze(0)  # [4]
#         self.nut_grab_q = quat_mul(grab_q, self.hand_down_q)  # [4]

#         self.screw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)

#     def reset(self, env_ids: torch.Tensor):
#         self.state[env_ids] = self.GO_ABOVE_NUT
#         self.screw_angle[env_ids] = 0.0
#         self.screw_start_z[env_ids] = float("nan")
#         self.dpose[env_ids] = 0.0
#         self.grip_sep[env_ids] = 0.08

#     def update(self, nut_pose, bolt_pose, hand_pose, current_grip_sep):
#         """
#         nut_pose, bolt_pose, hand_pose: [N,7] (x,y,z,qx,qy,qz,qw)
#         current_grip_sep: [N]
#         Returns:
#           dpose: [N,6]
#           grip_sep: [N]
#           state: [N]
#         """
#         N = self.N
#         dev = self.device

#         nut_pos  = nut_pose[:, 0:3]
#         nut_q    = nut_pose[:, 3:7]
#         bolt_pos = bolt_pose[:, 0:3]
#         hand_pos = hand_pose[:, 0:3]
#         hand_q   = hand_pose[:, 3:7]

#         target_pos  = torch.empty((N, 3), dtype=torch.float32, device=dev)
#         target_q    = torch.empty((N, 4), dtype=torch.float32, device=dev)
#         target_sep  = torch.empty((N,),    dtype=torch.float32, device=dev)

#         # defaults
#         target_pos[:] = hand_pos
#         target_q[:]   = self.hand_down_q
#         target_sep[:] = 0.08

#         s = self.state

#         # masks
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

#         m_release_done   = (s == self.RELEASE_DONE)
#         m_move_away      = (s == self.MOVE_AWAY)
#         m_done           = (s == self.DONE)

#         # GO_ABOVE_NUT
#         if m_go_above_nut.any():
#             target_sep[m_go_above_nut] = 0.08
#             target_pos[m_go_above_nut] = nut_pos[m_go_above_nut] + self.above_offset
#             target_q[m_go_above_nut]   = self.hand_down_q

#         # PREP_GRIP
#         if m_prep_grip.any():
#             target_sep[m_prep_grip] = 0.08
#             target_pos[m_prep_grip] = nut_pos[m_prep_grip] + self.grip_offset
#             target_q[m_prep_grip]   = quat_mul_const(nut_q[m_prep_grip], self.nut_grab_q)

#         # GRIP
#         if m_grip.any():
#             target_sep[m_grip] = 0.0
#             target_pos[m_grip] = nut_pos[m_grip] + self.grip_offset
#             target_q[m_grip]   = quat_mul_const(nut_q[m_grip], self.nut_grab_q)

#         # LIFT
#         if m_lift.any():
#             target_sep[m_lift] = 0.0
#             pos = nut_pos[m_lift].clone()
#             pos[:, 2] = bolt_pos[m_lift, 2] + 0.004
#             target_pos[m_lift] = pos + self.lift_offset
#             target_q[m_lift]   = self.hand_down_q

#         # GO_ABOVE_BOLT
#         if m_go_above_bolt.any():
#             target_sep[m_go_above_bolt] = 0.0
#             target_pos[m_go_above_bolt] = bolt_pos[m_go_above_bolt] + self.above_bolt_offset
#             target_q[m_go_above_bolt]   = self.hand_down_q

#         # GO_ON_BOLT
#         if m_go_on_bolt.any():
#             target_sep[m_go_on_bolt] = 0.0
#             pos = bolt_pos[m_go_on_bolt].clone()
#             pos[:, 2] = bolt_pos[m_go_on_bolt, 2]
#             target_pos[m_go_on_bolt] = pos + self.on_bolt_offset
#             target_q[m_go_on_bolt]   = self.hand_down_q

#         # LOOSEN_GRIP
#         if m_loosen_grip.any():
#             target_sep[m_loosen_grip] = 0.037
#             target_pos[m_loosen_grip] = bolt_pos[m_loosen_grip] + self.on_bolt_offset
#             target_q[m_loosen_grip]   = self.hand_down_q

#         # SCREW_MOTION
#         if m_screw_motion.any():
#             target_sep[m_screw_motion] = 0.037
#             pos = bolt_pos[m_screw_motion].clone()
#             pos[:, 2] = nut_pos[m_screw_motion, 2]
#             target_pos[m_screw_motion] = pos + self.grip_offset

#             # keep rotating while screwing
#             self.screw_angle[m_screw_motion] -= self.dt * self.screw_speed
#             ang = self.screw_angle[m_screw_motion].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_screw_motion] = quat_mul_const(screw_q, self.hand_down_q)

#         # UNGRIP_SCREW (kept for compatibility; may never be reached if z-travel triggers completion)
#         if m_ungrip_screw.any():
#             target_sep[m_ungrip_screw] = 0.06
#             pos = bolt_pos[m_ungrip_screw].clone()
#             pos[:, 2] = nut_pos[m_ungrip_screw, 2]
#             target_pos[m_ungrip_screw] = pos + self.grip_offset
#             ang = self.screw_angle[m_ungrip_screw].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_ungrip_screw] = quat_mul_const(screw_q, self.hand_down_q)

#         # ROTATE_BACK
#         if m_rotate_back.any():
#             target_sep[m_rotate_back] = 0.06
#             pos = bolt_pos[m_rotate_back].clone()
#             pos[:, 2] = nut_pos[m_rotate_back, 2]
#             target_pos[m_rotate_back] = pos + self.grip_offset
#             self.screw_angle[m_rotate_back] += self.dt * (2.0 * self.screw_speed)
#             ang = self.screw_angle[m_rotate_back].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_rotate_back] = quat_mul_const(screw_q, self.hand_down_q)

#         # BACK_TO_SCREW_GRIP
#         if m_back_to_grip.any():
#             target_sep[m_back_to_grip] = 0.037
#             pos = bolt_pos[m_back_to_grip].clone()
#             pos[:, 2] = nut_pos[m_back_to_grip, 2]
#             target_pos[m_back_to_grip] = pos + self.grip_offset
#             ang = self.screw_angle[m_back_to_grip].unsqueeze(-1)
#             screw_q = quat_from_angle_axis(ang, self.screw_axis).squeeze(1)
#             target_q[m_back_to_grip] = quat_mul_const(screw_q, self.hand_down_q)

#         # RELEASE_DONE
#         if m_release_done.any():
#             target_sep[m_release_done] = 0.08
#             target_pos[m_release_done] = bolt_pos[m_release_done] + self.release_offset
#             target_q[m_release_done]   = self.hand_down_q

#         # MOVE_AWAY
#         if m_move_away.any():
#             target_sep[m_move_away] = 0.08
#             target_pos[m_move_away] = bolt_pos[m_move_away] + self.away_offset
#             target_q[m_move_away]   = self.hand_down_q

#         # DONE
#         if m_done.any():
#             target_sep[m_done] = 0.08
#             target_pos[m_done] = bolt_pos[m_done] + self.away_offset
#             target_q[m_done]   = self.hand_down_q

#         # compute dpose
#         pos_err = target_pos - hand_pos
#         rot_err = orientation_error_batched(target_q, hand_q)
#         self.dpose[:, 0:3] = pos_err
#         self.dpose[:, 3:6] = rot_err
#         self.grip_sep[:] = target_sep

#         # transitions
#         err_norm = torch.linalg.norm(self.dpose, dim=1)

#         gripped_0035 = (current_grip_sep < 0.035)
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

#         # LOOSEN_GRIP -> SCREW_MOTION (record start z)
#         loosen_done = m_loosen_grip & (err_norm < 2e-3) & ungripped_target
#         self.screw_angle[loosen_done] = 0.0
#         self.screw_start_z[loosen_done] = nut_pos[loosen_done, 2]
#         print(f"the screw_start_z: {self.screw_start_z[loosen_done]}")
#         self.state[loosen_done] = self.SCREW_MOTION

#         # SCREW_MOTION -> RELEASE_DONE when z travel reached
#         valid_start = torch.isfinite(self.screw_start_z)
#         z_travel = torch.zeros((N,), device=dev, dtype=torch.float32)
#         z_travel[valid_start] = self.screw_start_z[valid_start] - nut_pos[valid_start, 2]
#         screw_done_by_z = m_screw_motion & valid_start & (z_travel >= self.screw_z_travel)
#         self.state[screw_done_by_z] = self.RELEASE_DONE






#         # (Optional legacy) SCREW_MOTION -> UNGRIP_SCREW by angle (only if you never trigger z completion)
#         screw_done_by_angle = m_screw_motion & (~screw_done_by_z) & (self.screw_angle < -self.screw_limit)
#         self.state[screw_done_by_angle] = self.UNGRIP_SCREW

#         # UNGRIP_SCREW -> ROTATE_BACK
#         ungrip_done = m_ungrip_screw & ungripped_target
#         self.state[ungrip_done] = self.ROTATE_BACK

#         # ROTATE_BACK -> BACK_TO_SCREW_GRIP
#         back_done = m_rotate_back & (self.screw_angle > 0.99 * self.screw_limit)
#         self.state[back_done] = self.BACK_TO_SCREW_GRIP

#         # BACK_TO_SCREW_GRIP -> SCREW_MOTION
#         gripped_near_target = (current_grip_sep < (target_sep * 1.01))
#         back_to_grip_done = m_back_to_grip & (err_norm < 2e-3) & gripped_near_target
#         self.screw_angle[back_to_grip_done] = self.screw_limit
#         self.state[back_to_grip_done] = self.SCREW_MOTION

#         # RELEASE_DONE -> MOVE_AWAY (wait until gripper is open enough)
#         release_done = m_release_done & (err_norm < 2e-3) & (current_grip_sep > 0.075)
#         self.state[release_done] = self.MOVE_AWAY

#         # MOVE_AWAY -> DONE
#         away_done = m_move_away & (err_norm < 2e-3)
#         self.state[away_done] = self.DONE

#         return self.dpose, self.grip_sep, self.state


# # -------------------------
# # Main
# # -------------------------

# def main():
#     np.random.seed(42)
#     torch.set_printoptions(precision=4, sci_mode=False)

#     # acquire gym
#     gym = gymapi.acquire_gym()

#     # args
#     custom_parameters = [
#         {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments"},
#         {"name": "--controller", "type": str, "default": "ik", "help": "Controller: ik or osc"},
#         {"name": "--screw_z_travel", "type": float, "default": 0.4, "help": "Nut z travel (m) to trigger release"},
#         {"name": "--record_demo", "type": bool, "default": True},
#         {"name": "--demo_path", "type": str, "default": "demos/screw_demo.h5"},
#         {"name": "--camera_res", "type": int, "default": 128},
#         {"name": "--save_every", "type": int, "default": 1},
#     ]
#     args = gymutil.parse_arguments(
#         description="Franka IK/OSC Nut-Bolt Screwing + Release on z-travel",
#         custom_parameters=custom_parameters,
#     )

#     # Force GPU sim
#     if not args.use_gpu or args.use_gpu_pipeline:
#         print("Forcing GPU sim - CPU sim not supported by SDF")
#         args.use_gpu = True
#         args.use_gpu_pipeline = True

#     device = args.sim_device
#     use_osc = (args.controller.lower() == "osc")

#     # sim params
#     sim_params = gymapi.SimParams()
#     sim_params.up_axis = gymapi.UP_AXIS_Z
#     sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
#     sim_params.dt = 1.0 / 60.0
#     sim_params.substeps = 2
#     sim_params.use_gpu_pipeline = args.use_gpu_pipeline

#     if args.physics_engine == gymapi.SIM_PHYSX:
#         sim_params.physx.solver_type = 1
#         sim_params.physx.num_position_iterations = 32
#         sim_params.physx.num_velocity_iterations = 1
#         sim_params.physx.rest_offset = 0.0
#         sim_params.physx.contact_offset = 0.005
#         sim_params.physx.friction_offset_threshold = 0.01
#         sim_params.physx.friction_correlation_distance = 0.0005
#         sim_params.physx.num_threads = args.num_threads
#         sim_params.physx.use_gpu = args.use_gpu
#     else:
#         raise RuntimeError("This example requires PhysX")
    
#     def create_camera(env, parent_actor, parent_body, offset, width, height):
#         cam_props = gymapi.CameraProperties()
#         cam_props.width = width
#         cam_props.height = height
#         cam_props.enable_tensors = True

#         cam = gym.create_camera_sensor(env, cam_props)

#         local_tf = gymapi.Transform()
#         local_tf.p = gymapi.Vec3(*offset)

#         gym.attach_camera_to_body(
#             cam,
#             env,
#             parent_body,
#             local_tf,
#             gymapi.FOLLOW_TRANSFORM,
#         )
#         return cam


#     # create sim
#     sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
#     if sim is None:
#         raise RuntimeError("Failed to create sim")

#     viewer = gym.create_viewer(sim, gymapi.CameraProperties())
#     if viewer is None:
#         raise RuntimeError("Failed to create viewer")

#     asset_root = "../../assets"

#     # ground
#     plane_params = gymapi.PlaneParams()
#     plane_params.normal = gymapi.Vec3(0, 0, 1)
#     gym.add_ground(sim, plane_params)

#     # table
#     table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
#     table_opts = gymapi.AssetOptions()
#     table_opts.fix_base_link = True
#     table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_opts)

#     # bolt
#     bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
#     bolt_opts = gymapi.AssetOptions()
#     bolt_opts.fix_base_link = True
#     bolt_opts.thickness = 0.0
#     bolt_opts.density = 800.0
#     bolt_opts.disable_gravity = False
#     bolt_opts.enable_gyroscopic_forces = True
#     bolt_asset = gym.load_asset(sim, asset_root, bolt_file, bolt_opts)

#     # nut
#     nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
#     nut_opts = gymapi.AssetOptions()
#     nut_opts.fix_base_link = False
#     nut_opts.thickness = 0.0
#     nut_opts.density = 800.0
#     nut_opts.disable_gravity = False
#     nut_opts.enable_gyroscopic_forces = True
#     nut_asset = gym.load_asset(sim, asset_root, nut_file, nut_opts)

#     # franka
#     franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
#     franka_opts = gymapi.AssetOptions()
#     franka_opts.armature = 0.01
#     franka_opts.fix_base_link = True
#     franka_opts.disable_gravity = True
#     franka_opts.flip_visual_attachments = True
#     franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, franka_opts)

#     # dof props
#     franka_dof_props = gym.get_asset_dof_properties(franka_asset)
#     franka_lower_limits = franka_dof_props["lower"]
#     franka_upper_limits = franka_dof_props["upper"]
#     franka_ranges = franka_upper_limits - franka_lower_limits
#     franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

#     # controller-specific arm mode
#     if use_osc:
#         franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
#         franka_dof_props["stiffness"][:7].fill(0.0)
#         franka_dof_props["damping"][:7].fill(0.0)
#     else:
#         franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
#         franka_dof_props["stiffness"][:7].fill(400.0)
#         franka_dof_props["damping"][:7].fill(40.0)

#     # grippers always position
#     franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
#     franka_dof_props["stiffness"][7:].fill(800.0)
#     franka_dof_props["damping"][7:].fill(40.0)

#     franka_num_dofs = gym.get_asset_dof_count(franka_asset)
#     default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
#     default_dof_pos[:7] = franka_mids[:7]
#     default_dof_pos[7:] = franka_upper_limits[7:]  # open grippers
#     default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

#     # hand index
#     franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
#     franka_hand_index = franka_link_dict["panda_hand"]

#     # env layout
#     num_envs = int(args.num_envs)
#     num_per_row = int(math.sqrt(num_envs))
#     spacing = 1.0
#     env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
#     env_upper = gymapi.Vec3(spacing, spacing, spacing)
#     print(f"Creating {num_envs} environments | controller={args.controller}")

#     franka_pose = gymapi.Transform()
#     franka_pose.p = gymapi.Vec3(0, 0, 0)

#     table_pose = gymapi.Transform()
#     table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

#     bolt_pose = gymapi.Transform()
#     nut_pose = gymapi.Transform()

#     envs = []
#     nut_idxs = []
#     bolt_idxs = []
#     hand_idxs = []

#     bolt_actor_idxs = []
#     nut_actor_idxs = []

#     franka_handles = []
#     nut_handles = []
#     bolt_handles = []

#     actor_handles = {}
#     actor_ids_sim = defaultdict(list)
#     # franka_actor_ids_sim = []  # within-sim indices
#     # nut_actor_ids_sim = []  # within-sim indices
#     # bolt_actor_ids_sim = []  # within-sim indices
#     # table_actor_ids_sim = []  # within-sim indices
#     actor_count = 0

#     # create envs/actors
#     for i in range(num_envs):
#         env = gym.create_env(sim, env_lower, env_upper, num_per_row)
#         envs.append(env)

#         # table
#         table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
#         actor_handles['table'] = table_handle
#         actor_ids_sim['table'].append(actor_count)
#         actor_count += 1

#         # bolt placement
#         bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
#         bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.0)
#         bolt_pose.p.z = table_dims.z
#         bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
#         bolt_handle = gym.create_actor(env, bolt_asset, bolt_pose, "bolt", i, 0)
#         bolt_handles.append(bolt_handle)
#         bolt_actor_idxs.append(gym.get_actor_index(env, bolt_handle, gymapi.DOMAIN_SIM))

#         # tweak bolt shape friction
#         bolt_props = gym.get_actor_rigid_shape_properties(env, bolt_handle)
#         bolt_props[0].friction = 0.0
#         bolt_props[0].rolling_friction = 0.0
#         bolt_props[0].torsion_friction = 0.0
#         bolt_props[0].restitution = 0.0
#         bolt_props[0].compliance = 0.0
#         bolt_props[0].thickness = 0.0
#         gym.set_actor_rigid_shape_properties(env, bolt_handle, bolt_props)

#         # bolt rigid body index (for rb_states)
#         bolt_idx = gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM)
#         bolt_idxs.append(bolt_idx)
#         actor_handles['bolt'] = bolt_handle
#         actor_ids_sim['bolt'].append(actor_count)
#         actor_count += 1

#         # nut placement
#         nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
#         nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
#         nut_pose.p.z = table_dims.z + 0.02
#         nut_handle = gym.create_actor(env, nut_asset, nut_pose, "nut", i, 0)
#         nut_handles.append(nut_handle)
#         nut_actor_idxs.append(gym.get_actor_index(env, nut_handle, gymapi.DOMAIN_SIM))

#         nut_props = gym.get_actor_rigid_shape_properties(env, nut_handle)
#         nut_props[0].friction = 0.2
#         nut_props[0].rolling_friction = 0.0
#         nut_props[0].torsion_friction = 0.0
#         nut_props[0].restitution = 0.0
#         nut_props[0].compliance = 0.0
#         nut_props[0].thickness = 0.0
#         gym.set_actor_rigid_shape_properties(env, nut_handle, nut_props)

#         # nut rigid body index
#         nut_idx = gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM)
#         nut_idxs.append(nut_idx)
#         actor_handles['nut'] = nut_handle
#         actor_ids_sim['nut'].append(actor_count)
#         actor_count += 1

#         # franka
#         franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)
#         franka_handles.append(franka_handle)
#         actor_handles['franka'] = franka_handle
#         actor_ids_sim['franka'].append(actor_count)
#         actor_count += 1
#         gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

#         hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
#         hand_idxs.append(hand_idx)
        

#     # camera
#     cam_pos = gymapi.Vec3(1, 0, 0.6)
#     cam_target = gymapi.Vec3(-1, 0, 0.5)
#     gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)


#     front_cams, wrist_cams, rear_cams = [], [], []

#     for env in envs:
#         front_cams.append(create_camera(env, None, table_body_id, [0.8, 0.0, 0.6],
#                                         args.camera_res, args.camera_res))
#         wrist_cams.append(create_camera(env, franka_handle, hand_body_id_env_actor,
#                                         [0.0, 0.0, 0.05], args.camera_res, args.camera_res))
#         rear_cams.append(create_camera(env, None, table_body_id, [-0.8, 0.0, 0.6],
#                                     args.camera_res, args.camera_res))

#     # prepare sim tensors
#     gym.prepare_sim(sim)

#     def get_cam_tensors(cam):
#         rgb = gymtorch.wrap_tensor(
#             gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_COLOR)
#         )
#         depth = gymtorch.wrap_tensor(
#             gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_DEPTH)
#         )
#         return rgb, depth


#     num_actors = int(actor_count / num_envs)  # per env
#     num_bodies = gym.get_env_rigid_body_count(env)  # per env
#     num_dofs = gym.get_env_dof_count(env)  # per env
#     actor_ids_sim_tensors = {key: torch.tensor(actor_ids_sim[key], dtype=torch.int32, device=device)
#                                       for key in actor_ids_sim.keys()}   

#     # tensors
#     _root_state = gym.acquire_actor_root_state_tensor(sim)  # shape = (num_envs * num_actors, 13)
#     _rb_states = gym.acquire_rigid_body_state_tensor(sim)
#     _dof_states = gym.acquire_dof_state_tensor(sim)

#     root_state = gymtorch.wrap_tensor(_root_state)
#     rb_states = gymtorch.wrap_tensor(_rb_states)  # [num_bodies,13]  
#     dof_state = gymtorch.wrap_tensor(_dof_states)  # [num_envs*9,2]

#     root_pos = root_state.view(num_envs, num_actors, 13)[..., 0:3]
#     root_quat = root_state.view(num_envs, num_actors, 13)[..., 3:7]
#     root_linvel = root_state.view(num_envs, num_actors, 13)[..., 7:10]
#     root_angvel = root_state.view(num_envs, num_actors, 13)[..., 10:13]
#     rb_pos = rb_states.view(num_envs, num_bodies, 13)[..., 0:3]
#     rb_quat = rb_states.view(num_envs, num_bodies, 13)[..., 3:7]
#     rb_linvel = rb_states.view(num_envs, num_bodies, 13)[..., 7:10]
#     rb_angvel = rb_states.view(num_envs, num_bodies, 13)[..., 10:13]
#     # dof_states = dof_states.view(num_envs, num_dofs, 2)
#     dof_pos = dof_state.view(num_envs, num_dofs, 2)[..., 0]
#     dof_vel = dof_state.view(num_envs, num_dofs, 2)[..., 1]

#     _jacobian = gym.acquire_jacobian_tensor(sim, "franka")
#     jacobian = gymtorch.wrap_tensor(_jacobian)  # [N, 10, 6, 9]
#     j_eef = jacobian[:, franka_hand_index - 1, :, :7]  # [N,6,7]

#     # mass matrix for OSC
#     _massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
#     massmatrix = gymtorch.wrap_tensor(_massmatrix)  # typically [N,9,9]

#     # actions
#     pos_action = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)
#     effort_action = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

#     # initialize dof targets/states
#     dof_pos[:, 0:9] = torch.tensor(default_dof_pos, dtype=torch.float32, device=device)
#     gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
#     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_pos[:, 0:9].contiguous()))


#     # For extracting root pos/quat
#     nut_actor_id_env = gym.find_actor_index(env, 'nut', gymapi.DOMAIN_ENV)
#     bolt_actor_id_env = gym.find_actor_index(env, 'bolt', gymapi.DOMAIN_ENV)

#     # For extracting body pos/quat, force, and Jacobian
#     nut_rb_names = gym.get_actor_rigid_body_names(envs[0], nut_actor_id_env)
#     assert len(nut_rb_names) == 1, 'We assume that there is a single rigid body in the bulb asset and use the name to retrieve rb id'
#     nut_body_id_env = gym.find_actor_rigid_body_index(envs[0], nut_actor_id_env,
#                                                                     nut_rb_names[0], gymapi.DOMAIN_ENV)
#     bolt_rb_names = gym.get_actor_rigid_body_names(envs[0], bolt_actor_id_env)
#     assert len(bolt_rb_names) == 1, 'We assume that there is a single rigid body in the socket asset and use the name to retrieve rb id'
#     bolt_body_id_env = gym.find_actor_rigid_body_index(envs[0], bolt_actor_id_env,
#                                                                     bolt_rb_names[0], gymapi.DOMAIN_ENV)
#     hand_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle, 'panda_hand',
#                                                                     gymapi.DOMAIN_ENV)
#     left_finger_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle, 'panda_leftfinger',
#                                                                         gymapi.DOMAIN_ENV)
#     right_finger_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
#     left_fingertip_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                             'panda_leftfingertip',
#                                                                             gymapi.DOMAIN_ENV)
#     right_fingertip_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                             'panda_rightfingertip',
#                                                                             gymapi.DOMAIN_ENV)
#     fingertip_centered_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                                 'panda_fingertip_centered',
#                                                                                 gymapi.DOMAIN_ENV)
#     hand_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle, 'panda_hand',
#                                                                         gymapi.DOMAIN_ACTOR)
#     left_finger_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                                 'panda_leftfinger',
#                                                                                 gymapi.DOMAIN_ACTOR)
#     right_finger_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                                 'panda_rightfinger',
#                                                                                 gymapi.DOMAIN_ACTOR)
#     left_fingertip_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                                     'panda_leftfingertip',
#                                                                                     gymapi.DOMAIN_ACTOR)
#     right_fingertip_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                                     'panda_rightfingertip',
#                                                                                     gymapi.DOMAIN_ACTOR)
#     fingertip_centered_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
#                                                                                         'panda_fingertip_centered',
#                                                                                         gymapi.DOMAIN_ACTOR)
#     table_body_id = gym.find_actor_rigid_body_index(envs[0], actor_handles['table'],
#                                                                 'box', gymapi.DOMAIN_ENV)

#     franka_body_names = gym.get_actor_rigid_body_names(env, franka_handle)
#     franka_body_ids_env = dict()


#     for b_name in franka_body_names:
#         franka_body_ids_env[b_name] = gym.find_actor_rigid_body_index(envs[0],
#                                                                                 actor_handles['franka'],
#                                                                                 b_name, gymapi.DOMAIN_ENV)

#     nut_pos = root_pos[:, nut_actor_id_env, 0:3]
#     nut_quat = root_quat[:, nut_actor_id_env, 0:4]
#     nut_linvel = root_linvel[:, nut_actor_id_env, 0:3]
#     nut_angvel = root_angvel[:, nut_actor_id_env, 0:3]

#     bolt_pos = root_pos[:, bolt_actor_id_env, 0:3]
#     bolt_quat = root_quat[:, bolt_actor_id_env, 0:4]

#     # FSM
#     fsm = ScrewFSMVec(
#         num_envs=num_envs,
#         sim_dt=sim_params.dt,
#         nut_height=0.016,
#         bolt_height=0.1,
#         screw_speed=30.0/180.0*math.pi,
#         screw_limit_angle=60.0/180.0*math.pi,
#         device=device,
#         screw_z_travel=args.screw_z_travel,
#     )

#     # controller params
#     damping = 0.15  # IK
#     # OSC gains
#     osc_kp = 150.0
#     osc_kd = 2.0 * math.sqrt(osc_kp)
#     osc_kp_null = 10.0
#     osc_kd_null = 2.0 * math.sqrt(osc_kp_null)

#     # sim loop
#     while not gym.query_viewer_has_closed(viewer):
#         gym.simulate(sim)
#         gym.fetch_results(sim, True)

#         gym.refresh_rigid_body_state_tensor(sim)
#         gym.refresh_dof_state_tensor(sim)
#         gym.refresh_jacobian_tensors(sim)
#         # Some builds require explicit refresh; try if available:
#         try:
#             gym.refresh_mass_matrix_tensors(sim)
#         except Exception:
#             pass

#         # gather poses
#         nut_poses = rb_states[nut_idxs, :7]
#         bolt_poses = rb_states[bolt_idxs, :7]
#         hand_poses = rb_states[hand_idxs, :7]

#         # current grip separation (sum of finger joints)
#         dof_pos_f = dof_pos[:, 0:9]  # [N,9]
#         dof_vel_f = dof_vel[:, 0:9]  # [N,9]
#         cur_grip_sep = dof_pos_f[:, 7] + dof_pos_f[:, 8]  # [N]

#         d_pose, grip_sep, fsm_state = fsm.update(nut_poses, bolt_poses, hand_poses, cur_grip_sep)

#         # gripper targets always position
#         pos_action[:, 7] = 0.5 * grip_sep
#         pos_action[:, 8] = 0.5 * grip_sep

#         if not use_osc:
#             # IK -> position targets for arm
#             dq = control_ik_batched(d_pose, j_eef, damping)
#             pos_action[:, :7] = dof_pos_f[:, :7] + dq
#             gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action.contiguous()))
#         else:
#             # OSC -> torque for arm, position for grippers
#             hand_linvel = rb_states[hand_idxs, 7:10]
#             hand_angvel = rb_states[hand_idxs, 10:13]
#             hand_vel_6 = torch.cat([hand_linvel, hand_angvel], dim=1)  # [N,6]

#             mm_7 = massmatrix[:, :7, :7]   # [N,7,7]
#             q_7 = dof_pos[:, :7, :]        # [N,7,1]
#             qd_7 = dof_vel[:, :7, :]       # [N,7,1]

#             tau = control_osc_batched(
#                 dpose=d_pose,
#                 j_eef=j_eef,
#                 mm_7=mm_7,
#                 q_7=q_7,
#                 qd_7=qd_7,
#                 hand_vel_6=hand_vel_6,
#                 q_des_7=default_dof_pos_tensor[:7],
#                 kp=osc_kp,
#                 kd=osc_kd,
#                 kp_null=osc_kp_null,
#                 kd_null=osc_kd_null,
#             )

#             effort_action.zero_()
#             effort_action[:, :7] = tau

#             gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action.contiguous()))
#             gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action.contiguous()))

#         # render
#         gym.step_graphics(sim)
#         gym.draw_viewer(viewer, sim, False)
#         gym.sync_frame_time(sim)

#     gym.destroy_viewer(viewer)
#     gym.destroy_sim(sim)


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Collect Demonstrations for Imitation Learning
============================================

Franka Nut-Bolt Screwing with:
- Vectorized FSM
- IK or OSC control
- Multiview RGB-D observations
- Robot state + actions
- HDF5 output (BC / Diffusion / ACT ready)

Usage:
------
python collect_screw_demos.py --num_envs 4 --controller ik
python collect_screw_demos.py --num_envs 4 --controller osc
"""
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import (
    to_torch,
    quat_mul,
    quat_conjugate,
    quat_from_angle_axis,
)
import os
import math
import h5py
import numpy as np
import torch
from collections import defaultdict


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


# =============================================================================
# Quaternion helpers
# =============================================================================

# @torch.jit.script
# def orientation_error(desired, current):
#     cc = quat_conjugate(current)
#     q_r = quat_mul(desired, cc)
#     return q_r[:, 0:3] * torch.sign(q_r[:, 3:4])


# @torch.jit.script
# def orientation_error_batched(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
#     """
#     desired/current: [N,4] (xyzw)
#     returns: [N,3]
#     """
#     cc = quat_conjugate(current)
#     q_r = quat_mul(desired, cc)
#     sign = torch.sign(q_r[:, 3:4])
#     return q_r[:, 0:3] * sign

# def quat_mul_const(a, q):
#     return quat_mul(a, q.unsqueeze(0).expand_as(a))


# =============================================================================
# Controllers
# =============================================================================

# def control_ik(dpose, j_eef, damping=0.15):
#     if dpose.ndim == 2:
#         dpose = dpose.unsqueeze(-1)
#     jT = j_eef.transpose(1, 2)
#     A = j_eef @ jT
#     I = torch.eye(6, device=A.device).unsqueeze(0)
#     dq = jT @ torch.linalg.solve(A + damping**2 * I, dpose)
#     return dq.squeeze(-1)

# def control_ik(dpose, j_eef, damping=0.05):
#     """
#     Damped Least Squares IK
#     dpose:  [N, 6] or [N, 6, 1]
#     j_eef:  [N, 6, 7]
#     returns dq: [N, 7, 1]
#     """

#     device = j_eef.device
#     dtype  = j_eef.dtype

#     # ---------------------------
#     # FORCE DEVICE + SHAPE
#     # ---------------------------
#     dpose = dpose.to(device=device, dtype=dtype)

#     if dpose.ndim == 2:
#         dpose = dpose.unsqueeze(-1)     # [N, 6, 1]

#     assert dpose.shape[1] == 6, f"dpose must have 6 DOF, got {dpose.shape}"

#     N = j_eef.shape[0]

#     jT = j_eef.transpose(1, 2)           # [N, 7, 6]
#     A  = j_eef @ jT                      # [N, 6, 6]

#     I = torch.eye(
#         6, device=device, dtype=dtype
#     ).unsqueeze(0).expand(N, -1, -1)

#     damping2 = damping * damping

#     x  = torch.linalg.solve(A + damping2 * I, dpose)  # [N, 6, 1]
#     dq = jT @ x                                       # [N, 7, 1]

#     return dq




# def control_osc(
#     dpose, j_eef, mm, q, qd, hand_vel,
#     kp=150.0, kd=25.0, kp_null=10.0, kd_null=6.0
# ):
#     if dpose.ndim == 2:
#         dpose = dpose.unsqueeze(-1)

#     mm7 = mm[:, :7, :7]
#     mm_inv = torch.inverse(mm7)
#     jT = j_eef.transpose(1, 2)

#     lambda_inv = j_eef @ mm_inv @ jT
#     lambda_mat = torch.inverse(lambda_inv)

#     u_task = jT @ lambda_mat @ (kp * dpose - kd * hand_vel.unsqueeze(-1))

#     q_err = ((q - q.detach()) + math.pi) % (2 * math.pi) - math.pi
#     u_null = mm7 @ (kp_null * q_err - kd_null * qd)

#     proj = torch.eye(7, device=q.device).unsqueeze(0) - jT @ (lambda_mat @ j_eef @ mm_inv)
#     return (u_task + proj @ u_null).squeeze(-1)


# =============================================================================
# FSM (simplified terminal-safe version)
# =============================================================================

# class ScrewFSM:
#     SCREW = 0
#     RELEASE = 1
#     DONE = 2

#     def __init__(self, num_envs, z_target=0.4):
#         self.state = torch.zeros(num_envs, dtype=torch.long)
#         self.z_target = z_target
#         self.dpose = torch.zeros(num_envs, 6)
#         self.grip = torch.zeros(num_envs)

#     def update(self, nut_pos, hand_pos, hand_q, grip_sep):
#         self.dpose.zero_()
#         self.grip.fill_(0.04)

#         done = nut_pos[:, 2] <= self.z_target
#         self.state[done] = self.DONE

#         pos_err = nut_pos - hand_pos
#         self.dpose[:, :3] = pos_err

#         return self.dpose, self.grip, self.state

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


# =============================================================================
# Demo Collector
# =============================================================================

class DemoBuffer:
    def __init__(self):
        self.buf = defaultdict(list)

    def push(self, **kwargs):
        for k, v in kwargs.items():
            self.buf[k].append(v.detach().cpu())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            for k, v in self.buf.items():
                f.create_dataset(k, data=torch.stack(v).numpy())


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
        {"name": "--demo_path", "type": str, "default": "demos/screw_demo.h5"},
        {"name": "--camera_res", "type": int, "default": 128},
        {"name": "--save_every", "type": int, "default": 1},
        {"name": "--screw_z_travel", "type": float, "default": 0.4, "help": "Nut z travel (m) to trigger release"},
        {"name": "--record_demo", "type": bool, "default": True},
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
    
    def create_camera(env, parent_actor, parent_body, offset, width, height):
        cam_props = gymapi.CameraProperties()
        cam_props.width = width
        cam_props.height = height
        cam_props.enable_tensors = True

        cam = gym.create_camera_sensor(env, cam_props)

        local_tf = gymapi.Transform()
        local_tf.p = gymapi.Vec3(*offset)

        gym.attach_camera_to_body(
            cam,
            env,
            parent_body,
            local_tf,
            gymapi.FOLLOW_TRANSFORM,
        )
        return cam


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

    actor_handles = {}
    actor_ids_sim = defaultdict(list)
    # franka_actor_ids_sim = []  # within-sim indices
    # nut_actor_ids_sim = []  # within-sim indices
    # bolt_actor_ids_sim = []  # within-sim indices
    # table_actor_ids_sim = []  # within-sim indices
    actor_count = 0

    # create envs/actors
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # table
        table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
        actor_handles['table'] = table_handle
        actor_ids_sim['table'].append(actor_count)
        actor_count += 1

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
        actor_handles['bolt'] = bolt_handle
        actor_ids_sim['bolt'].append(actor_count)
        actor_count += 1

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
        actor_handles['nut'] = nut_handle
        actor_ids_sim['nut'].append(actor_count)
        actor_count += 1

        # franka
        franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)
        franka_handles.append(franka_handle)
        actor_handles['franka'] = franka_handle
        actor_ids_sim['franka'].append(actor_count)
        actor_count += 1
        gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)
        

    # camera
    cam_pos = gymapi.Vec3(1, 0, 0.6)
    cam_target = gymapi.Vec3(-1, 0, 0.5)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)


    # For extracting root pos/quat
    nut_actor_id_env = gym.find_actor_index(env, 'nut', gymapi.DOMAIN_ENV)
    bolt_actor_id_env = gym.find_actor_index(env, 'bolt', gymapi.DOMAIN_ENV)

    # For extracting body pos/quat, force, and Jacobian
    nut_rb_names = gym.get_actor_rigid_body_names(envs[0], nut_actor_id_env)
    assert len(nut_rb_names) == 1, 'We assume that there is a single rigid body in the bulb asset and use the name to retrieve rb id'
    nut_body_id_env = gym.find_actor_rigid_body_index(envs[0], nut_actor_id_env,
                                                                    nut_rb_names[0], gymapi.DOMAIN_ENV)
    bolt_rb_names = gym.get_actor_rigid_body_names(envs[0], bolt_actor_id_env)
    assert len(bolt_rb_names) == 1, 'We assume that there is a single rigid body in the socket asset and use the name to retrieve rb id'
    bolt_body_id_env = gym.find_actor_rigid_body_index(envs[0], bolt_actor_id_env,
                                                                    bolt_rb_names[0], gymapi.DOMAIN_ENV)
    hand_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle, 'panda_hand',
                                                                    gymapi.DOMAIN_ENV)
    left_finger_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle, 'panda_leftfinger',
                                                                        gymapi.DOMAIN_ENV)
    right_finger_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                            'panda_rightfinger', gymapi.DOMAIN_ENV)
    left_fingertip_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                            'panda_leftfingertip',
                                                                            gymapi.DOMAIN_ENV)
    right_fingertip_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                            'panda_rightfingertip',
                                                                            gymapi.DOMAIN_ENV)
    fingertip_centered_body_id_env = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                                'panda_fingertip_centered',
                                                                                gymapi.DOMAIN_ENV)
    hand_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle, 'panda_hand',
                                                                        gymapi.DOMAIN_ACTOR)
    left_finger_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                                'panda_leftfinger',
                                                                                gymapi.DOMAIN_ACTOR)
    right_finger_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                                'panda_rightfinger',
                                                                                gymapi.DOMAIN_ACTOR)
    left_fingertip_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                                    'panda_leftfingertip',
                                                                                    gymapi.DOMAIN_ACTOR)
    right_fingertip_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                                    'panda_rightfingertip',
                                                                                    gymapi.DOMAIN_ACTOR)
    fingertip_centered_body_id_env_actor = gym.find_actor_rigid_body_index(env, franka_handle,
                                                                                        'panda_fingertip_centered',
                                                                                        gymapi.DOMAIN_ACTOR)


    franka_body_names = gym.get_actor_rigid_body_names(env, franka_handle)
    franka_body_ids_env = dict()


    for b_name in franka_body_names:
        franka_body_ids_env[b_name] = gym.find_actor_rigid_body_index(envs[0],
                                                                                actor_handles['franka'],
                                                                                b_name, gymapi.DOMAIN_ENV)

    table_body_id = gym.find_actor_rigid_body_index(envs[0], actor_handles['table'],
                                                                'box', gymapi.DOMAIN_ENV)

    # get the camera readings
    front_cams, wrist_cams, rear_cams = [], [], []

    for env in envs:
        front_cams.append(create_camera(env, None, table_body_id, [0.8, 0.0, 0.6],
                                        args.camera_res, args.camera_res))
        wrist_cams.append(create_camera(env, franka_handle, hand_body_id_env_actor,
                                        [0.0, 0.0, 0.05], args.camera_res, args.camera_res))
        rear_cams.append(create_camera(env, None, table_body_id, [-0.8, 0.0, 0.6],
                                    args.camera_res, args.camera_res))

    # prepare sim tensors
    gym.prepare_sim(sim)

    def get_cam_tensors(cam):
        rgb = gymtorch.wrap_tensor(
            gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_COLOR)
        )
        depth = gymtorch.wrap_tensor(
            gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_DEPTH)
        )
        return rgb, depth


    num_actors = int(actor_count / num_envs)  # per env
    num_bodies = gym.get_env_rigid_body_count(env)  # per env
    num_dofs = gym.get_env_dof_count(env)  # per env
    actor_ids_sim_tensors = {key: torch.tensor(actor_ids_sim[key], dtype=torch.int32, device=device)
                                      for key in actor_ids_sim.keys()}   

    # tensors
    _root_state = gym.acquire_actor_root_state_tensor(sim)  # shape = (num_envs * num_actors, 13)
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    _dof_states = gym.acquire_dof_state_tensor(sim)

    root_state = gymtorch.wrap_tensor(_root_state)
    rb_states = gymtorch.wrap_tensor(_rb_states)  # [num_bodies,13]  
    dof_state = gymtorch.wrap_tensor(_dof_states)  # [num_envs*9,2]

    root_pos = root_state.view(num_envs, num_actors, 13)[..., 0:3]
    root_quat = root_state.view(num_envs, num_actors, 13)[..., 3:7]
    root_linvel = root_state.view(num_envs, num_actors, 13)[..., 7:10]
    root_angvel = root_state.view(num_envs, num_actors, 13)[..., 10:13]
    rb_pos = rb_states.view(num_envs, num_bodies, 13)[..., 0:3]
    rb_quat = rb_states.view(num_envs, num_bodies, 13)[..., 3:7]
    rb_linvel = rb_states.view(num_envs, num_bodies, 13)[..., 7:10]
    rb_angvel = rb_states.view(num_envs, num_bodies, 13)[..., 10:13]
    # dof_states = dof_states.view(num_envs, num_dofs, 2)
    dof_pos = dof_state.view(num_envs, num_dofs, 2)[..., 0]
    dof_vel = dof_state.view(num_envs, num_dofs, 2)[..., 1]

    _jacobian = gym.acquire_jacobian_tensor(sim, "franka")
    jacobian = gymtorch.wrap_tensor(_jacobian)  # [N, 10, 6, 9]
    j_eef = jacobian[:, franka_hand_index - 1, :, :7]  # [N,6,7]

    # mass matrix for OSC
    _massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
    massmatrix = gymtorch.wrap_tensor(_massmatrix)  # typically [N,9,9]

    # actions
    pos_action = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)
    effort_action = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

    # initialize dof targets/states
    dof_pos[:, 0:9] = torch.tensor(default_dof_pos, dtype=torch.float32, device=device)
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_pos[:, 0:9].contiguous()))


    nut_pos = root_pos[:, nut_actor_id_env, 0:3]
    nut_quat = root_quat[:, nut_actor_id_env, 0:4]
    nut_linvel = root_linvel[:, nut_actor_id_env, 0:3]
    nut_angvel = root_angvel[:, nut_actor_id_env, 0:3]

    bolt_pos = root_pos[:, bolt_actor_id_env, 0:3]
    bolt_quat = root_quat[:, bolt_actor_id_env, 0:4]

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

    # demo
    demo_collector = DemoBuffer()

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
        dof_pos_f = dof_pos[:, 0:9]  # [N,9]
        dof_vel_f = dof_vel[:, 0:9]  # [N,9]
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

        if args.record_demo:
            # --- robot state ---
            demo_collector.push(
                q=dof_pos_f.clone(),
                qd=dof_vel_f.clone(),
                eef_pos=hand_poses[:, :3].clone(),
                eef_quat=hand_poses[:, 3:7].clone(),
                grip_sep=cur_grip_sep.unsqueeze(-1),
                fsm_state=fsm_state.clone(),
                controller=torch.full((num_envs,1), int(use_osc)),
            )

            # --- actions ---
            if not use_osc:
                demo_collector.push(
                    arm_action=dq.clone(),
                    gripper_action=pos_action[:, 7:9].clone(),
                )
            else:
                demo_collector.push(
                    arm_action=tau.clone(),
                    gripper_action=pos_action[:, 7:9].clone(),
                )

            for i in range(num_envs):
                frgb, fdepth = get_cam_tensors(front_cams[i])
                wrgb, wdepth = get_cam_tensors(wrist_cams[i])
                rrgb, rdepth = get_cam_tensors(rear_cams[i])

                demo_collector.push(
                    front_rgb=frgb,
                    front_depth=fdepth,
                    wrist_rgb=wrgb,
                    wrist_depth=wdepth,
                    rear_rgb=rrgb,
                    rear_depth=rdepth,
                )
        if torch.any(fsm_state == fsm.DONE):
            if args.record_demo:
                demo_collector.save(args.demo_path)
                print(f"[DEMO] Saved to {args.demo_path}")
            break

        # render
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
