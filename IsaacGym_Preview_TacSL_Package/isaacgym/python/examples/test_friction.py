"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Apply Forces At Positions (apply_forces_at_pos.py)
----------------------------
This example shows how to apply rigid body forces at given positions using the tensor API.

"""

from signal import Handlers
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
from numpy import random
import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Box-ground friction test")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 0.2
plane_params.dynamic_friction = 0.2
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# load ball asset
asset_root = "../../assets"
asset_file = "urdf/cube.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

num_bodies = gym.get_asset_rigid_body_count(asset)
# print('num_bodies', num_bodies)

# default pose
pose = gymapi.Transform()
pose.p.z = 0.25

# set up the env grid
num_envs = 4
num_per_row = num_envs
env_spacing = 1.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# set random seed
np.random.seed(7)

envs = []
handles = []
asset_options = gymapi.AssetOptions()

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

scale = 10.0

friction_delta = 1.0 / float(num_envs - 1)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)

    gym.set_actor_scale(env, actor_handle, scale)

    handles.append(actor_handle)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    print("Default friction: ", shape_props[0].friction)

    # dynamic/static friction?
    # shape_props[0].friction = 0.0  # default = ?
    # shape_props[0].rolling_friction = 0.0  # default = 0.0
    # shape_props[0].torsion_friction = 0.0  # default = 0.0
    shape_props[0].friction = friction_delta * float(i)
    print("Number of shapes: ", len(shape_props))
    print(i, shape_props[0].friction)
    # shape_props[0].compliance = 0.0  # default = 0.0
    # shape_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)

mid = 0.5 * env_spacing * num_per_row + 0.5
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(15, mid, 3.0), gymapi.Vec3(0, mid, 0.5))

gym.prepare_sim(sim)

# create a local copy of initial state, which we can send back for reset
rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_tensor)
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)
rb_velocities = rb_states[:, 7:10].view(num_envs, num_bodies, 3)

# 1 simulation step to update state tensor
gym.simulate(sim)
gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim)

rb_states_copy = torch.clone(rb_states)

rb_states_copy[:, 8] = 7.0

gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(rb_states_copy))

force_offset = 0.2

frame_count = 0
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            print("Reset!")
            for i in range(num_envs):
                shape_props = gym.get_actor_rigid_shape_properties(envs[i], handles[i])
                print(shape_props[0].friction)
                # set new random friction between 0 and 1
                shape_props[0].friction = random.random()
                gym.set_actor_rigid_shape_properties(envs[i], handles[i], shape_props)
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(rb_states_copy))

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
