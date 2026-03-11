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

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch
import time


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("mjcf/nv_humanoid.xml", False),
]


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environemnts"},
        {"name": "--env_spacing", "type": float, "default": 2.0, "help": "Number of environemnts"},
        {"name": "--render", "action": "store_true", "help": "Enable viewer"},
        {"name": "--enable_pvd", "action": "store_false", "help": "Enable pvd capture"}
    ])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()

if args.enable_pvd:
    env_names = [
        "ant",
        "humanoid",
    ]

    pvd_file = env_names[args.asset_id] + "_envs_" + str(args.num_envs) + "_spacing_" + str(args.env_spacing) + "_pvd"

    # set environment variable to use custom pvd file
    import os
    os.environ["GYM_PVD_FILE"] = pvd_file

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.default_buffer_size_multiplier = 5.0
    sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
else:
    raise Exception("Wrong simulation engine")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
if args.render:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

# load ball asset
asset_root = "../../assets"
asset_file = asset_descriptors[args.asset_id].file_name
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies)

# default pose
pose = gymapi.Transform()
pose.p.z = 1.0

# set up the env grid
num_envs = args.num_envs
num_per_row = int(np.sqrt(num_envs))
env_spacing = args.env_spacing
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# set random seed
np.random.seed(17)

envs = []
handles = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    ahandle = gym.create_actor(env, asset, pose, "actor", i, 1)
    handles.append(ahandle)

    # generate random bright color
    # c = 0.5 + 0.5 * np.random.random(3)
    # color = gymapi.Vec3(c[0], c[1], c[2])
    # gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

mid = 0.5 * env_spacing * (num_per_row - 1)

if args.render:
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, mid, 5), gymapi.Vec3(0, mid, 1))

gym.prepare_sim(sim)

rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_tensor)
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)

force_offset = 0.2

num_benchmark_steps = 300
frame_count = 0

start_time = time.time_ns()

while frame_count < num_benchmark_steps:

    if (frame_count - 59) % 60 == 0:

        gym.refresh_rigid_body_state_tensor(sim)

        # set forces and force positions for ant root bodies (first body in each env)
        forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        force_positions = rb_positions.clone()
        forces[:, 0, 2] = 400
        force_positions[:, 0, 1] += force_offset
        gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

        force_offset = -force_offset

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if args.render:
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    frame_count += 1

end_time = time.time_ns()

run_time = (end_time - start_time) / 1e9
avg_time = run_time / num_benchmark_steps

# avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
avg_steps_second = float(num_envs)/avg_time

print(f"envs: {num_envs} steps/second {avg_steps_second:0.1f} avg_time {avg_time:0.9f}")


if args.render:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
