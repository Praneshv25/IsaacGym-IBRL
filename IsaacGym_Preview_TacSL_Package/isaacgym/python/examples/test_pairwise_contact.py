"""
Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Pairwise contact force reporting
-------------------------
Demonstrates how to access the pairwise contact force between all rigid bodies within an environment.
It currently does not report collision with the isaacgym's ground plane (gymapi.PlaneParams),
so as a work-around you can create wide box as the ground plane.

The important function call is:
    _pairwise_contact_force = gym.acquire_pairwise_contact_force_tensor(sim)  # shape = (num_envs * num_bodies * num_bodies, 3)

    pairwise_contact_force = gymtorch.wrap_tensor(_pairwise_contact_force).view(num_envs, num_bodies, num_bodies, 3)

Press 'R' to reset the  simulation
"""

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from math import sqrt
from collections import defaultdict
import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
        {"name": "--remove_box_slab", "action": "store_true", "help": "Add big box slab on top of the floor"},
        {"name": "--pyramid_size", "type": int, "default": 2, "help": "Number of balls on each side of the pyramid base"}])

# Always use GPU, as pairwise contact force is only implemented in GPU mode.
args.use_gpu = True
args.pipeline = 'cuda'

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = args.pipeline in ('gpu', 'cuda')

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load ball asset
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

box_width = 100
box_height = 0.5
box_asset = gym.create_box(sim, box_width, box_height, box_width, gymapi.AssetOptions())

# set up the env grid
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
np.random.seed(17)

actor_count = 0
actor_handles = {}
actor_ids_sim = defaultdict(list)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Balls in the same env should collide, but not with balls from different envs.
    # Use one group per env, and filtering masks set to 0.
    collision_group = i
    collision_filter = 0
    if not args.remove_box_slab:
        actor_name = 'box_platform'
        pose = gymapi.Transform()
        pose.p.y = box_height/2
        ahandle = gym.create_actor(env, box_asset, pose, actor_name, collision_group, collision_filter)
        actor_handles[actor_name] = ahandle
        actor_ids_sim[actor_name].append(actor_count)
        actor_count += 1

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # create ball pyramid
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    n = 2
    n = args.pyramid_size
    radius = 0.2
    ball_spacing = 2.5 * radius
    min_coord = -0.5 * (n - 1) * ball_spacing
    y = min_coord+4
    while n > 0:
        z = min_coord
        for j in range(n):
            x = min_coord
            for k in range(n):
                pose.p = gymapi.Vec3(x, 1.5 + y, z)

                # Balls in the same env should collide, but not with balls from different envs.
                # Use one group per env, and filtering masks set to 0.
                collision_group = i
                collision_filter = 0

                actor_name = f'ball_{n}_{j}_{k}'
                ahandle = gym.create_actor(env, asset, pose, actor_name, collision_group, collision_filter)
                gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                actor_handles[actor_name] = ahandle
                actor_ids_sim[actor_name].append(actor_count)
                actor_count += 1

                x += ball_spacing
            z += ball_spacing
        y += ball_spacing
        n -= 1
        min_coord = -0.5 * (n - 1) * ball_spacing

# get constants from env simulation
num_actors = gym.get_actor_count(envs[0])  # per env
num_bodies = gym.get_env_rigid_body_count(envs[0])  # per env
num_dofs = gym.get_env_dof_count(envs[0])  # per env

actor_ids_sim_tensors = {key: torch.tensor(actor_ids_sim[key], dtype=torch.int32, device=args.sim_device)
                         for key in actor_ids_sim.keys()}

# start GPU
gym.prepare_sim(sim)

# Acquire tensors
_root_state = gym.acquire_actor_root_state_tensor(sim)  # shape = (num_envs * num_actors, 13)
_net_contact_force = gym.acquire_net_contact_force_tensor(sim)  # shape = (num_envs * num_bodies, 3)
_pairwise_contact_force = gym.acquire_pairwise_contact_force_tensor(sim)  # shape = (num_envs * num_bodies * num_bodies, 3)

root_state = gymtorch.wrap_tensor(_root_state)
net_contact_force = gymtorch.wrap_tensor(_net_contact_force)
pairwise_contact_force = gymtorch.wrap_tensor(_pairwise_contact_force)

net_contact_force_view = net_contact_force.view(num_envs, num_bodies, 3)
pairwise_contact_force_view = pairwise_contact_force.view(num_envs, num_bodies, num_bodies, 3)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

# create a local copy of initial state, which we can send back for reset
gym.refresh_actor_root_state_tensor(sim)
initial_state = root_state.clone()


def freeze_sim_and_render():
    print('Ctrl-C to continue')
    while True:
        try:
            # update the viewer
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
        except:
            print('EXIT')
            break


while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            root_state[:] = initial_state[:]
            gym.set_actor_root_state_tensor_indexed(sim,
                                                    gymtorch.unwrap_tensor(root_state),
                                                    gymtorch.unwrap_tensor(actor_ids_sim_tensors),
                                                    len(actor_ids_sim_tensors))

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # compute and compare net force with collated pairwise force
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_pairwise_contact_force_tensor(sim)

    collated_pairwise_forces = pairwise_contact_force_view.sum(2)
    error = pairwise_contact_force_view.sum(2) - net_contact_force_view
    error_threshold = 1e-5
    # ignore the contact forces of the first body (box platform base) because it collides with the floor
    # assert torch.abs(error[:, 1:, :]).sum() < error_threshold, f'pairwise collation differ from net forces by {torch.abs(error).sum()}'
    print('\nError between aggregated pairwise contact force and the reported net contact force of each body')
    body_start_idx = 1 - int(args.remove_box_slab)
    print(f'sum error: {torch.abs(error[:, body_start_idx:, :]).sum()}')
    print(f'max error: {torch.abs(error[:, body_start_idx:, :]).max()}')

    if torch.abs(error[:, 1:, :]).max() > 0:
        import ipdb
        ipdb.set_trace()

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
