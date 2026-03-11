"""
Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Compliant contacts modelling
-------------------------
Demonstrates how to use compliant contacts between rigid bodies within an environment.
When the `compliance` and `compliant_damping` is set for a rigid body, it generates soft contacts (with spring-like interpenetration)
 when in collision with another body.

Usage requires getting and setting the corresponding rigid shape properties within a rigid body:
    --get_actor_rigid_shape_properties to acquire the rigid shape properties
    --Modify `compliance` and `compliant_damping` fields of the GymRigidShapeProperties object, 
    --set_actor_rigid_shape_properties to update the rigid shape properties within the physics simulator.

Use commandline arg `--use_acceleration_spring` to use acceleration springs for compliant contacts.

Press 'R' to reset the  simulation
"""

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
        {"name": "--use_acceleration_spring", "action": "store_true", "help": "Use acceleration spring."}, ])

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
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
    sim_params.physx.rest_offset = 0.00

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Env spacing
num_spheres = 5
mass = 1.0
gravity = 9.8  # abs(sim_params.gravity.y)
deflection = 5.0 * 1e-2  # cm
stiffness = mass * gravity / deflection
stiffnessDelta = stiffness
damping = 0.1 * stiffness
dampingDelta = 0.1 * damping

radius = 0.25
position_delta = 1.5
x0 = - position_delta * (num_spheres - 1) / 2.0

# set up the env grid
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = (num_spheres + 1) * position_delta
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []
sphere_handles = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
np.random.seed(17)

for env_id in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    color = np.array([0.0 / 255.0, 25.0 / 255.0, 1.0])
    color_delta = 1.0 / num_spheres

    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)

    collision_group = 0
    collision_filter = 0
    for i in range(num_spheres):
        print('='*20)
        print('Creating sphere actor ...')

        asset_options = gymapi.AssetOptions()
        volume = 4./3 * np.pi * np.power(radius, 3)
        density = mass / volume
        # density = mass / volume * (i+1)
        asset_options.density = density
        ball_asset = gym.create_sphere(sim, radius, asset_options)

        pose.p = gymapi.Vec3(x0 + i * position_delta, 0, 2)
        ahandle = gym.create_actor(env, ball_asset, pose, f'ball_{i}', collision_group, collision_filter)
        sphere_handles.append(ahandle)
        color[0] += color_delta
        gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))

        # change compliance
        rs_props = gym.get_actor_rigid_shape_properties(env, ahandle)
        rs_props[0].compliance = stiffness + (i + 1) % num_spheres * stiffnessDelta
        rs_props[0].compliant_damping = damping + i * dampingDelta
        rs_props[0].use_acceleration_spring = args.use_acceleration_spring
        print(f'restitution: {rs_props[0].restitution}, \t '
              f'compliance: {rs_props[0].compliance}, \t '
              f'compliant_damping: {rs_props[0].compliant_damping}, \t '
              f'use_acceleration_spring: {rs_props[0].use_acceleration_spring}\n')
        gym.set_actor_rigid_shape_properties(env, ahandle, rs_props)

        rb_props = gym.get_actor_rigid_body_properties(env, ahandle)
        # print(f'mass: {rb_props[0].mass}')

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(-2, 7, 2), gymapi.Vec3(0, 0, 2))

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

print('\n\n\nInspect dynamics ...')
for env_id in range(num_envs):
    for i in range(num_spheres):
        rs_props = gym.get_actor_rigid_shape_properties(envs[env_id], sphere_handles[i])
        print(f'restitution: {rs_props[0].restitution}, \t '
              f'compliance: {rs_props[0].compliance}, \t '
              f'compliant_damping: {rs_props[0].compliant_damping}')

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
