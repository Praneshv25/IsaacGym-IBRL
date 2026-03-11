"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Signed Distance Field (SDF) querying example
-------------------------------
An example that demonstrates how to query the SDF values at locations around a mesh.
This is done in two steps
1. Create the tensor used to hold the SDF result of the query. To do this specify
- the number of shapes of interest
- the number of queries that would be asked of each shape

2. Get the SDF result of specified shapes at specified location. To do this, specify
- the global indicies of the shape in simulation world.
- the query location in the local frame of the shape
"""

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import torch_utils
from collections import defaultdict
import itertools
import numpy as np
import torch
import os
import trimesh
from urdfpy import URDF


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="SDF Query Example")

assert 'cuda' in args.pipeline, 'Only works with GPU pipeline. Pass in args "--pipeline cuda"'

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = True
if args.use_gpu_pipeline:
    print("WARNING: Only works with GPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 3
spacing = 1.8
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []

# create nut asset with gravity disabled
asset_root = "../../assets"
asset_file_nut = "urdf/nut_bolt/nut_m4_tight_SI.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
print("Loading asset '%s' from '%s'" % (asset_file_nut, asset_root))
asset_nut = gym.load_asset(sim, asset_root, asset_file_nut, asset_options)


# create static box asset
asset_options.fix_base_link = True
asset_slab = gym.create_box(sim, 0.5, 0.1, 0.5, asset_options)

print('Creating %d environments' % num_envs)

actor_count = 0
actor_handles_dict = {}
actor_ids_sim = defaultdict(list)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env)

    dx = 0
    name = 'nut_{}'.format(dx)
    nut_handle = gym.create_actor(env, asset_nut, gymapi.Transform(p=gymapi.Vec3(dx, 0.1, 0.75)), name, i, 0)
    gym.set_rigid_body_color(env, nut_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.))
    actor_handles_dict['nut'] = nut_handle
    actor_ids_sim['nut'].append(actor_count)
    actor_count += 1


# Get global shape index that will be used by PhysX to compute SDF
shape_global_indicies = defaultdict(list)

for env_ptr in envs:
    # get rigid body shape of nut
    actor_handle = actor_handles_dict['nut']
    gym.get_actor_rigid_body_names(env_ptr, actor_handle)

    link_name = 'nut_m4_tight_SI'
    link_rb_id = gym.find_actor_rigid_body_index(env_ptr, actor_handle, link_name, gymapi.DOMAIN_ACTOR)

    rb_shape_indices = gym.get_actor_rigid_body_shape_indices(env_ptr, actor_handle)
    rb_shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
    link_rb_shape_id_actor = rb_shape_indices[link_rb_id].start
    print(f'\nlink_rb_shape_id_actor: {link_rb_shape_id_actor}')
    print(f'global index: {rb_shape_props[link_rb_shape_id_actor].global_index}')
    shape_global_indicies[f'{"nut"}_{link_name}'].append(rb_shape_props[link_rb_shape_id_actor].global_index)


num_shapes_per_env = len(shape_global_indicies.keys())
sdf_shape_global_ids_per_env = torch.zeros((num_envs, num_shapes_per_env), dtype=torch.int32, device=args.sim_device)
sdf_tensor_shape_idx_map = dict()
for idx, key in enumerate(shape_global_indicies.keys()):

    sdf_shape_global_ids_per_env[:, idx] = torch.tensor(shape_global_indicies[key], dtype=torch.int32, device=args.sim_device)
    sdf_tensor_shape_idx_map[key] = idx


num_queries_per_shape = 16
test_sdf_query = True
if test_sdf_query:
    # create sdf query view
    _sdf_tensor = gym.acquire_sdf_view_tensor(sim, num_shapes_per_env, num_queries_per_shape)
    sdf_tensor = gymtorch.wrap_tensor(_sdf_tensor)
    # print(f'sdf_tensor: \t {sdf_tensor}')
    print(f'sdf_tensor.shape: \t {sdf_tensor.shape}')


# Load URDF for trimesh computation
robot = URDF.load(os.path.join(asset_root, asset_file_nut))
mesh = robot.links[-1].visuals[0].geometry.mesh.meshes[0]
origin = robot.links[-1].visuals[0].origin
# mesh_transformed = mesh.apply_transform(origin)

mesh_dims = np.diff(mesh.bounds, axis=0)
mesh_com = mesh.center_mass


# start GPU sim
gym.prepare_sim(sim)

corners_3D = list(itertools.product(*[[-1, 1]]*3))

query_points = torch.zeros((num_envs, num_shapes_per_env, num_queries_per_shape, 3), dtype=torch.float32, device=args.sim_device)
query_points[:, :, :8, :3] = torch.tensor(corners_3D, device=args.sim_device)  # outer cube lattice
query_points[:, :, 8:, :3] = torch.tensor(corners_3D, device=args.sim_device) * .5  # inner cube lattice
query_points *= mesh_dims.max()/2
query_points /= 2
query_points += torch.tensor(mesh_com, device=query_points.device)

query_points_np = query_points[0].cpu().numpy()
pointcloud_query = trimesh.PointCloud(query_points_np.reshape([-1, 3]), colors=(0., 1., 0.))
print('\nVisualizing mesh in Trimesh')
print('Close the window to proceed...')
trimesh.Scene([mesh, pointcloud_query]).show()

# Trimesh SDF computation
sdf_oracle = trimesh.proximity.ProximityQuery(mesh)
distance_np_trimesh = sdf_oracle.signed_distance(query_points_np.reshape([-1, 3])).reshape(query_points_np.shape[:-1] + (1,))

# Approximate SDF gradient computation using surface normal
closest_np, _, _ = sdf_oracle.on_surface(query_points_np.reshape([-1, 3]))
closest_np = closest_np.reshape(query_points_np.shape)
normal = torch_utils.normalize(torch.tensor(closest_np - query_points_np, dtype=query_points.dtype, device=args.sim_device))
normal_np_trimesh = normal.cpu().numpy()


# look at the first env
cam_pos = gymapi.Vec3(6, 4.5, 3)
cam_target = gymapi.Vec3(-0.8, 0.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    if test_sdf_query:
        gym.refresh_sdf_view_tensor(sim,
                                    gymtorch.unwrap_tensor(sdf_shape_global_ids_per_env),
                                    gymtorch.unwrap_tensor(query_points))

        # # print(f'sdf_tensor: \t {sdf_tensor}')
        # print(f'PhysX SDF: {sdf_tensor[0, ..., 3:]}')
        # print(f'Trimesh SDF: {distance_np_trimesh}')
        print(f'\nDifference between PhysX and Trimesh SDF values: \n {sdf_tensor[0, ..., 3:].cpu().numpy() + distance_np_trimesh}')

        # print(f'PhysX SDF grad: {sdf_tensor[0, ..., :3]}')
        # print(f'Trimesh SDF grad: {normal_np_trimesh}')
        print(f'\nDifference between PhysX and Trimesh SDF Gradients: \n {sdf_tensor[0, ..., :3].cpu().numpy() + normal_np_trimesh}')
        print(f'\nDot product between PhysX and Trimesh SDF Gradients: \n {np.sum(sdf_tensor[0, ..., :3].cpu().numpy()* -normal_np_trimesh, axis=-1)}')
        import ipdb
        ipdb.set_trace()

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
