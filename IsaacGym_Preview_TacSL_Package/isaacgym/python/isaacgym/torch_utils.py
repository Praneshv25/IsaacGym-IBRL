"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import torch
import numpy as np
from typing import Tuple


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    # q: (N,4)
    denom = torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True)).clamp(min=1e-8)
    return q / denom


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    # matrix: (N,3,3) -> quat (N,4) in (x,y,z,w)

    m00 = matrix[:, 0, 0]
    m11 = matrix[:, 1, 1]
    m22 = matrix[:, 2, 2]
    trace = m00 + m11 + m22

    qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 0.5
    denom = 4.0 * qw + 1e-8

    qx = (matrix[:, 2, 1] - matrix[:, 1, 2]) / denom
    qy = (matrix[:, 0, 2] - matrix[:, 2, 0]) / denom
    qz = (matrix[:, 1, 0] - matrix[:, 0, 1]) / denom

    quat = torch.stack((qx, qy, qz, qw), dim=-1)
    quat = normalize_quat(quat)
    return quat

@torch.jit.script
def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (N,4) (x,y,z,w)
    to rotation matrix (N,3,3)
    TorchScript compatible.
    """

    # Normalize for safety
    quat = normalize_quat(quat)

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - zw)
    m02 = 2.0 * (xz + yw)

    m10 = 2.0 * (xy + zw)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - xw)

    m20 = 2.0 * (xz - yw)
    m21 = 2.0 * (yz + xw)
    m22 = ww - xx - yy + zz

    row0 = torch.stack((m00, m01, m02), dim=-1)
    row1 = torch.stack((m10, m11, m12), dim=-1)
    row2 = torch.stack((m20, m21, m22), dim=-1)

    return torch.stack((row0, row1, row2), dim=-2)



@torch.jit.script
def rotvec_to_rotmat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    TorchScript-compatible axis-angle (rotvec) to rotation matrix.

    Args:
        rotvec: (N,3) or (...,3)

    Returns:
        rotmat: (...,3,3)
    """

    eps = 1e-6

    # Flatten leading dims for easier math
    orig_shape = rotvec.shape
    rotvec_flat = rotvec.reshape(-1, 3)

    theta = torch.norm(rotvec_flat, dim=1, keepdim=True)  # (B,1)
    theta2 = theta * theta

    # sin(theta)/theta with Taylor fallback
    sin_theta = torch.sin(theta)
    sin_term = sin_theta / (theta + eps)

    # (1 - cos(theta)) / theta^2 with fallback
    cos_theta = torch.cos(theta)
    cos_term = (1.0 - cos_theta) / (theta2 + eps)

    # Small angle correction (Taylor expansion)
    small = theta < eps
    if small.any():
        sin_term = torch.where(small, 1.0 - theta2 / 6.0, sin_term)
        cos_term = torch.where(small, 0.5 - theta2 / 24.0, cos_term)

    # Normalize axis
    axis = rotvec_flat / (theta + eps)

    x = axis[:, 0:1]
    y = axis[:, 1:2]
    z = axis[:, 2:3]

    zeros = torch.zeros_like(x)

    # Skew-symmetric matrix K
    K = torch.cat([
        zeros, -z,     y,
        z,     zeros, -x,
        -y,     x,    zeros
    ], dim=1).reshape(-1, 3, 3)

    # Identity
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).unsqueeze(0)
    I = I.expand(rotvec_flat.shape[0], 3, 3)

    # Rodrigues
    R = I + sin_term.unsqueeze(-1) * K + \
        cos_term.unsqueeze(-1) * torch.bmm(K, K)

    # Restore original batch shape
    out_shape = orig_shape[:-1] + (3, 3)
    return R.reshape(out_shape)

def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
