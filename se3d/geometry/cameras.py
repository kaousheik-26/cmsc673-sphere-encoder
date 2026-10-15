"""
Camera utilities. Convention everywhere: OpenCV (x right, y down, z forward),
world-to-camera extrinsics w2c = [R | t] (4x4), intrinsics K (3x3, pixels).
World frame for object renders: Z up (as SyncDreamer / Blender).
"""
import math

import numpy as np
import torch


def look_at_w2c(cam_pos, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """OpenCV w2c (4x4) for a camera at cam_pos looking at target."""
    c = np.asarray(cam_pos, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-8:  # looking straight up/down
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # y down
    R = np.stack([x, y, z], 0)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ c
    return w2c


def orbit_w2c(azimuth_deg, elevation_deg, radius, target=(0.0, 0.0, 0.0)):
    """SyncDreamer convention: azimuth from +X toward +Y, elevation above XY plane."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    pos = np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]) * radius
    return look_at_w2c(pos + np.asarray(target), target)


def intrinsics_from_fov(fov_deg, width, height=None):
    height = height or width
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def syncdreamer_K(size=256):
    # lens 35 mm, sensor 32 mm -> f = 35/32 * size; 280 px at 256 (camera-16.pkl)
    f = 35.0 / 32.0 * size
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])


def cam_center(w2c):
    R, t = w2c[:3, :3], w2c[:3, 3]
    return -R.T @ t


def relative_pose(w2c_src, w2c_tgt):
    """pose of target camera expressed in the source camera frame: T_tgt<-src.
    Returns the 4x4 that maps source-camera coordinates to target-camera coordinates."""
    return w2c_tgt @ np.linalg.inv(w2c_src)


def opengl_proj(K, width, height, near=0.05, far=20.0):
    """OpenGL clip matrix for an OpenCV pinhole (applied to OpenGL camera coords,
    i.e. OpenCV coords with y and z negated)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([
        [2 * fx / width, 0, 1 - 2 * cx / width, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ])


def plucker_rays(K, w2c, height, width, stride=1):
    """Plücker ray map [6, H/stride, W/stride] (d, o x d) in world frame,
    sampled at pixel centers."""
    K = torch.as_tensor(K, dtype=torch.float32)
    w2c = torch.as_tensor(w2c, dtype=torch.float32)
    ys, xs = torch.meshgrid(
        torch.arange(0, height, stride, dtype=torch.float32) + stride / 2,
        torch.arange(0, width, stride, dtype=torch.float32) + stride / 2,
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1)
    d_cam = pix @ torch.linalg.inv(K).T
    R = w2c[:3, :3]
    d = d_cam @ R  # R^T d_cam, row-vector form
    d = d / d.norm(dim=-1, keepdim=True)
    o = (-R.T @ w2c[:3, 3]).expand_as(d)
    return torch.cat([d, torch.cross(o, d, dim=-1)], -1).permute(2, 0, 1)


def unproject_depth(depth, K, w2c, mask=None):
    """depth [H,W] (z along optical axis) -> world points [M,3] (and pixel idx)."""
    H, W = depth.shape
    dev = depth.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    valid = depth > 0 if mask is None else (mask & (depth > 0))
    z = depth[valid]
    u, v = xs[valid].float() + 0.5, ys[valid].float() + 0.5
    K = torch.as_tensor(K, dtype=torch.float32, device=dev)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = torch.stack([x, y, z], -1)
    w2c = torch.as_tensor(w2c, dtype=torch.float32, device=dev)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pw = (pc - t) @ R  # R^T (pc - t)
    return pw, valid


def project_points(pw, K, w2c):
    """world points [M,3] -> pixel (u, v) [M,2] and depth z [M]."""
    K = torch.as_tensor(K, dtype=torch.float32, device=pw.device)
    w2c = torch.as_tensor(w2c, dtype=torch.float32, device=pw.device)
    pc = pw @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / z.clamp_min(1e-8)[:, None] * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])
    return uv, z
