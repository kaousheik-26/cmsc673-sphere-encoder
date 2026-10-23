"""
Operations on the SE2 latent sphere and on its 16x16 token grid.

Conventions
- z: [B, N, D] latent, N = h*w tokens (16x16), D channels (384).
- "global" sphere: radius sqrt(N*D) over all non-batch dims (the SE2 latent).
- "local" sphere: each token has radius sqrt(D) (grid_rms_norm).
"""

import math

import torch
import torch.nn.functional as F


def _dims(z, mode):
    return tuple(range(1, z.ndim)) if mode == "global" else (z.ndim - 1,)


def radius(z, mode="global"):
    return math.sqrt(math.prod(z.shape[d] for d in _dims(z, mode)))


def normalize(z, mode="global"):
    dims = _dims(z, mode)
    r = radius(z, mode)
    return z * (r / z.norm(dim=dims, keepdim=True).clamp_min(1e-12))


def cos_angle(a, b, mode="global"):
    dims = _dims(a, mode)
    return (a * b).sum(dims) / (a.norm(dim=dims) * b.norm(dim=dims)).clamp_min(1e-12)


def angle_deg(a, b, mode="global"):
    """geodesic angle between a and b in degrees ([B] global, [B, N] local)."""
    return torch.rad2deg(torch.acos(cos_angle(a, b, mode).clamp(-1.0, 1.0)))


def slerp(a, b, t, mode="global"):
    """spherical interpolation at fraction t (float or [B]) on the sphere of a."""
    dims = _dims(a, mode)
    an, bn = normalize(a, mode), normalize(b, mode)
    r = radius(a, mode)
    cos = ((an * bn).sum(dims, keepdim=True) / (r * r)).clamp(-1 + 1e-7, 1 - 1e-7)
    om = torch.acos(cos)
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    if t.ndim == 1:
        t = t.reshape(-1, *([1] * (a.ndim - 1)))
    so = torch.sin(om)
    return (torch.sin((1 - t) * om) / so) * an + (torch.sin(t * om) / so) * bn


def log_map(p, q, mode="global"):
    """tangent vector at p pointing to q, with length = arc length (radius * angle)."""
    dims = _dims(p, mode)
    r = radius(p, mode)
    pn, qn = normalize(p, mode), normalize(q, mode)
    cos = ((pn * qn).sum(dims, keepdim=True) / (r * r)).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    v = qn - cos * pn
    vn = v.norm(dim=dims, keepdim=True)
    return torch.where(vn > 1e-9, v * (theta * r / vn.clamp_min(1e-12)), torch.zeros_like(v))


def exp_map(p, v, mode="global"):
    """walk from p along tangent v (length = arc length) on the sphere of p."""
    dims = _dims(p, mode)
    r = radius(p, mode)
    pn = normalize(p, mode)
    v = v - ((pn * v).sum(dims, keepdim=True) / (r * r)) * pn  # keep it tangent
    vn = v.norm(dim=dims, keepdim=True)
    theta = vn / r
    out = torch.cos(theta) * pn + torch.sin(theta) * r * v / vn.clamp_min(1e-12)
    return torch.where(vn > 1e-9, out, pn)


def project_tangent(p, g, mode="global"):
    """component of g orthogonal to p (Riemannian gradient on the sphere)."""
    dims = _dims(p, mode)
    pn = normalize(p, mode)
    r = radius(p, mode)
    return g - ((pn * g).sum(dims, keepdim=True) / (r * r)) * pn


def token_norms(z):
    """per-token L2 norm, [B, N]."""
    return z.norm(dim=-1)


# ------------------------------------------------------------- token grids
def grid_hw(z):
    n = z.shape[1]
    h = int(round(math.sqrt(n)))
    assert h * h == n, f"{n} tokens is not a square grid"
    return h, h


def to_grid(z):
    """[B, N, D] -> [B, D, h, w]"""
    h, w = grid_hw(z)
    return z.transpose(1, 2).reshape(z.shape[0], z.shape[2], h, w)


def from_grid(g):
    """[B, D, h, w] -> [B, N, D]"""
    return g.flatten(2).transpose(1, 2)


def token_mask_from_pixels(mask, grid=16, mode="min"):
    """pixel mask [B,1,H,W] (1 = known) -> token mask [B, grid*grid] in {0,1}.
    mode='min': a token is known only if every pixel inside it is known (Gen3C)."""
    m = mask.float()
    if mode == "min":
        t = -F.adaptive_max_pool2d(-m, grid)
    elif mode == "mean":
        t = F.adaptive_avg_pool2d(m, grid)
    else:
        raise ValueError(mode)
    return t.flatten(1)


def warp_tokens_nearest(z, src_xy, valid, grid=16):
    """Gather-warp a token grid.

    z      : [B, N, D] source tokens
    src_xy : [B, grid, grid, 2] for each TARGET token, the source pixel coordinate
             in normalized [-1, 1] (x, y) from which it should be taken
    valid  : [B, grid, grid] bool, target tokens with a valid source
    returns warped tokens [B, N, D] (invalid tokens zeroed) and valid [B, N]
    """
    g = to_grid(z)
    out = F.grid_sample(g, src_xy, mode="nearest", align_corners=False)
    out = out * valid[:, None].float()
    return from_grid(out), valid.flatten(1)
