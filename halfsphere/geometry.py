import math

import torch


def _bv(s: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return s.view(-1, *([1] * (like.ndim - 1)))


def numel(v: torch.Tensor) -> int:
    return v[0].numel()


def bdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.flatten(1) * b.flatten(1)).sum(1)


def bnorm(a: torch.Tensor) -> torch.Tensor:
    return a.flatten(1).norm(dim=1)


def f(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    z = z.float()
    m = z.square().mean(dim=tuple(range(1, z.ndim)), keepdim=True)
    return z * torch.rsqrt(m + eps)


def f_local(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    z = z.float()
    return z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps)


def tangent(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return g - _bv(bdot(v, g) / bdot(v, v), v) * v


def exp_map(v: torch.Tensor, xi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    L = numel(v)
    n = bnorm(xi).clamp_min(eps)
    th = n / math.sqrt(L)
    out = _bv(torch.cos(th), v) * v + _bv(math.sqrt(L) * torch.sin(th) / n, v) * xi
    return f(out)


def retract(v: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    return f(v + xi)


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cos = bdot(a, b) / (bnorm(a) * bnorm(b))
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def rotate(v: torch.Tensor, e_hat: torch.Tensor, alpha_deg) -> torch.Tensor:
    a = torch.deg2rad(torch.as_tensor(alpha_deg, dtype=torch.float32, device=v.device))
    if a.ndim == 0:
        return torch.cos(a) * v + torch.sin(a) * e_hat
    return _bv(torch.cos(a), v) * v + _bv(torch.sin(a), v) * e_hat


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    cos = (bdot(a, b) / (bnorm(a) * bnorm(b))).clamp(-1 + 1e-7, 1 - 1e-7)
    om = torch.acos(cos)
    so = torch.sin(om)
    return _bv(torch.sin((1 - t) * om) / so, a) * a + _bv(torch.sin(t * om) / so, a) * b


def effective_angle_deg(energy_frac_A: float, alpha_A_deg: float, alpha_B_deg: float) -> float:
    c = energy_frac_A * math.cos(math.radians(alpha_A_deg)) + (1 - energy_frac_A) * math.cos(
        math.radians(alpha_B_deg)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def token_manifold_gap(v: torch.Tensor) -> dict:
    v = v.float()
    D = v.shape[-1]
    mu = v.mean(-1)
    nr = v.norm(dim=-1) / math.sqrt(D)
    return {
        "tok_mean_abs": mu.abs().mean(-1),
        "tok_norm_dev": (nr - 1).abs().mean(-1),
        "dist_frac": bnorm(v - project_tokens(v)) / bnorm(v),
    }


def project_tokens(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    z = z.float()
    z = z - z.mean(-1, keepdim=True)
    return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
