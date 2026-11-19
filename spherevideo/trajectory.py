import math

import torch

from halfsphere.geometry import angle_deg
from spherevideo.predict import log_map


def gram_pca(X: torch.Tensor, k: int = 3):
    X = X.flatten(1).double()
    Xc = X - X.mean(0, keepdim=True)
    G = Xc @ Xc.T
    w, V = torch.linalg.eigh(G)
    w, V = w.flip(0).clamp_min(0), V.flip(1)
    coords = V[:, :k] * w[:k].sqrt()
    return coords.float(), (w / w.sum().clamp_min(1e-12)).float()


def angle_from_start(z: torch.Tensor) -> torch.Tensor:
    return angle_deg(z[:1].expand_as(z), z)


def growth_exponent(a: torch.Tensor, t_min: int = 1) -> float:
    t = torch.arange(len(a), dtype=torch.float64, device=a.device)[t_min:]
    y = a.double()[t_min:].clamp_min(1e-6)
    lx, ly = t.log(), y.log()
    lx, ly = lx - lx.mean(), ly - ly.mean()
    return float((lx * ly).sum() / (lx * lx).sum())


def self_similarity(z: torch.Tensor) -> torch.Tensor:
    u = z.flatten(1).double()
    u = u / u.norm(dim=1, keepdim=True)
    return torch.rad2deg(torch.acos((u @ u.T).clamp(-1, 1))).float()


def token_motion(z: torch.Tensor, grid: int) -> torch.Tensor:
    v = log_map(z[:-1], z[1:])
    return v.norm(dim=-1).mean(0).view(grid, grid)


def straightness(z: torch.Tensor) -> float:
    steps = angle_deg(z[:-1], z[1:]).sum()
    return float(angle_from_start(z)[-1] / steps.clamp_min(1e-9))


def shared_directions(Zs, k: int = 10):
    V = torch.cat([log_map(z[:-1], z[1:]).flatten(1) for z in Zs]).double()
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = V @ V.T
    w = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    frac = w / w.sum()
    within = []
    for z in Zs:
        u = log_map(z[:-1], z[1:]).flatten(1).double()
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        wz = torch.linalg.eigvalsh(u @ u.T).flip(0).clamp_min(0)
        within.append(float((wz / wz.sum())[:k].sum()))
    mean_dir = V.mean(0)
    return {"pooled_top_k": float(frac[:k].sum()), "within_clip_top_k": sum(within) / len(within), "k": k,
            "n_steps": V.shape[0], "mean_unit_step_norm": float(mean_dir.norm()), "random_mean_norm": 1.0 / math.sqrt(V.shape[0])}


def sphere_embed(z: torch.Tensor):
    th = torch.deg2rad(angle_from_start(z)).double()
    V = log_map(z[:1].expand_as(z), z).flatten(1).double()
    w, U = torch.linalg.eigh(V @ V.T)
    w, U = w.flip(0).clamp_min(0), U.flip(1)
    c = U[:, :2] * w[:2].sqrt()
    d = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d[0] = 0
    xyz = torch.stack([th.sin() * d[:, 0], th.sin() * d[:, 1], th.cos()], dim=1)
    return xyz.float(), float(w[:2].sum() / w.sum().clamp_min(1e-12))


def analyze(z: torch.Tensor, grid: int = 16) -> dict:
    coords, ev = gram_pca(z, 3)
    xyz, captured = sphere_embed(z)
    a = angle_from_start(z)
    steps = angle_deg(z[:-1], z[1:])
    return {
        "pca": coords.tolist(), "pca_var": ev[:5].tolist(),
        "sphere3": xyz.tolist(), "sphere3_captured": captured,
        "angle_from_start": a.tolist(), "growth_H": growth_exponent(a),
        "step_deg": steps.tolist(), "straightness": straightness(z),
        "similarity": self_similarity(z).tolist(),
        "token_motion": token_motion(z, grid).tolist(),
    }
