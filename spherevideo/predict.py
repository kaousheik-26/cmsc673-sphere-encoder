import math
from dataclasses import dataclass

import torch

from halfsphere.geometry import _bv, angle_deg, bdot, bnorm, exp_map, f, numel, tangent


def log_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = numel(a)
    cos = (bdot(a, b) / (bnorm(a) * bnorm(b))).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(cos)
    u = tangent(a, b)
    return _bv(th * math.sqrt(L) / bnorm(u).clamp_min(1e-12), a) * u


def velocities(z: torch.Tensor) -> torch.Tensor:
    return log_map(z[:-1], z[1:])


def turning_deg(z: torch.Tensor) -> torch.Tensor:
    return angle_deg(log_map(z[1:-1], z[2:]), -log_map(z[1:-1], z[:-2]))


def extrapolate(z_prev: torch.Tensor, z_cur: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    return exp_map(z_cur, -gain * log_map(z_cur, z_prev))


@dataclass
class VelocityMap:
    gain: float
    basis: torch.Tensor = None
    W: torch.Tensor = None

    @classmethod
    def fit(cls, Z: torch.Tensor, rank: int = 0, ridge: float = 1e-2):
        prev, nxt = [], []
        for z in Z:
            v = velocities(z)
            prev.append(v[:-1].flatten(1))
            nxt.append(v[1:].flatten(1))
        P = torch.cat(prev).double()
        N = torch.cat(nxt).double()
        gain = float((P * N).sum() / (P * P).sum())
        if rank <= 0:
            return cls(gain)
        q = min(rank, P.shape[0] - 1)
        _, _, V = torch.pca_lowrank(torch.cat([P, N]), q=q, center=False, niter=4)
        B = V[:, :q]
        a, b = P @ B, N @ B
        A =a.T @ a + ridge * torch.trace(a.T @ a) / q * torch.eye(q, dtype=a.dtype, device=a.device)
        W = torch.linalg.solve(A, a.T @ (b - gain * a))
        return cls(gain, B.float(), W.float())

    def step(self, z_prev: torch.Tensor, z_cur: torch.Tensor) -> torch.Tensor:
        v = -log_map(z_cur, z_prev)
        xi = self.gain * v
        if self.basis is not None:
            a = v.flatten(1) @ self.basis
            xi = xi + ((a @ self.W) @ self.basis.T).view_as(v)
        return exp_map(z_cur, tangent(z_cur, xi))


def snap(bb, z: torch.Tensor, alpha_deg: float = 0.0, e_hat=None) -> torch.Tensor:
    x = bb.decode(z, 0.0)
    v = bb.encode(x.clamp(-1, 1))
    if alpha_deg > 0 and e_hat is not None:
        v = f(bb.rotate(v, e_hat, alpha_deg))
    return v


def search(bb, x_tgt, z_init, lpips_fn, steps=150, lr=1e-2, mu=1.0, z_anchor=None):
    z_anchor = z_init if z_anchor is None else z_anchor
    u = z_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    best = z_init.detach().clone()
    best_l = torch.full((x_tgt.shape[0],), float("inf"), device=x_tgt.device)
    for s in range(steps + 1):
        z = f(u)
        x = bb.decode(z, 0.0)
        lp = lpips_fn.fn(x, x_tgt).view(-1)
        l1 = 0.5 * (x - x_tgt).abs().flatten(1).mean(1)
        cos = bdot(z, z_anchor) / (bnorm(z) * bnorm(z_anchor))
        per = lp + l1 + mu * 2 * (1 - cos)
        with torch.no_grad():
            better = lp < best_l
            best_l = torch.where(better, lp, best_l)
            best[better] = z.detach()[better]
        if s == steps:
            break
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
    return best, best_l
