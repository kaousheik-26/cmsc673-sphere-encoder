import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from halfsphere.geometry import _bv, bdot, bnorm, exp_map, numel, tangent
from spherevideo.framenet import Block, time_embedding
from spherevideo.predict import log_map


def unit_dir(z0: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = tangent(z0, v)
    return _bv(1.0 / bnorm(u).clamp_min(1e-12), u) * u


def compose(z0: torch.Tensor, theta_deg: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    L = numel(z0)
    xi = _bv(torch.deg2rad(theta_deg.float()) * math.sqrt(L), z0) * unit_dir(z0, d)
    return exp_map(z0, xi)


def split_target(z0: torch.Tensor, zt: torch.Tensor):
    v = log_map(z0, zt)
    theta = torch.rad2deg(bnorm(v) / math.sqrt(numel(z0)))
    return theta, unit_dir(z0, v)


def dir_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return bdot(a, b) / (bnorm(a) * bnorm(b)).clamp_min(1e-12)


class RotNet(nn.Module):
    def __init__(self, token_dim=384, n_tokens=256, width=384, depth=4, heads=6, hint="none", angle_scale=10.0):
        super().__init__()
        self.hint = hint
        self.angle_scale = angle_scale
        self.inp = nn.Linear(token_dim * (2 if hint == "two" else 1), width)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, width))
        nn.init.normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.dir_out = nn.Linear(width, token_dim)
        self.ang_out = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, z0: torch.Tensor, t: torch.Tensor, z1: torch.Tensor = None):
        x = z0 if self.hint == "none" else torch.cat([z0, log_map(z0, z1)], dim=-1)
        h = self.inp(x) + self.pos
        c = self.temb(time_embedding(t, h.shape[-1]))
        for b in self.blocks:
            h = b(h, c)
        h = self.out_norm(h)
        theta = F.softplus(self.ang_out(torch.cat([h.mean(1), c], dim=-1)).squeeze(-1)) * self.angle_scale
        theta = theta * (t.float() > 0)
        return theta, unit_dir(z0, self.dir_out(h))
