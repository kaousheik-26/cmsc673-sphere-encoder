import math

import torch
import torch.nn as nn

from halfsphere.geometry import bdot, bnorm, exp_map, tangent


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 100.0 * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.mod = nn.Linear(dim, 4 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h, c):
        s1, b1, s2, b2 = self.mod(c)[:, None].chunk(4, dim=-1)
        x = self.n1(h) * (1 + s1) + b1
        h = h + self.attn(x, x, x, need_weights=False)[0]
        x = self.n2(h) * (1 + s2) + b2
        return h + self.mlp(x)


class FrameNet(nn.Module):
    def __init__(self, token_dim=384, n_tokens=256, width=384, depth=4, heads=6, arch="transformer"):
        super().__init__()
        self.arch = arch
        self.inp = nn.Linear(token_dim, width)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, width))
        self.temb = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        if arch == "transformer":
            self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out = nn.Linear(width, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(z0) + self.pos
        c = self.temb(time_embedding(t, h.shape[-1]))
        for b in self.blocks:
            h = b(h, c) if self.arch == "transformer" else h + b(h + c[:, None])
        xi = self.out(self.out_norm(h)) * (t.float().view(-1, 1, 1) > 0)
        return exp_map(z0, tangent(z0, xi))


def sphere_loss(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    return 1.0 - bdot(z_pred, z_true) / (bnorm(z_pred) * bnorm(z_true))
