"""Text-conditioned flow-matching DiT over the frozen SE2 latent (256 tokens x 384 channels).

x_t = (1 - t) x0 + t eps,  model predicts v = eps - x0,  t ~ shifted logit-normal.
Conditioning: adaLN-Zero from (timestep embedding + MLP(CLAP text embedding)); a learned null text vector
replaces the caption for classifier-free guidance.
Sampling: Euler from t = 1 to 0 on the same shifted schedule, CFG on v.
x0 is the spherified SE2 latent, normalised per channel with dataset statistics (mu, sd); to decode:
z = SE2.f(x * sd + mu) -> frozen SE2 decoder at angle 0.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_2d(n, dim):
    g = int(math.isqrt(n)); assert g * g == n
    y, x = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing="ij")
    om = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    out = [torch.sin(x.flatten()[:, None] * om), torch.cos(x.flatten()[:, None] * om),
           torch.sin(y.flatten()[:, None] * om), torch.cos(y.flatten()[:, None] * om)]
    return torch.cat(out, 1)[None]  # [1, n, dim]


def t_embed(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000 * f[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class Block(nn.Module):
    def __init__(self, h, heads, mlp=4.0):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(h, 3 * h)
        self.qn = nn.RMSNorm(h // heads, eps=1e-6); self.kn = nn.RMSNorm(h // heads, eps=1e-6)
        self.proj = nn.Linear(h, h)
        self.n2 = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(h, int(h * mlp)), nn.GELU(approximate="tanh"), nn.Linear(int(h * mlp), h))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(h, 6 * h))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        B, N, H = x.shape
        y = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(y).view(B, N, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(self.qn(q), self.kn(k), v).transpose(1, 2).reshape(B, N, H)
        x = x + g1 * self.proj(y)
        x = x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)
        return x


class LatentDiT(nn.Module):
    def __init__(self, tokens=256, chans=384, hidden=768, depth=16, heads=12, text_dim=512):
        super().__init__()
        self.tokens, self.chans = tokens, chans
        self.inp = nn.Linear(chans, hidden)
        self.register_buffer("pos", sincos_2d(tokens, hidden), persistent=False)
        self.temb = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.text = nn.Sequential(nn.Linear(text_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.null = nn.Parameter(torch.zeros(hidden))
        self.blocks = nn.ModuleList([Block(hidden, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.out = nn.Linear(hidden, chans)
        for m in (self.ada_f[1], self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.register_buffer("mu", torch.zeros(chans)); self.register_buffer("sd", torch.ones(chans))

    def cond(self, temb, drop=None):
        c = self.text(temb.float())
        if drop is not None:
            c = torch.where(drop[:, None], self.null.expand_as(c), c)
        return c

    def forward(self, x, t, c=None, temb=None, drop=None):
        if c is None:
            c = self.cond(temb, drop)
        c = c + self.temb(t_embed(t))
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h, c)
        s, b_ = self.ada_f(c)[:, None].chunk(2, -1)
        return self.out(self.nf(h) * (1 + s) + b_)

    # ---- normalisation of the SE2 latent ----
    def norm(self, z):
        return (z - self.mu) / self.sd

    def denorm(self, x):
        return x * self.sd + self.mu


def shift_t(t, s):
    return s * t / (1 + (s - 1) * t)


def fm_loss(model, x0, temb, drop, shift=3.0):
    B = x0.shape[0]
    t = shift_t(torch.sigmoid(torch.randn(B, device=x0.device)), shift)
    e = torch.randn_like(x0)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * e
    v = model(xt, t, temb=temb, drop=drop)
    return F.mse_loss(v.float(), (e - x0).float())


@torch.no_grad()
def sample(model, temb, steps=50, cfg=3.0, shift=3.0, noise=None):
    """temb [B, 512] CLAP text embeddings -> normalised latent [B, 256, 384]."""
    B, dev = temb.shape[0], temb.device
    x = noise if noise is not None else torch.randn(B, model.tokens, model.chans, device=dev)
    cc = model.cond(temb)
    cu = model.null.expand_as(cc)
    ts = shift_t(torch.linspace(1, 0, steps + 1, device=dev), shift)
    for i in range(steps):
        t = ts[i].expand(B)
        if cfg != 1.0:
            v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cc, cu]))
            vc, vu = v.chunk(2)
            v = vu + cfg * (vc - vu)
        else:
            v = model(x, t, cc)
        x = x + (ts[i + 1] - ts[i]) * v.float()
    return x
