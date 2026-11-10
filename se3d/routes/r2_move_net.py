"""
Camera-conditioned network that moves an SE2 latent to a new camera (the learned latent move
of the latent search, r2_delta_model.py / r2_delta_scale.py), on SE2's 16x16 DINOv3 token grid.

    {z_tgt^j} = T_theta({z_src^i, cam_i}, {cam_j})

Sequence = [S source views x 256 tokens ; T target views x 256 tokens].
- source token = Linear(384 -> W)(DINO token) + Linear(6*p*p -> W)(Plucker patch)
- target token = learned embedding + Linear(6*p*p -> W)(Plucker patch)
- every attention block uses PRoPE (projective relative positional encoding,
  GTA-style): per head, half the channels carry the 4x4 camera projection
  P = [K_ndc R | K_ndc t ; 0 0 0 1] in 4-d blocks (q <- P_i^T q, k <- P_j^-1 k,
  v <- P_j^-1 v, o <- P_i o, so scores depend on P_i P_j^-1 only), the other
  half carries 2D RoPE of the patch position.
- output head (zero-init) adds a residual to a base: target base = the token of
  the FIRST source view at the same grid position, so training starts at
  "copy the source" (identity).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ndc_projection(K, w2c, width, height):
    """4x4 projective matrices in normalised image coordinates. K, w2c: [..., 3,3] / [..., 4,4]."""
    Kn = K.clone()
    Kn[..., 0, :] = Kn[..., 0, :] * (2.0 / width)
    Kn[..., 1, :] = Kn[..., 1, :] * (2.0 / height)
    Kn[..., 0, 2] -= 1.0
    Kn[..., 1, 2] -= 1.0
    P = torch.zeros(*K.shape[:-2], 4, 4, device=K.device, dtype=K.dtype)
    P[..., :3, :] = Kn @ w2c[..., :3, :]
    P[..., 3, 3] = 1.0
    return P


def plucker_patches(K, w2c, H, W, patch):
    """Plucker rays (d, o x d) sampled per pixel, patchified -> [B, (H/p)(W/p), 6 p p]."""
    B = K.shape[0]
    dev = K.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, -1, 3).expand(B, -1, -1)
    d_cam = pix @ torch.linalg.inv(K).transpose(1, 2)
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    d = F.normalize(d_cam @ R, dim=-1)  # R^T d_cam (row form)
    o = -(R.transpose(1, 2) @ t[..., None])[..., 0]
    m = torch.cross(o[:, None].expand_as(d), d, dim=-1)
    ray = torch.cat([d, m], -1).view(B, H, W, 6)
    g = H // patch
    ray = ray.view(B, g, patch, g, patch, 6).permute(0, 1, 3, 5, 2, 4).reshape(B, g * g, 6 * patch * patch)
    return ray


def rope_2d(dim, g, device):
    """cos/sin tables [g*g, dim] for 2D RoPE (dim/2 for x, dim/2 for y)."""
    half = dim // 2
    freqs = 1.0 / (100.0 ** (torch.arange(0, half, 2, device=device).float() / half))
    pos = torch.arange(g, device=device).float()
    fx = torch.outer(pos, freqs)  # [g, half/2]
    ang_x = fx[None, :, :].expand(g, g, -1)
    ang_y = fx[:, None, :].expand(g, g, -1)
    ang = torch.cat([ang_x, ang_y], -1).reshape(g * g, half)
    ang = torch.cat([ang, ang], -1)  # rotate-half convention
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


class PRoPEAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.h, self.dh = heads, width // heads
        assert self.dh % 8 == 0, "head dim must split into 4-d camera blocks and a RoPE half"
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.qn = nn.LayerNorm(self.dh)
        self.kn = nn.LayerNorm(self.dh)

    def forward(self, x, P, Pinv, cos, sin):
        """x [B,N,W]; P / Pinv [B,N,4,4] per token; cos/sin [N, dh/2]."""
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # [B,h,N,dh]
        q, k = self.qn(q), self.kn(k)
        c = self.dh // 2  # camera half / rope half
        nb = c // 4

        def cam(t, M):  # t [B,h,N,c] in 4-d blocks, M [B,N,4,4]
            tb = t.reshape(B, self.h, N, nb, 4)
            return torch.einsum("bnij,bhnkj->bhnki", M, tb).reshape(B, self.h, N, c)

        PT = P.transpose(-1, -2)
        qc, qr = q[..., :c], q[..., c:]
        kc, kr = k[..., :c], k[..., c:]
        vc, vr = v[..., :c], v[..., c:]
        q = torch.cat([cam(qc, PT), apply_rope(qr, cos, sin)], -1)
        k = torch.cat([cam(kc, Pinv), apply_rope(kr, cos, sin)], -1)
        v = torch.cat([cam(vc, Pinv), vr], -1)
        o = F.scaled_dot_product_attention(q, k, v)
        o = torch.cat([cam(o[..., :c], P), o[..., c:]], -1)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class Block(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = PRoPEAttention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.GELU(), nn.Linear(int(width * mlp_ratio), width))

    def forward(self, x, P, Pinv, cos, sin):
        x = x + self.attn(self.n1(x), P, Pinv, cos, sin)
        return x + self.mlp(self.n2(x))


class LatentMoveNet(nn.Module):
    def __init__(self, token_dim=384, width=512, depth=8, heads=8, grid=16, image_size=256, use_prope=True):
        super().__init__()
        self.grid, self.patch, self.img = grid, image_size // grid, image_size
        self.use_prope = use_prope
        self.src_in = nn.Linear(token_dim, width)
        self.ray_in = nn.Linear(6 * self.patch * self.patch, width)
        self.tgt_emb = nn.Parameter(torch.zeros(1, 1, width))
        self.view_emb = nn.Parameter(torch.zeros(2, width))  # source / target
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, token_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.tgt_emb, std=0.02)
        nn.init.normal_(self.view_emb, std=0.02)
        self.heads = heads
        self.width = width

    def forward(self, src_tokens, src_K, src_w2c, tgt_K, tgt_w2c):
        """src_tokens [B,S,256,D] (raw DINO tokens), cameras [B,S|T,3,3]/[B,S|T,4,4]
        -> predicted target tokens [B,T,256,D]."""
        B, S, N, D = src_tokens.shape
        T = tgt_K.shape[1]
        Kall = torch.cat([src_K, tgt_K], 1).flatten(0, 1)
        Wall = torch.cat([src_w2c, tgt_w2c], 1).flatten(0, 1)
        rays = plucker_patches(Kall, Wall, self.img, self.img, self.patch).view(B, S + T, N, -1)
        xs = self.src_in(src_tokens) + self.ray_in(rays[:, :S]) + self.view_emb[0]
        xt = self.tgt_emb.expand(B, T * N, -1).view(B, T, N, -1) + self.ray_in(rays[:, S:]) + self.view_emb[1]
        x = torch.cat([xs, xt], 1).flatten(1, 2)  # [B,(S+T)N,W]
        if self.use_prope:
            P = ndc_projection(Kall, Wall, self.img, self.img).view(B, S + T, 4, 4)
        else:
            P = torch.eye(4, device=x.device).expand(B, S + T, 4, 4)
        Pinv = torch.linalg.inv(P)
        P = P[:, :, None].expand(B, S + T, N, 4, 4).reshape(B, (S + T) * N, 4, 4)
        Pinv = Pinv[:, :, None].expand(B, S + T, N, 4, 4).reshape(B, (S + T) * N, 4, 4)
        cos, sin = rope_2d(self.width // self.heads // 2, self.grid, x.device)
        cos, sin = cos.repeat(S + T, 1), sin.repeat(S + T, 1)
        for blk in self.blocks:
            x = blk(x, P, Pinv, cos, sin)
        out = self.head(self.norm(x)).view(B, S + T, N, D)[:, S:]
        base = src_tokens[:, :1].expand(B, T, N, D)
        return base + out


class MixerBlock(nn.Module):
    def __init__(self, width, n_tokens, token_mixing=True, mlp_ratio=4.0):
        super().__init__()
        self.token_mixing = token_mixing
        if token_mixing:
            self.n1 = nn.LayerNorm(width)
            self.tok = nn.Sequential(nn.Linear(n_tokens, 2 * n_tokens), nn.GELU(), nn.Linear(2 * n_tokens, n_tokens))
        self.n2 = nn.LayerNorm(width)
        self.ch = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.GELU(), nn.Linear(int(width * mlp_ratio), width))

    def forward(self, x):  # [B,N,W]
        if self.token_mixing:
            x = x + self.tok(self.n1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(self.n2(x))


class LatentMoveMLP(nn.Module):
    """Small camera-conditioned MLP alternative to LatentMoveNet (same forward signature).
    Each target view is processed on its own: per token, input = source token + the source and
    target Plucker rays of that patch + a learned position embedding.
    kind='token': per-token MLP only (cannot move content between tokens).
    kind='mixer': MLP-Mixer blocks (token-mixing MLP across the 16x16 grid + channel MLP)."""

    def __init__(self, token_dim=384, width=384, depth=4, grid=16, image_size=256, kind="mixer"):
        super().__init__()
        self.grid, self.patch, self.img = grid, image_size // grid, image_size
        n = grid * grid
        self.src_in = nn.Linear(token_dim, width)
        self.ray_src = nn.Linear(6 * self.patch * self.patch, width)
        self.ray_tgt = nn.Linear(6 * self.patch * self.patch, width)
        self.pos = nn.Parameter(torch.zeros(1, n, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([MixerBlock(width, n, token_mixing=(kind == "mixer")) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, token_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, src_tokens, src_K, src_w2c, tgt_K, tgt_w2c):
        B, S, N, D = src_tokens.shape
        T = tgt_K.shape[1]
        rs = plucker_patches(src_K[:, 0], src_w2c[:, 0], self.img, self.img, self.patch).view(B, 1, N, -1)
        rt = plucker_patches(tgt_K.flatten(0, 1), tgt_w2c.flatten(0, 1), self.img, self.img, self.patch).view(B, T, N, -1)
        x = self.src_in(src_tokens[:, :1]) + self.ray_src(rs) + self.ray_tgt(rt) + self.pos[:, None]  # [B,T,N,W]
        x = x.flatten(0, 1)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(self.norm(x)).view(B, T, N, D)
        return src_tokens[:, :1].expand(B, T, N, D) + out


def build_move_net(a):
    """network from a training-args dict (config.json "args"); defaults = the transformer."""
    arch = a.get("arch", "transformer")
    if arch == "transformer":
        return LatentMoveNet(width=a.get("width", 512), depth=a.get("depth", 8), heads=a.get("heads", 8),
                             use_prope=not a.get("no_prope", False))
    return LatentMoveMLP(width=a.get("width", 384), depth=a.get("depth", 4), kind=arch)
