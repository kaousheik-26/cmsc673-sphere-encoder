from typing import Callable, Optional
from torch import Tensor

import logging
import math
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cli_utils import get_device_type

logger = logging.getLogger(__name__)

"""
attn import
"""

_SDPA_MODE = "manual"

if hasattr(F, "scaled_dot_product_attention"):
    _SDPA_MODE = "sdpa"

try:
    from flash_attn import flash_attn_func

    _SDPA_MODE = "flash_attn"
except ImportError:
    pass


"""
basic functions
"""


def vector_rms_norm(z, zero_mean=False, eps=1e-5):
    _dtype = z.dtype
    z = z.float()
    assert z.ndim in [3, 4]  # [B, C, H, W] or [B, N, D]
    dim = tuple(range(1, z.ndim))  # w/o batch dimension
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    m = z.square().mean(dim=dim, keepdim=True)
    m = torch.rsqrt(m + eps)
    return (z * m).to(_dtype)


def grid_rms_norm(z, zero_mean=False, eps=1e-5):
    _dtype = z.dtype
    z = z.float()
    assert z.ndim in [3]  # [B, N, D]
    dim = 2  # normalize over feature dimension
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    m = z.square().mean(dim=dim, keepdim=True)
    m = torch.rsqrt(m + eps)
    return (z * m).to(_dtype)


@torch.no_grad()
def stratified_unit_radii_ddp(
    size,
    rank,
    world_size,
    step_seed=0,
    shuffle=True,
    including_zero=True,
    device=get_device_type(),
    dtype=torch.float32,
):
    """
    stratified values in [0, 1], balanced across all ddp ranks: every rank
    builds the same global list from step_seed and takes its own slice

    size            : local shape (batch_size_per_rank, ...)
    rank            : current ddp rank
    world_size      : total number of ranks
    step_seed       : a seed that is the same across all ranks for the current step
    shuffle         : shuffle the global list before slicing, to avoid rank bias
    including_zero  : pin one sample of the global batch at 0
    out             : [size[0], 1, ..., 1]
    """

    # 0. get info
    local_N = size[0]
    global_N = local_N * world_size

    # 1. use a synchronized generator so all ranks create the same global list
    g = torch.Generator(device=device)
    g.manual_seed(step_seed)

    # 2. calculate M for the global pool
    M = global_N - 2 if including_zero else global_N - 1

    # 3. create global stratified points
    i = torch.arange(M, device=device, dtype=torch.float32)
    v_noise = torch.rand(M, device=device, generator=g)
    v_noise = torch.clamp(v_noise, min=1e-3)

    v_global = (i + v_noise) / M

    # 4. handle boundary values (0.0 and 1.0)
    w = [1.0, 0.0] if including_zero else [1.0]
    v_global = torch.cat([v_global, torch.tensor(w, device=device)])

    # 5. global shuffle (must use the synchronized generator)
    if shuffle:
        indices = torch.randperm(global_N, generator=g, device=device)
        v_global = v_global[indices]

    # 6. slice the global tensor to get the local chunk for THIS rank
    start_idx = rank * local_N
    end_idx = start_idx + local_N
    v_local = v_global[start_idx:end_idx]

    # reshape to match the requested size (e.g., [batch, 1, 1, 1] for images)
    return v_local.reshape(-1, *[1] * len(size[1:])).to(dtype=dtype)


"""
functions and modules for models
"""


def modulate(x, shift=None, scale=None):
    if shift is None and scale is None:
        return x
    if x.ndim == shift.ndim:
        return x * (1 + scale) + shift
    elif x.ndim == shift.ndim + 1:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        raise ValueError(
            f"shift shape {shift.shape} and x shape {x.shape} are not compatible"
        )


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size [M,]
    out: [M, D]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [D/2,]

    pos = pos.reshape(-1)  # [M,]
    out = np.einsum("m,d->md", pos, omega)  # [M, D/2], outer product

    embed_sin = np.sin(out)  # [M, D/2]
    embed_cos = np.cos(out)  # [M, D/2]

    embed = np.concatenate([embed_sin, embed_cos], axis=1)  # [M, D]
    return embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    embed_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    embed_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]
    return np.concatenate([embed_h, embed_w], axis=1)  # [H*W, D]


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    2d sin-cos position embedding over a square grid

    embed_dim    : output dimension for each position
    grid_size    : height and width of the grid
    cls_token    : prepend zero rows for the extra tokens
    extra_tokens : number of zero rows to prepend, used with cls_token
    out          : [grid_size * grid_size, embed_dim], or
                   [extra_tokens + grid_size * grid_size, embed_dim] with cls_token
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_rope_tensor(dim, seq_h, seq_w, pad_size=0, theta=10000.0):
    """
    2D axial rope tensor: [pad_size + (seq_h * seq_w), dim * 2] (cos, sin)
    """
    assert (
        dim % 4 == 0
    ), "dim must be divisible by 4 for 2D RoPE (div 2 for axial, div 2 for cos/sin)"
    half_dim = dim // 2

    # create freqs for half the dim
    # 1 / (theta ^ ( 2i / (D/2) ))
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))

    # repeat freqs to interleave rotations
    # [f1, f2] -> [f1, f1, f2, f2] for rotate_half
    freqs = einops.repeat(freqs, "n -> (n r)", r=2)  # [D/2]

    # create 2D grid from seq_len
    # t_h: [0, 1, ... H-1]
    # t_w: [0, 1, ... W-1]
    t_h = torch.arange(seq_h).float()
    t_w = torch.arange(seq_w).float()

    # freqs x coords
    freqs_h = torch.outer(t_h, freqs)  # [H, D/2]
    freqs_w = torch.outer(t_w, freqs)  # [W, D/2]

    # broadcast and concat for 2D axial
    freqs_h = einops.repeat(freqs_h, "h d -> h w d", w=seq_w)
    freqs_w = einops.repeat(freqs_w, "w d -> h w d", h=seq_h)

    freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1)  # [H, W, D]
    freqs_2d = freqs_2d.view(-1, dim)  # [H*W, D]

    cos_img = freqs_2d.cos()
    sin_img = freqs_2d.sin()

    if pad_size > 0:
        ones = torch.ones(pad_size, dim, device=cos_img.device, dtype=cos_img.dtype)
        zeros = torch.zeros(pad_size, dim, device=sin_img.device, dtype=sin_img.dtype)

        cos_img = torch.cat([ones, cos_img], dim=0)
        sin_img = torch.cat([zeros, sin_img], dim=0)

    return torch.cat([cos_img, sin_img], dim=-1)


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_emb(x, freqs_cis):
    freqs_cos, freqs_sin = freqs_cis.unsqueeze(1).chunk(2, dim=-1)
    freqs_cos = freqs_cos.to(x.dtype)
    freqs_sin = freqs_sin.to(x.dtype)
    return x * freqs_cos + rotate_half(x) * freqs_sin


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor, Optional[Tensor]], Tensor],
    sample_drop_ratio: float = 0.0,
):
    """
    DINOv2-style stochastic depth:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/block.py#L173
    """

    # extract subset using permutation
    B = x.shape[0]
    sample_subset_size = max(int(B * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(B, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # apply residual_func to get residual
    residual = residual_func(x_subset, brange)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = B / sample_subset_size

    # add the residual back, scaled to compensate for the dropped samples
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        _dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x.to(_dtype)


class SwiGLUFFN(nn.Module):

    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class ScaledTanh(nn.Module):
    """
    scaled tanh for the final pixel head: s * tanh(x), where s is chosen so
    the output reaches the pixel rails (+/-1) at x = +/-rail_input. only
    applied in training; at inference the input is returned unchanged

    rail_input : input value at which the output reaches +/-1
    out        : same shape as x
    """

    def __init__(self, rail_input: float = 1.5):
        super().__init__()
        self.rail_input = rail_input
        self.scale = 1.0 / math.tanh(rail_input)

    def forward(self, x):
        if not self.training:
            return x
        return self.scale * torch.tanh(x)


"""
attention 
"""


def _sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sdpa_mode: Optional[str] = None
) -> torch.Tensor:
    """
    scaled dot-product attention

    q, k, v   : [B, H, N, D]
    sdpa_mode : flash_attn | sdpa | sdpa_fp32 | manual (default when None)
    out       : [B, H, N, D]
    """
    _SDPA_MODE = sdpa_mode if sdpa_mode is not None else "manual"

    if _SDPA_MODE == "flash_attn":
        # flash_attn expects [B, N, H, D] with bf16
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = flash_attn_func(q, k, v, causal=False)
        return out.transpose(1, 2)  #  [B, H, N, D]

    elif _SDPA_MODE == "sdpa":
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ],
            set_priority=True,
        ):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    elif _SDPA_MODE == "sdpa_fp32":
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ],
            set_priority=True,
        ), torch.amp.autocast(device_type=q.device.type, enabled=False):
            out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=False
            )
        return out.to(v.dtype)  # [B, H, N, D]

    else:  # manual mode (default): fp32 Q@K and fp32 P@V
        scale = 1.0 / math.sqrt(q.size(-1))
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = torch.matmul(attn, v.float())
        return out.to(v.dtype)  # [B, H, N, D]


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        sdpa_mode: Optional[str] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads != 0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # per-head RMSNorm on q/k, applied before rope
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()

        if sdpa_mode is not None:
            self.sdpa_mode = sdpa_mode
            assert self.sdpa_mode in [
                "manual",
                "sdpa",
                "sdpa_fp32",
                "flash_attn",
            ], f"invalid sdpa_mode: {self.sdpa_mode}"
        else:
            self.sdpa_mode = _SDPA_MODE
        logger.info(f"attention mode: {self.sdpa_mode}")

    def forward(self, x: torch.Tensor, rope: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x)

        # [B, N, 3, H, D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # [3, B, H, N, D]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rotary_emb(q, rope).to(v.dtype)
        k = apply_rotary_emb(k, rope).to(v.dtype)

        # [B, H, N, D]
        x = _sdpa(q, k, v, sdpa_mode=self.sdpa_mode)

        # [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


"""
modules
"""


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000):
        """
        sinusoidal timestep embeddings

        t          : [N] indices, one per batch element, may be fractional
        dim        : dimension of the output
        max_period : controls the minimum frequency of the embeddings
        out        : [N, dim]
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings.unsqueeze(1)


class AngleEmbedder(nn.Module):
    """
    noise-angle ("timestep") conditioning for the decoder:

        alpha (rad) -> u = alpha / alpha_max -> sinusoidal bank -> mlp

    the output projection is zero-initialized, so the module contributes
    nothing at init: adding it to an existing (modulated) checkpoint preserves
    that model's exact function

    hidden_size   : dimension of the output embedding
    max_angle_deg : alpha_max in degrees, the angle mapped to u = 1
    freq_dim      : size of the sinusoidal bank (cos / sin halves), must be even
    max_period    : controls the minimum frequency of the bank
    time_scale    : multiplier on u before the sinusoids
    """

    def __init__(
        self,
        hidden_size: int,
        max_angle_deg: float = 90.0,
        freq_dim: int = 256,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        assert freq_dim % 2 == 0, "freq_dim must be even (cos/sin halves)"
        assert max_angle_deg > 0, f"max_angle_deg must be positive"

        self.hidden_size = hidden_size
        self.max_angle_deg = float(max_angle_deg)
        self.max_angle_rad = math.radians(float(max_angle_deg))
        self.freq_dim = freq_dim
        self.time_scale = float(time_scale)

        half = freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        # non-persistent: derived constant, kept out of the state dict
        self.register_buffer("freqs", freqs, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def initialize_weights(self, zero_out: bool = True):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.constant_(self.mlp[0].bias, 0)
        if zero_out:
            nn.init.constant_(self.mlp[2].weight, 0)
        else:
            nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.constant_(self.mlp[2].bias, 0)

    def forward(self, alpha):
        """
        alpha : angles in radians, [B] or any [B, ...] shape carrying one angle
                per sample (e.g. [B, 1, 1])
        out   : [B, 1, hidden_size], matching LabelEmbedder's layout so the two
                conditioning signals simply add
        """
        a = alpha.float().reshape(alpha.shape[0], -1)[:, 0]  # [B]
        u = (a / self.max_angle_rad).clamp(0.0, 1.0)
        t = u[:, None] * self.freqs[None, :] * self.time_scale  # [B, half]
        v = torch.cat([torch.cos(t), torch.sin(t)], dim=-1)  # [B, freq_dim]
        v = v.to(self.mlp[0].weight.dtype)
        return self.mlp(v).unsqueeze(1)  # [B, 1, D]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(hidden_size={self.hidden_size}, "
            f"max_angle_deg={self.max_angle_deg}, "
            f"freq_dim={self.freq_dim})"
        )


class ModulatedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_modulation: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_modulation:
            self.norm = RMSNorm(in_features, eps=1e-5)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(in_features, 2 * in_features, bias=bias)
            )
        self.use_modulation = use_modulation

    def forward(self, x, cond=None):
        if self.use_modulation:
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
            x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_modulation: bool = False,
        qk_norm: bool = False,
        drop_residual_path_prob: float = 0.0,
        sdpa_mode: Optional[str] = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-5)
        self.norm2 = RMSNorm(hidden_size, eps=1e-5)

        self.attn = Attention(
            hidden_size, num_heads, qk_norm=qk_norm, sdpa_mode=sdpa_mode
        )
        self.mlp = SwiGLUFFN(hidden_size, expansion_factor=2 / 3 * mlp_ratio)

        self.sample_drop_ratio = drop_residual_path_prob

        self.use_modulation = use_modulation
        self.adaLN_modulation = None

        if self.use_modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, cond=None, rope=None, mod=None):
        if mod is not None:
            # pre-computed modulation, [B, 1, 6 * hidden_size]. Lets a caller
            # own the cond -> modulation map instead of this block owning it
            # (adaLN-single: one shared map for the whole stack, plus a small
            # per-block offset), so the block keeps no Linear of its own.
            out = mod.chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa = out[:3]
            shift_mlp, scale_mlp, gate_mlp = out[3:]
        elif cond is not None:
            out = self.adaLN_modulation(cond).chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa = out[:3]
            shift_mlp, scale_mlp, gate_mlp = out[3:]
        else:
            shift_msa, scale_msa, gate_msa = None, None, 1.0
            shift_mlp, scale_mlp, gate_mlp = None, None, 1.0

        def _index(c, brange):
            # index per-sample conditioning by the subset; pass through
            # scalars (gate == 1.0) and None (no modulation) unchanged
            if brange is None or not torch.is_tensor(c):
                return c
            return c[brange]

        def attn_residual_func(x_sub, brange=None):
            # rope carries no batch dim, so it is never subset-indexed
            return _index(gate_msa, brange) * self.attn(
                modulate(
                    self.norm1(x_sub),
                    _index(shift_msa, brange),
                    _index(scale_msa, brange),
                ),
                rope=rope,
            )

        def mlp_residual_func(x_sub, brange=None):
            return _index(gate_mlp, brange) * self.mlp(
                modulate(
                    self.norm2(x_sub),
                    _index(shift_mlp, brange),
                    _index(scale_mlp, brange),
                ),
            )

        if self.training and self.sample_drop_ratio > 0.0:
            x = drop_add_residual_stochastic_depth(
                x, attn_residual_func, self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, mlp_residual_func, self.sample_drop_ratio
            )
        else:
            x = x + attn_residual_func(x)
            x = x + mlp_residual_func(x)
        return x
