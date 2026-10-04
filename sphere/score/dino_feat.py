from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


"""
window crop
"""


def _linspace_indices(limit: int, count: int) -> List[int]:
    if count <= 1:
        return [0]
    return sorted({int(round(i * (limit / (count - 1)))) for i in range(count)})


def _gen_positions_1d(length: int, crop: int, slots: int) -> List[int]:
    limit = max(length - crop, 0)
    pos = _linspace_indices(limit, max(slots, 1))
    pos = [max(0, min(p, limit)) for p in pos]
    if slots > 1:
        pos[0] = 0
        pos[-1] = limit
    return pos


class RandomWindowCrop:
    """Random crop with a fixed catalog of windows (XLA-friendly variant)."""

    def __init__(
        self,
        input_size: int | Tuple[int, int],
        crop: int,
        num_windows: int,
        per_sample: bool = False,
    ):
        if isinstance(input_size, int):
            self.H = self.W = int(input_size)
        else:
            self.H, self.W = map(int, input_size)
        self.crop = int(crop)
        self.per_sample = bool(per_sample)

        if self.crop <= 0:
            raise ValueError("crop must be > 0")
        if self.crop > self.H or self.crop > self.W:
            raise ValueError(f"crop={self.crop} exceeds input {(self.H, self.W)}")
        if num_windows <= 0:
            raise ValueError("num_windows must be > 0")

        rows_min = math.ceil(self.H / self.crop)
        cols_min = math.ceil(self.W / self.crop)
        n_min = rows_min * cols_min
        if num_windows < n_min:
            raise ValueError(
                f"num_windows={num_windows} too small to cover {(self.H, self.W)} with crop {self.crop}"
            )

        t_rows = _gen_positions_1d(self.H, self.crop, rows_min)
        l_cols = _gen_positions_1d(self.W, self.crop, cols_min)
        base_offsets = [(t, l) for t in t_rows for l in l_cols]

        offsets = list(base_offsets)
        if num_windows > len(offsets):
            rows_t = max(
                rows_min, int(math.floor(math.sqrt(num_windows * self.H / self.W)))
            )
            cols_t = max(cols_min, int(math.ceil(num_windows / rows_t)))
            while rows_t * cols_t < num_windows:
                cols_t += 1

            t_more = _gen_positions_1d(self.H, self.crop, rows_t)
            l_more = _gen_positions_1d(self.W, self.crop, cols_t)
            dense = [(t, l) for t in t_more for l in l_more]

            seen = set(offsets)
            for off in dense:
                if len(offsets) >= num_windows:
                    break
                if off not in seen:
                    offsets.append(off)
                    seen.add(off)

            idx = 0
            while len(offsets) < num_windows and idx < len(dense):
                offsets.append(dense[idx])
                idx += 1

        self.offsets: List[Tuple[int, int]] = offsets[:num_windows]
        self.num_windows = len(self.offsets)

    def __repr__(self) -> str:
        return (
            f"RandomWindowCrop(input={(self.H, self.W)}, crop={self.crop}, "
            f"windows={self.num_windows}, per_sample={self.per_sample})"
        )

    def _rand_idx(self) -> int:
        return torch.randint(0, self.num_windows, (1,)).item()

    def __call__(self, tensor: Tensor) -> Tensor:
        H, W = tensor.shape[-2], tensor.shape[-1]
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Expected input {(self.H, self.W)}, got {(H, W)}")

        crop = self.crop
        if self.per_sample and tensor.dim() >= 4 and tensor.shape[0] > 1:
            outputs = []
            for i in range(tensor.shape[0]):
                top, left = self.offsets[self._rand_idx()]
                outputs.append(tensor[i, ..., top : top + crop, left : left + crop])
            return torch.stack(outputs, dim=0)

        top, left = self.offsets[self._rand_idx()]
        return tensor[..., top : top + crop, left : left + crop]


"""
frozen backbone
"""


dropout_add_layer_norm = fused_mlp_func = None
flash_attn_qkvpacked_func = None


def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL

    if attn_mask is not None:
        attn.add_(attn_mask)

    return (
        F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
        if dropout_p > 0
        else attn.softmax(dim=-1)
    ) @ value


class MLPNoDrop(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        fused_if_available=True,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func  # None for TPU
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation="gelu_approx",
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=0,
                process_group=None,
            )
        else:
            return self.fc2(self.act(self.fc1(x)))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class SelfAttentionNoDrop(nn.Module):
    def __init__(
        self,
        block_idx,
        embed_dim=768,
        num_heads=12,
        flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )  # =64
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv, self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=True), nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.using_sdpa = bool(flash_if_available) and hasattr(
            F, "scaled_dot_product_attention"
        )

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
        if self.using_sdpa:
            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ],
                set_priority=True,
            ):
                oup = F.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, is_causal=False
                )
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale)
        return self.proj(oup.transpose(1, 2).reshape(B, L, C))

    def extra_repr(self) -> str:
        return f"using_sdpa={self.using_sdpa}"


class SABlockNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim, num_heads, mlp_ratio, norm_eps):
        super(SABlockNoDrop, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            flash_if_available=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            fused_if_available=True,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


recipes = {
    "S_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "S_8": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 8,
        "in_chans": 3,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "B_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class FrozenDINONoDrop(nn.Module):
    def __init__(
        self,
        depth=12,
        key_depths=(2, 5, 8, 11),
        norm_eps=1e-6,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        crop_prob: float = -0.5,
        no_resize: bool = False,
        original_input_size: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.img_size = 224
        self.original_input_size = (
            original_input_size if original_input_size is not None else self.img_size
        )
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_size = patch_size
        self.patch_nums = self.img_size // patch_size

        # x \in [-1, 1]
        # x = ((x+1)/2 - m) / s = 0.5x/s + 0.5/s - m/s = (0.5/s) x + (0.5-m)/s
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))
        self.crop = RandomWindowCrop(self.original_input_size, self.img_size, 9, False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim)
        )

        self.key_depths = set(d for d in key_depths if d < depth)
        self.blocks = nn.Sequential(
            *[
                SABlockNoDrop(
                    block_idx=i,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_eps=norm_eps,
                )
                for i in range(max(depth, 1 + max(self.key_depths, default=0)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.crop_prob = crop_prob
        self.no_resize = no_resize
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def inter_pos_embed(self, patch_nums=(14, 14)):
        if patch_nums[0] == self.patch_nums and patch_nums[1] == self.patch_nums:
            return self.pos_embed
        pe_cls, pe_grid = self.pos_embed[:, :1], self.pos_embed[0, 1:]
        pe_grid = pe_grid.reshape(1, self.patch_nums, self.patch_nums, -1).permute(
            0, 3, 1, 2
        )
        pe_grid = F.interpolate(
            pe_grid, size=patch_nums, mode="bilinear", align_corners=False
        )
        pe_grid = pe_grid.permute(0, 2, 3, 1).reshape(
            1, patch_nums[0] * patch_nums[1], -1
        )
        return torch.cat([pe_cls, pe_grid], dim=1)

    def forward(self, x, grad_ckpt=False, return_cls=False):
        """
        Returns the per-key-depth patch-token activations. `return_cls` also
        hands back the trunk's [CLS] embedding -- the final block's class token
        through the trunk LayerNorm, i.e. exactly what DINO's own head reads and
        what a linear probe is fit on. It is dropped by default (and out of the
        activation list entirely) because the discriminator scores patch tokens.
        """
        size = self.img_size if self.no_resize else self.original_input_size
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(
                x,
                size=(size, size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        if (
            not self.no_resize
            and self.crop_prob > 0
            and torch.rand(()) < self.crop_prob
        ):
            x = self.crop(x)

        x = x * self.x_scale + self.x_shift
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.shape[1] != self.pos_embed.shape[1]:
            h = w = int(math.sqrt(x.shape[1] - 1))
            pos_embed = self.inter_pos_embed((h, w))
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed

        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.key_depths:
                activations.append(x[:, 1:, :].transpose(1, 2))
        activations.insert(0, x[:, 1:, :].transpose(1, 2))
        if return_cls:
            return activations, self.norm(x[:, 0])
        return activations


"""
featurizer
"""


class DinoV1NoDrop(nn.Module):
    """
    Per-block patch-token featurizer over a frozen DINO v1 trunk.

        forward -> [ [B, N, D] ] per depth in `layers`, ascending.

    Every block runs whatever `layers` selects, so a shallow selection saves
    nothing here -- unlike the ViT featurizers, this trunk has no early exit.

    With `return_final` forward hands back a tuple instead, the second entry
    being a dict of head-side tensors (free: the trunk runs to the end anyway):

        aux["final_tokens"] : [B, N, D] last block's patch tokens, raw
        aux["final_pooled"] : [B, D]    last block's [CLS] through the trunk
                                        LayerNorm -- what DINO's head reads
    """

    def __init__(
        self,
        dino_ckpt_path: str,
        recipe: str = "S_8",
        layers: list | None = None,
        img_size: int = 224,
        device: torch.device = "cpu",
        return_final: bool = False,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 5, 8, 11]
        self.return_final = bool(return_final)
        state = torch.load(dino_ckpt_path, map_location="cpu")

        # same qkv-bias fix-up the discriminator applies on load
        for key in sorted(state.keys()):
            if ".attn.qkv.bias" in key:
                bias = state[key]
                C = bias.numel() // 3
                bias[C : 2 * C].zero_()

        cfg = dict(recipes[recipe])
        depth = cfg["depth"]
        # resolved, sorted, de-duplicated layer depths within range.
        resolved = sorted({int(d) for d in layers if 0 <= int(d) < depth})
        assert resolved, (
            f"dino_layers={layers} resolved to nothing for recipe={recipe} "
            f"(depth={depth}); pick block depths in [0, {depth})"
        )

        cfg.update(
            {
                "key_depths": tuple(resolved),
                "norm_eps": 1e-6,
                "original_input_size": img_size,
            }
        )
        dino = FrozenDINONoDrop(**cfg)
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [
            m for m in missing if all(x not in m for x in {"x_scale", "x_shift"})
        ]
        if missing:
            raise RuntimeError(f"DINO checkpoint missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"DINO checkpoint has unexpected keys: {unexpected}")
        dino.eval().requires_grad_(False)

        # proxy tuple: keep DINO out of .parameters() / state_dict / DDP
        self.proxy = (dino.to(device),)
        self.layers = resolved
        self.patch_size = dino.patch_size
        self.embed_dim = dino.embed_dim
        self.img_size = img_size
        if self.img_size != dino.img_size:
            self.num_tokens = (self.img_size // self.patch_size) ** 2
        else:
            self.num_tokens = dino.patch_nums**2

        # per-block shapes (ascending block order, matching forward's output)
        self.embed_dims = [self.embed_dim] * len(resolved)
        self.num_tokens_list = [self.num_tokens] * len(resolved)
        self.final_dim = self.embed_dim if self.return_final else None

    def forward(self, x: torch.Tensor):
        # acts[0] is the last block's patch tokens; acts[1:] are the extracted
        # key depths, ascending. Every block runs whatever key_depths hold.
        if not self.return_final:
            acts = self.proxy[0](x.float(), grad_ckpt=False)
            return [a.transpose(1, 2).contiguous() for a in acts[1:]]
        acts, cls = self.proxy[0](x.float(), grad_ckpt=False, return_cls=True)
        feats = [a.transpose(1, 2).contiguous() for a in acts[1:]]
        aux = {
            "final_tokens": acts[0].transpose(1, 2).float().contiguous(),
            "final_pooled": cls.float(),
        }
        return feats, aux
