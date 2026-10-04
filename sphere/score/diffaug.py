# Differentiable Augmentation
# Modified from
#   https://github.com/mit-han-lab/data-efficient-gans

import torch

# ----------------------------------------------------------------------------
# stateless ops. each returns the FULLY augmented batch (per-sample random
# parameters); the per-sample probability gating is handled by DiffAug below.
# strengths default to the original DiffAugment magnitudes. images are assumed
# channels-first [B, C, H, W] and roughly unit-scaled (e.g. [-1, 1]).
# ----------------------------------------------------------------------------


def rand_brightness(x, strength=0.5):
    # additive delta in [-strength, +strength]
    delta = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x = x + (delta * 2 - 1) * strength
    return x


def rand_saturation(x, strength=1.0):
    # scale saturation by a factor in [1 - strength, 1 + strength]
    x_mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    factor = 1 + (factor * 2 - 1) * strength
    x = (x - x_mean) * factor + x_mean
    return x


def rand_contrast(x, strength=0.5):
    # scale contrast by a factor in [1 - strength, 1 + strength]
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    factor = 1 + (factor * 2 - 1) * strength
    x = (x - x_mean) * factor + x_mean
    return x


def _wrap_indices(idx, size, mode):
    # map (possibly out-of-range) integer indices back into [0, size-1].
    if mode == "circular":
        return torch.remainder(idx, size)
    if mode == "reflect":  # reflect_101 (no edge repeat), like F.pad(mode="reflect")
        if size == 1:
            return torch.zeros_like(idx)
        period = 2 * (size - 1)
        idx = torch.remainder(idx, period)
        return torch.where(idx >= size, period - idx, idx)
    raise ValueError(f"unknown translation padding_mode: {mode}")


def rand_translation(x, ratio=0.125, padding_mode="zeros"):
    # per-sample integer shift up to +-ratio of each spatial dim. padding_mode
    # fills the vacated strip:
    #   "zeros"    - black fill (classic DiffAugment; leaks a border tell into a
    #                fake-only distribution critic).
    #   "reflect"  - mirror the edge (no border artifact; leak-safe).
    #   "circular" - wrap around (no border artifact; leak-safe, wrap seam).
    B, _, H, W = x.shape
    shift_h, shift_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    tx = torch.randint(-shift_h, shift_h + 1, size=[B, 1, 1], device=x.device)
    ty = torch.randint(-shift_w, shift_w + 1, size=[B, 1, 1], device=x.device)
    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=x.device),
        torch.arange(H, dtype=torch.long, device=x.device),
        torch.arange(W, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    src_h = grid_h + tx  # [B, H, W] source coords each output pixel reads from
    src_w = grid_w + ty
    if padding_mode == "zeros":
        valid = (src_h >= 0) & (src_h < H) & (src_w >= 0) & (src_w < W)
        src_h, src_w = src_h.clamp(0, H - 1), src_w.clamp(0, W - 1)
    else:
        src_h = _wrap_indices(src_h, H, padding_mode)
        src_w = _wrap_indices(src_w, W, padding_mode)
        valid = None
    x = (
        x.permute(0, 2, 3, 1)
        .contiguous()[grid_b, src_h, src_w]  # [B, H, W, C]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    if valid is not None:
        x = x * valid.unsqueeze(1).to(x.dtype)
    return x


def rand_cutout(x, ratio=0.5):
    # zero out a random square of side ~ratio * H/W per sample
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_hflip(x):
    # horizontal flip of every sample; per-sample gating handled by DiffAug
    return torch.flip(x, dims=[-1])


class DiffAug:
    """
    Per-module differentiable augmentation
    """

    def __init__(
        self,
        brightness_prob=0.0,
        saturation_prob=0.0,
        contrast_prob=0.0,
        translation_prob=0.0,
        flip_prob=0.0,
        cutout_prob=0.0,
        brightness_strength=0.5,
        saturation_strength=1.0,
        contrast_strength=0.5,
        translation_ratio=0.125,
        translation_padding="zeros",
        cutout_ratio=0.5,
    ):
        # (name, callable(x) -> aug'd x, per-sample prob); ordered pipeline.
        self.pipeline = [
            (
                "brightness",
                lambda x: rand_brightness(x, brightness_strength),
                brightness_prob,
            ),
            (
                "saturation",
                lambda x: rand_saturation(x, saturation_strength),
                saturation_prob,
            ),
            ("contrast", lambda x: rand_contrast(x, contrast_strength), contrast_prob),
            (
                "translation",
                lambda x: rand_translation(x, translation_ratio, translation_padding),
                translation_prob,
            ),
            ("flip", rand_hflip, flip_prob),
            ("cutout", lambda x: rand_cutout(x, cutout_ratio), cutout_prob),
        ]

    def __str__(self):
        active = ", ".join(
            f"{name}(p={prob:g})" for name, _, prob in self.pipeline if prob > 0
        )
        return f"DiffAug({active or 'identity'})"

    __repr__ = __str__

    @staticmethod
    def _blend(x, x_aug, prob):
        # per-sample Bernoulli(prob) selection between augmented and original,
        # differentiable w.r.t. both branches.
        if prob >= 1.0:
            return x_aug
        mask = (torch.rand(x.size(0), device=x.device) < prob).view(-1, 1, 1, 1)
        return torch.where(mask, x_aug, x)

    def aug(self, x):
        _dtype = x.dtype
        x = x.float()
        for _, fn, prob in self.pipeline:
            if prob <= 0.0:
                continue
            x = self._blend(x, fn(x), prob)
        return x.to(_dtype)
