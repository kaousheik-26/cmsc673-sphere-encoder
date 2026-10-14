import torch
import torch.nn.functional as F


def half_mask(B, H, W, split="lr", observed="first", device="cpu", dtype=torch.float32):
    m = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    if split == "lr":
        m[..., : W // 2] = 1.0
    elif split == "tb":
        m[..., : H // 2, :] = 1.0
    else:
        raise ValueError(split)
    return m if observed == "first" else 1.0 - m


def fill_missing(y, mA, mode="mirror", split="lr"):
    if mode == "zeros":
        fill = torch.zeros_like(y)
    elif mode == "mean":
        mu = (y * mA).sum((2, 3), keepdim=True) / mA.sum((2, 3), keepdim=True)
        fill = mu.expand_as(y)
    elif mode == "mirror":
        fill = torch.flip(y, dims=[3 if split == "lr" else 2])
    elif mode == "blur":
        k = 31
        num = F.avg_pool2d(y * mA, k, stride=1, padding=k // 2)
        den = F.avg_pool2d(mA, k, stride=1, padding=k // 2).clamp_min(1e-3)
        fill = num / den
        for _ in range(8):
            num = F.avg_pool2d(fill, k, stride=1, padding=k // 2)
            fill = mA * y + (1 - mA) * num
    else:
        raise ValueError(mode)
    return mA * y + (1 - mA) * fill


def prox_mask(x_prior, y, mA, lam):
    return mA * (y + lam * x_prior) / (1.0 + lam) + (1.0 - mA) * x_prior


def token_index_for_half(H, W, patch, split="lr", observed="first"):
    h, w = H // patch, W // patch
    rr, cc = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    inA = cc < w // 2 if split == "lr" else rr < h // 2
    if observed != "first":
        inA = ~inA
    flat = inA.flatten()
    return flat.nonzero().squeeze(1), (~flat).nonzero().squeeze(1)
