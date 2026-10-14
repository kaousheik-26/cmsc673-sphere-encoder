import os
import os.path as osp

import numpy as np
import torch

from .geometry import angle_deg


def half_crop(x, split="lr", which="B"):
    H, W = x.shape[-2:]
    if split == "lr":
        return x[..., : W // 2] if which == "A" else x[..., W // 2 :]
    return x[..., : H // 2, :] if which == "A" else x[..., H // 2 :, :]


def psnr_masked(x, ref, m):
    mse = ((m * (x - ref)) ** 2).sum((1, 2, 3)) / (m.sum((1, 2, 3)) * x.shape[1])
    return 10 * torch.log10(4.0 / mse.clamp_min(1e-12))


def seam_score(x, split="lr", band=2):
    if split == "lr":
        g = (x[..., 1:] - x[..., :-1]).abs().mean((1, 2))
        c = x.shape[-1] // 2 - 1
    else:
        g = (x[..., 1:, :] - x[..., :-1, :]).abs().mean((1, 3))
        c = x.shape[-2] // 2 - 1
    at = g[:, c - band + 1 : c + band].mean(1)
    rest = torch.cat([g[:, : c - band + 1], g[:, c + band :]], 1).mean(1)
    return at / rest.clamp_min(1e-8)


def latent_angle(v_hat, v_true):
    return angle_deg(v_hat, v_true)


class HalfLPIPS:
    def __init__(self, device="cuda", ckpt_dir=None):
        from sphere.lpips import LPIPS

        ckpt_dir = ckpt_dir or osp.join(os.environ.get("SPHERE_WORKSPACE", "workspace"), "pretrained", "lpips")
        self.fn = LPIPS(ckpt_pth=ckpt_dir).to(device).eval()
        sd = torch.load(osp.join(ckpt_dir, "vgg.pth"), map_location="cpu")
        for k, v in sd.items():
            assert torch.equal(self.fn.state_dict()[k].cpu(), v), f"LPIPS head {k} not loaded"

    @torch.no_grad()
    def __call__(self, a, b, split="lr", which="B"):
        return self.fn(half_crop(a, split, which).clamp(-1, 1), half_crop(b, split, which).clamp(-1, 1)).view(-1)

    def loss(self, a, b, split="lr", which="A"):
        return self.fn(half_crop(a, split, which), half_crop(b, split, which)).view(-1)

    @torch.no_grad()
    def full(self, a, b):
        return self.fn(a.clamp(-1, 1), b.clamp(-1, 1)).view(-1)


def diversity(samples, split="lr", lp=None):
    S = samples.shape[0]
    vals = []
    for i in range(S):
        for j in range(i + 1, S):
            if lp is not None:
                vals.append(lp(samples[i], samples[j], split, "B"))
            else:
                vals.append((half_crop(samples[i], split) - half_crop(samples[j], split)).abs().mean((1, 2, 3)))
    return torch.stack(vals).mean(0)


class InceptionFeatures:
    def __init__(self, device="cuda", weight_path=None):
        from sphere.metric import create_metric_feature_extractor, ensure_inception_weights

        weight_path = weight_path or osp.join(
            os.environ.get("SPHERE_WORKSPACE", "workspace"),
            "pretrained/fid_pretrained_models/weights-inception-2015-12-05-6726825d.pth",
        )
        ensure_inception_weights(weight_path)
        self.net, _ = create_metric_feature_extractor(weight_path, device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, x, batch=64):
        out = []
        for i in range(0, x.shape[0], batch):
            u8 = ((x[i : i + batch].clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8)
            feats = self.net(u8.to(self.device))
            out.append(feats[0].double().cpu())
        return torch.cat(out)


def kid(feat_a, feat_b, subset_size=100, n_subsets=100, seed=0):
    a, b = feat_a.double(), feat_b.double()
    d = a.shape[1]
    m = min(subset_size, a.shape[0], b.shape[0])
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        x = a[torch.randperm(a.shape[0], generator=g)[:m]]
        y = b[torch.randperm(b.shape[0], generator=g)[:m]]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = (kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (
            m * (m - 1)
        ) - 2 * kxy.mean()
        vals.append(mmd.item())
    return float(np.mean(vals)), float(np.std(vals))


def fid(feat_a, feat_b):
    from sphere.metric import compute_fid

    a, b = feat_a.double(), feat_b.double()
    return float(compute_fid(a.mean(0), b.mean(0), torch.cov(a.T), torch.cov(b.T)))


def real_features(ds, incep, image_size, cache_dir=None, batch=128):
    from .data import load_batch

    cache_dir = cache_dir or osp.join(os.environ.get("SPHERE_STORE", "workspace"), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{osp.basename(osp.normpath(ds.root))}_{ds.split}_{image_size}px_n{len(ds)}"
    path = osp.join(cache_dir, f"inception_{tag}.pt")
    if osp.isfile(path):
        return torch.load(path)
    feats = torch.cat([incep(load_batch(ds, list(range(i, min(i + batch, len(ds)))), "cpu")[0]) for i in range(0, len(ds), batch)])
    torch.save(feats, path)
    return feats
