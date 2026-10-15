"""
Monocular depth (MoGe-2) + alignment helpers.

    md = MoGeDepth()                       # Ruicheng/moge-2-vitl-normal
    d, m = md(img01 [3,H,W], fov_x_deg)    # depth along z (metric-ish), valid mask
    s, t = align_scale_shift(d, d_ref, mask)   # LSQ: s*d + t ~= d_ref on mask
"""
import functools

import torch


class MoGeDepth:
    def __init__(self, name="Ruicheng/moge-2-vitl-normal", device="cuda"):
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img01, fov_x_deg=None):
        out = self.model.infer(img01.to(self.device).float(), fov_x=fov_x_deg)
        d = out["depth"]
        m = out["mask"].bool() & torch.isfinite(d) & (d > 0)
        return torch.where(m, d, torch.zeros_like(d)), m


def fov_x_from_K(K, width):
    import math

    return math.degrees(2 * math.atan(width / (2 * float(K[0][0]))))


def align_scale_shift(pred, ref, mask, robust_iters=2):
    """least squares s, t with s*pred + t ~= ref over mask (then 2 rounds of
    trimming the worst 10% residuals)."""
    m = mask & (pred > 0) & (ref > 0)
    a, b = pred[m].double(), ref[m].double()
    if a.numel() < 10:
        return 1.0, 0.0
    keep = torch.ones_like(a, dtype=torch.bool)
    for _ in range(robust_iters + 1):
        A = torch.stack([a[keep], torch.ones_like(a[keep])], 1)
        sol = torch.linalg.lstsq(A, b[keep][:, None]).solution[:, 0]
        r = (sol[0] * a + sol[1] - b).abs()
        keep = r <= torch.quantile(r, 0.9)
    return float(sol[0]), float(sol[1])


def align_scale(pred, ref, mask):
    m = mask & (pred > 0) & (ref > 0)
    if m.sum() < 10:
        return 1.0
    return float((ref[m] / pred[m]).median())


def harmonic_fill(values, known, region, iters=300):
    """Fill `values` [H,W] inside `region` (bool) by harmonic (Laplace) interpolation
    from the `known` pixels (bool, kept fixed). Nearest-known initialisation + Jacobi."""
    import cv2
    import numpy as np
    import torch.nn.functional as F

    kn = known.cpu().numpy()
    if not kn.any():
        return values
    _, lab = cv2.distanceTransformWithLabels((~kn).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(kn)
    lut = np.zeros(lab.max() + 1, dtype=np.int64)
    lut[lab[ys, xs]] = ys * kn.shape[1] + xs
    init = values.flatten()[torch.tensor(lut[lab].reshape(-1), device=values.device)].view_as(values)
    v = torch.where(known, values, init)
    upd = (region & ~known)[None, None].float()
    ker = torch.tensor([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], device=values.device)[None, None]
    x = v[None, None]
    for _ in range(iters):
        avg = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), ker)
        x = upd * avg + (1 - upd) * x
    return x[0, 0]


def fuse_depth(d_mono, d_ref, known, region, iters=300):
    """Gradient-preserving depth for new pixels: global LSQ scale/shift of the
    monocular depth, then a smooth (harmonic) offset field that makes it agree with
    the reference depth along the known pixels (Invisible-Stitch-style seams)."""
    s, t = align_scale_shift(d_mono, d_ref, known & (d_mono > 0))
    d = s * d_mono + t
    off = torch.where(known, d_ref - d, torch.zeros_like(d))
    off = harmonic_fill(off, known, region, iters)
    return d + off
