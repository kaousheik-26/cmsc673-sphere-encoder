"""
Forward warping with a z-buffer, mask hygiene and hole pre-fill (R1 / P4 / P5).

forward_warp(src_rgb, src_depth, K_src, w2c_src, K_tgt, w2c_tgt, H, W)
  -> rgb [H,W,3], known [H,W] bool, depth [H,W] (0 = hole)
Points are splatted to the nearest target pixel; the nearest depth wins
(scatter_reduce amin). `splat_radius=1` also writes the 4-neighbourhood at
slightly larger depth to close 1-px cracks when the target zooms in.

mask_hygiene (Text2Room): holes smaller than a 3x3 erosion are filled by Telea
inpainting and count as known; the remaining unknown mask is dilated 7x7 x5
before the generator sees it.

prefill: fill unknown pixels so the encoder never sees zeros ('nn' nearest
known pixel, 'median' masked median blur, 'blur' normalized Gaussian, 'telea').
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from se3d.geometry.cameras import project_points, unproject_depth


def splat_points(xyz, cols, K_tgt, w2c_tgt, H, W, splat_radius=1):
    """z-buffered splat of world points xyz [M,3] with features cols [M,C] into a
    target camera. Returns feat [H,W,C], known [H,W] bool, depth [H,W]."""
    dev = xyz.device
    uv, z = project_points(xyz, K_tgt, w2c_tgt)
    zbuf = torch.full((H * W,), float("inf"), device=dev)
    offs = [(0, 0)] if splat_radius == 0 else [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    cand = []
    for k, (dx, dy) in enumerate(offs):
        u = torch.floor(uv[:, 0]).long() + dx
        v = torch.floor(uv[:, 1]).long() + dy
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 1e-4)
        zz = z * (1.0 + 1e-3 * (k > 0))  # neighbours lose ties against true centres
        idx = v * W + u
        zbuf.scatter_reduce_(0, idx[ok], zz[ok], reduce="amin")
        cand.append((idx, ok, zz))
    known = torch.isfinite(zbuf)
    feat = torch.zeros(H * W, cols.shape[-1], device=dev)
    for idx, ok, zz in cand:
        win = ok & (zz <= zbuf[idx.clamp(0, H * W - 1)])
        feat[idx[win]] = cols[win]
    depth = torch.where(known, zbuf, torch.zeros_like(zbuf))
    return feat.view(H, W, -1), known.view(H, W), depth.view(H, W)


def forward_warp(src_rgb, src_depth, K_src, w2c_src, K_tgt, w2c_tgt, H, W, src_mask=None,
                 splat_radius=1, extra=None):
    """src_rgb [h,w,3], src_depth [h,w] (0 = invalid). extra: optional [h,w,C] warped too."""
    m = src_depth > 0 if src_mask is None else (src_mask & (src_depth > 0))
    pw, valid = unproject_depth(src_depth, K_src, w2c_src, m)
    cols = src_rgb[valid]
    if extra is not None:
        cols = torch.cat([cols, extra[valid]], -1)
    feat, known, depth = splat_points(pw, cols, K_tgt, w2c_tgt, H, W, splat_radius)
    if extra is not None:
        return feat[..., :3], known, depth, feat[..., 3:]
    return feat, known, depth


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def mask_hygiene(rgb, known, erode=3, dilate=7, dilate_iters=5, small_hole_inpaint=True):
    """returns (rgb_filled, known_clean, gen_mask). gen_mask = pixels the generator
    may change (dilated unknown). rgb in [0,1] torch [H,W,3], known bool [H,W]."""
    dev = rgb.device
    k = _np(known).astype(np.uint8)
    img = (np.clip(_np(rgb), 0, 1) * 255).astype(np.uint8)
    unknown = 1 - k
    if small_hole_inpaint:
        # holes that vanish under a 3x3 opening of the known mask are "small"
        ker = np.ones((erode, erode), np.uint8)
        closed = cv2.morphologyEx(k, cv2.MORPH_CLOSE, ker)
        small = (closed == 1) & (k == 0)
        if small.any():
            img = cv2.inpaint(img, small.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            k = np.where(small, 1, k).astype(np.uint8)
            unknown = 1 - k
    gen = cv2.dilate(unknown, np.ones((dilate, dilate), np.uint8), iterations=dilate_iters)
    t = lambda a: torch.tensor(a, device=dev)
    return t(img.astype(np.float32) / 255.0), t(k.astype(bool)), t(gen.astype(bool))


def prefill(rgb, known, mode="nn"):
    """fill unknown pixels of rgb [H,W,3] in [0,1] (torch)."""
    dev = rgb.device
    img = (np.clip(_np(rgb), 0, 1) * 255).astype(np.uint8)
    k = _np(known).astype(bool)
    if k.all():
        return rgb
    if not k.any():
        return torch.full_like(rgb, 0.5)
    if mode == "nn":
        _, labels = cv2.distanceTransformWithLabels((~k).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(k)
        lut = np.zeros(labels.max() + 1, dtype=np.int64)
        # label of each known pixel -> its flat index
        lut[labels[ys, xs]] = ys * k.shape[1] + xs
        src = lut[labels]
        out = img.reshape(-1, 3)[src.reshape(-1)].reshape(img.shape)
    elif mode == "telea":
        out = cv2.inpaint(img, (~k).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    elif mode in ("median", "blur"):
        # multi-scale normalized convolution: each hole pixel takes the smallest
        # scale at which known pixels reach it; 'median' then median-filters holes
        out = img.astype(np.float32)
        w = k.astype(np.float32)
        acc = out * w[..., None]
        res = out.copy()
        filled = k.copy()
        for s in [3, 7, 15, 31, 63, 127]:
            if mode == "blur":
                num, den = cv2.GaussianBlur(acc, (0, 0), s), cv2.GaussianBlur(w, (0, 0), s)
            else:
                num, den = cv2.blur(acc, (2 * s + 1, 2 * s + 1)), cv2.blur(w, (2 * s + 1, 2 * s + 1))
            new_px = (den > 1e-3) & ~filled
            res[new_px] = num[new_px] / den[new_px, None]
            filled |= new_px
            if filled.all():
                break
        res[~filled] = 127.0
        if mode == "median":
            med = cv2.medianBlur(np.clip(res, 0, 255).astype(np.uint8), 5).astype(np.float32)
            res = np.where(k[..., None], out, med)
        out = np.clip(res, 0, 255).astype(np.uint8)
    else:
        raise ValueError(mode)
    out = np.where(k[..., None], img, out)
    return torch.tensor(out.astype(np.float32) / 255.0, device=dev)


def with_background(rgb, known_obj, bg=1.0, margin=16):
    """Object-centric scenes on a constant backdrop: pixels farther than `margin`
    px from the convex hull of the warped object are known background (white);
    pixels inside hull+margin that the warp missed stay unknown (disocclusions and
    the silhouette band). Returns (rgb with bg painted, known_total bool)."""
    dev = rgb.device
    k = _np(known_obj).astype(np.uint8)
    region = np.zeros_like(k)
    pts = cv2.findNonZero(k)
    if pts is not None:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(region, hull, 1)
        region = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=margin)
    far_bg = (region == 0)
    known = k.astype(bool) | far_bg
    out = _np(rgb).copy()
    out[far_bg] = bg
    return torch.tensor(out, device=dev, dtype=torch.float32), torch.tensor(known, device=dev)


def prefill_object(rgb, known, known_obj, mode="nn", bg=1.0):
    """Object-centric pre-fill: unknown pixels inside the convex hull of the
    known object pixels get `mode` fill (likely disocclusions); unknown pixels
    outside the hull start as background (avoids NN smearing a halo of object
    colours into the silhouette band)."""
    k = _np(known_obj).astype(np.uint8)
    hull = np.zeros_like(k)
    pts = cv2.findNonZero(k)
    if pts is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    outside = torch.tensor((hull == 0), device=rgb.device) & ~known
    base = rgb.clone()
    base[outside] = bg
    known2 = known | outside
    return prefill(base, known2, mode)
