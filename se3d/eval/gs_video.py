"""
Side-by-side videos (true | ours ...) and a Gaussian-splat fit on GENERATED views.
- visual_hull_points: GS init from generated views only (no GT): grid points inside
  the object's silhouette (non-white) in at least `frac` of the views that see them.
- write_panels: one mp4, each frame = labelled panels side by side.
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def visual_hull_points(images, w2cs, K, res=128, bound=0.55, frac=0.8, thr=0.08, max_pts=80_000):
    """images [N,H,W,3] in [0,1] (white background), w2cs [N,4,4], K [3,3] -> xyz [P,3], rgb [P,3]."""
    dev = images.device
    N, H, W, _ = images.shape
    g = torch.linspace(-bound, bound, res, device=dev)
    xyz = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).view(-1, 3)
    fg = ((1 - images).amax(-1) > thr).float()  # [N,H,W]
    fg = F.max_pool2d(fg[:, None], 5, 1, 2)[:, 0]  # 2 px slack for view-to-view jitter
    K = torch.as_tensor(K, dtype=torch.float32, device=dev)
    w2cs = torch.as_tensor(w2cs, dtype=torch.float32, device=dev)
    hits = torch.zeros(xyz.shape[0], device=dev)
    seen = torch.zeros(xyz.shape[0], device=dev)
    col = torch.zeros(xyz.shape[0], 3, device=dev)
    for i in range(N):
        pc = xyz @ w2cs[i, :3, :3].T + w2cs[i, :3, 3]
        uv = pc @ K.T
        uv = uv[:, :2] / uv[:, 2:].clamp_min(1e-6)
        grid = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)[None, None]
        inside = (grid[0, 0].abs() <= 1).all(-1) & (pc[:, 2] > 0)
        f = F.grid_sample(fg[i][None, None], grid, align_corners=False)[0, 0, 0]
        c = F.grid_sample(images[i].permute(2, 0, 1)[None], grid, align_corners=False)[0, :, 0].T
        seen += inside.float()
        hits += inside.float() * (f > 0.5).float()
        col += inside.float()[:, None] * (f > 0.5).float()[:, None] * c
    keep = (seen > 0) & (hits >= frac * seen)
    xyz, rgb = xyz[keep], (col[keep] / hits[keep, None].clamp_min(1)).clamp(0, 1)
    if xyz.shape[0] > max_pts:
        idx = torch.randperm(xyz.shape[0], device=dev)[:max_pts]
        xyz, rgb = xyz[idx], rgb[idx]
    xyz = xyz + (torch.rand_like(xyz) - 0.5) * (2 * bound / res)  # break the grid
    return xyz, rgb


def write_panels(path, panels, fps=24):
    """panels: list of (label, frames) with frames [T,3,H,W] in [0,1] (T=1: held still)."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw

    T = max(f.shape[0] for _, f in panels)
    H, W = panels[0][1].shape[-2:]
    gap, top = 6, 24
    w = iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2, pixelformat="yuv420p")
    for t in range(T):
        F_ = Image.new("RGB", (len(panels) * (W + gap) - gap, H + top), "white")
        d = ImageDraw.Draw(F_)
        for k, (lab, fr) in enumerate(panels):
            x = fr[min(t, fr.shape[0] - 1)].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            F_.paste(Image.fromarray((x * 255).astype(np.uint8)), (k * (W + gap), top))
            d.text((k * (W + gap) + 6, 6), lab, fill=(30, 30, 30))
        w.append_data(np.array(F_))
    w.close()
