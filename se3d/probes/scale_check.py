"""Which single-image rule recovers the depth scale relative to a known orbit
(camera looks at the object centre from distance r)? For each GSO object's source
view (dev_orbit az 0): MoGe-2 depth d (scale-ambiguous), GT scale s* = LSQ
scale-only fit to GT depth on the mask. Candidate rules give s; report
|log(s / s*)| per rule (mean, median, max over objects)."""
import math
import os

import numpy as np
import torch

from se3d.geometry.cameras import unproject_depth
from se3d.geometry.depth import MoGeDepth, fov_x_from_K
from se3d.gs.gt_ceiling import DATA, load_views

R = 1.5


def rules(d, m, K, w2c):
    v = d[m]
    out = {f"q{q}": R / torch.quantile(v, q / 100).item() for q in (50, 75, 90, 95)}
    out["mean"] = R / v.mean().item()
    # centre-depth rule: object centre ~ front depth + half the lateral extent (isotropic thickness)
    pts, _ = unproject_depth(d, K, w2c, m)  # in units of d (scale 1), world frame
    cam = -w2c[:3, :3].T @ w2c[:3, 3]
    fwd = w2c[2, :3]
    rel = pts - cam
    depth_along = rel @ fwd
    lat = rel - depth_along[:, None] * fwd
    right, up = w2c[0, :3], w2c[1, :3]
    wx = (lat @ right).max() - (lat @ right).min()
    wy = (lat @ up).max() - (lat @ up).min()
    front = torch.quantile(depth_along, 0.02)
    for k, name in [(0.5, "centre_k0.5"), (1.0, "centre_k1"), (1.5, "centre_k1.5")]:
        thick = k * 0.5 * float(torch.maximum(wx, wy))
        out[name] = R / (front.item() + thick)
    # unit-box normalisation (SyncDreamer / SV3D protocols: max bbox side = 1):
    # visible points give a LOWER bound on every extent, so these over-estimate s
    aabb = (pts.max(0).values - pts.min(0).values).max().item()
    out["aabb_unit"] = 1.0 / aabb
    out["lateral_unit"] = 1.0 / float(torch.maximum(wx, wy))
    out["geo_q90_aabb"] = math.sqrt(out["q90"] * out["aabb_unit"])
    out["geo_centre_aabb"] = math.sqrt(out["centre_k0.5"] * out["aabb_unit"])
    return out


def main():
    md = MoGeDepth()
    res = {}
    for obj in sorted(os.listdir(f"{DATA}/dev_orbit")):
        v = load_views(f"{DATA}/dev_orbit/{obj}", [0])
        m = (v["alpha"][0] > 0.5) & (v["depth"][0] > 0)
        d, mv = md(v["rgb"][0].permute(2, 0, 1), fov_x_from_K(v["K"], 256))
        mm = m & mv
        s_star = float((v["depth"][0][mm] * d[mm]).sum() / (d[mm] ** 2).sum())
        rr = rules(d, mm, v["K"], v["w2c"][0])
        res[obj] = {k: abs(math.log(s / s_star)) for k, s in rr.items()}
        print(obj, {k: round(x, 3) for k, x in res[obj].items()}, flush=True)
    keys = next(iter(res.values())).keys()
    print("RULE  mean|log err|  median  max")
    for k in keys:
        a = np.array([res[o][k] for o in res])
        print(f"{k:12s} {a.mean():.4f} {np.median(a):.4f} {a.max():.4f}")
    print("SCALE_CHECK_DONE")


if __name__ == "__main__":
    main()
