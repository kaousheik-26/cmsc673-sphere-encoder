"""
Single-object depth diagnostic: how accurate is MoGe-2 on clean renders of one
object (per view, after the best LSQ scale+shift on the object mask)? Reports
AbsRel and delta<1.05 per view and the depth error at silhouette edges.
Usage: python se3d/probes/depth_check.py --object hat
"""
import argparse
import json

import torch
import torch.nn.functional as F

from se3d.geometry.depth import MoGeDepth, align_scale_shift, fov_x_from_K
from se3d.gs.gt_ceiling import DATA, load_views


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", default="hat")
    p.add_argument("--protocol", default="dev_orbit")
    args = p.parse_args()
    v = load_views(f"{DATA}/{args.protocol}/{args.object}")
    md = MoGeDepth()
    rows = []
    for i, fr in enumerate(v["frames"]):
        gt, m = v["depth"][i], (v["alpha"][i] > 0.5) & (v["depth"][i] > 0)
        d, mv = md(v["rgb"][i].permute(2, 0, 1), fov_x_from_K(v["K"], 256))
        mm = m & mv
        s, t = align_scale_shift(d, gt, mm, robust_iters=0)
        da = s * d + t
        rel = ((da - gt).abs() / gt)[mm]
        er = -F.max_pool2d(-m.float()[None, None], 7, 1, 3)[0, 0] > 0.5  # interior (eroded 3 px)
        edge = mm & ~er
        rows.append({"az": fr["azimuth"], "absrel": rel.mean().item(), "d105": (rel < 0.05).float().mean().item(),
                     "absrel_edge": ((da - gt).abs() / gt)[edge].mean().item() if edge.any() else None,
                     "absrel_interior": ((da - gt).abs() / gt)[mm & er].mean().item(), "mask_iou_moge":
                     ((m & mv).sum() / (m | mv).sum()).item()})
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rows[-1].items()}), flush=True)
    print("MEAN absrel %.4f  d<1.05 %.3f  edge %.4f  interior %.4f" % (
        sum(r["absrel"] for r in rows) / len(rows), sum(r["d105"] for r in rows) / len(rows),
        sum(r["absrel_edge"] or 0 for r in rows) / len(rows), sum(r["absrel_interior"] for r in rows) / len(rows)))
    print("DEPTH_CHECK_DONE")


if __name__ == "__main__":
    main()
