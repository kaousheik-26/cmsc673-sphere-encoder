"""
Phase 0a gate: GS fit on GROUND-TRUTH renders. Train on ring32 even views (16,
= SyncDreamer sd16 cameras), init from their GT depth; evaluate on ring32 odd
views (interleaved holdout) and ext8 (extrapolated elevations 0 / 60 deg).
Target: interleaved held-out PSNR >= 30 dB.

Also reusable: `load_views(dir, idx)` loads a rendered protocol folder.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image

from se3d.eval.metrics import all_metrics
from se3d.gs.fit_gsplat import fit_gs, points_from_views, render_gs

DATA = "/fs/nexus-scratch/vinayakg/se3d/data/gso_renders"


def load_views(folder, idx=None, device="cuda"):
    meta = json.load(open(osp.join(folder, "cameras.json")))
    frames = meta["frames"] if idx is None else [meta["frames"][i] for i in idx]
    rgb, alpha, depth, w2c = [], [], [], []
    for fr in frames:
        a = np.array(Image.open(osp.join(folder, fr["file"]))).astype(np.float32) / 255.0
        rgb.append(a[..., :3])
        alpha.append(a[..., 3])
        dp = osp.join(folder, fr["depth"])
        depth.append(np.load(dp) if osp.exists(dp) else np.zeros(a.shape[:2], np.float32))
        w2c.append(np.array(fr["w2c"], dtype=np.float32))
    t = lambda x: torch.tensor(np.stack(x), device=device)
    return {"rgb": t(rgb), "alpha": t(alpha), "depth": t(depth), "w2c": t(w2c),
            "K": torch.tensor(meta["K"], dtype=torch.float32, device=device), "frames": frames}


def eval_views(gs, views):
    H, W = views["rgb"].shape[1:3]
    preds = []
    for i in range(views["rgb"].shape[0]):
        with torch.no_grad():
            rgb, alpha, _, _ = render_gs(gs, views["w2c"][i], views["K"], H, W)
        preds.append(rgb[0].clamp(0, 1))
    pred = torch.stack(preds).permute(0, 3, 1, 2)
    gt = views["rgb"].permute(0, 3, 1, 2)
    return all_metrics(pred, gt, views["alpha"]), pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lpips_weight", type=float, default=0.0)
    p.add_argument("--out", default="/fs/nexus-scratch/vinayakg/se3d/results/0a_gs_gt_ceiling")
    args = p.parse_args()
    objs = args.objects or sorted(os.listdir(osp.join(DATA, "ring32")))
    os.makedirs(args.out, exist_ok=True)
    rows = {}
    for obj in objs:
        tr = load_views(osp.join(DATA, "ring32", obj), list(range(0, 32, 2)))
        te = load_views(osp.join(DATA, "ring32", obj), list(range(1, 32, 2)))
        ex = load_views(osp.join(DATA, "ext8", obj))
        xyz, rgb = points_from_views(tr["rgb"], tr["depth"], tr["alpha"], tr["w2c"], tr["K"], stride=2)
        gs = fit_gs(tr["rgb"], tr["w2c"], tr["K"], xyz, rgb, iters=args.iters,
                    lpips_weight=args.lpips_weight)
        m_tr, _ = eval_views(gs, tr)
        m_te, pred = eval_views(gs, te)
        m_ex, _ = eval_views(gs, ex)
        rows[obj] = {"fit_s": round(gs.fit_seconds, 1), "n_gauss": gs.n, "init_pts": int(xyz.shape[0]),
                     "train": m_tr, "interleaved": m_te, "extrapolated": m_ex}
        print(obj, json.dumps({k: (round(v["psnr"], 2), round(v["psnr_masked"], 2), round(v["lpips_vgg"], 3))
                               for k, v in rows[obj].items() if isinstance(v, dict)}),
              f"fit {gs.fit_seconds:.1f}s n={gs.n}", flush=True)
    agg = {}
    for split in ["train", "interleaved", "extrapolated"]:
        agg[split] = {k: float(np.mean([rows[o][split][k] for o in rows])) for k in rows[objs[0]][split]}
    agg["fit_s"] = float(np.mean([rows[o]["fit_s"] for o in rows]))
    json.dump({"per_object": rows, "mean": agg, "args": vars(args)}, open(osp.join(args.out, "metrics.json"), "w"), indent=1)
    print("MEAN", json.dumps(agg))
    print("GT_CEILING_DONE")


if __name__ == "__main__":
    main()
