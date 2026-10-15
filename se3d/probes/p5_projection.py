"""
Probe P5: SE2 projection on degraded (warped + pre-filled) inputs, no data consistency.

Source = dev_orbit az 0; targets az {15, 30, 45} (elev 30). The source is
forward-warped to the target with GT depth, holes are cleaned (mask hygiene) and
pre-filled (nearest). Then x <- D(rotate(F(E(x)), e, a), y=null, a) is iterated 10
times at fixed a in {20, 40, 60, 75, 85} (shared e per object). Per iteration:
LPIPS / PSNR to the GT target view (whole image, known pixels, hole pixels) and
KID vs the GT renders of all GSO30 ring32 views. Iteration 0 = the pre-filled warp.
Output: metrics.json, curves png, a grid per object.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from torchvision.utils import save_image

from se3d.eval.kid import inception_features, kid
from se3d.eval.metrics import lpips, psnr
from se3d.geometry.warp import forward_warp, mask_hygiene, prefill, prefill_object, with_background
from se3d.gs.gt_ceiling import DATA, load_views
from se3d.se2.wrapper import SE2

DEV_GSO = ["bell", "chicken", "hat", "shoe1", "teapot", "toaster", "turtle"]
TARGET_AZ = [15, 30, 45]
ALPHAS = [20, 40, 60, 75, 85]


def warped_inputs(v, j):
    """warp source (view 0) to view j with GT depth; white backdrop known outside
    hull+margin of the warped object; Text2Room hygiene; nearest pre-fill."""
    rgb, known_obj, _ = forward_warp(v["rgb"][0], v["depth"][0], v["K"], v["w2c"][0], v["K"], v["w2c"][j], 256, 256)
    rgb, known = with_background(rgb, known_obj, margin=8)
    rgb_c, known_c, gen = mask_hygiene(rgb, known)
    x = prefill_object(rgb_c, known_c, known_obj, "nn")  # hull-aware (no halo)
    return x, known_c, gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="large-fdlite-256")
    p.add_argument("--objects", nargs="*", default=DEV_GSO)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/fs/nexus-scratch/vinayakg/se3d/results/0b_p5_projection")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    se2 = SE2(args.model)

    # KID reference: GT renders of all GSO30 objects (ring32)
    ref = []
    for obj in sorted(os.listdir(osp.join(DATA, "ring32"))):
        ref.append(load_views(osp.join(DATA, "ring32", obj))["rgb"].permute(0, 3, 1, 2))
    f_ref = inception_features(torch.cat(ref))

    samples = []  # (obj, az, x0, gt, alpha_gt, known, gen)
    for oi, obj in enumerate(args.objects):
        v = load_views(osp.join(DATA, "dev_orbit", obj))
        azs = [fr["azimuth"] for fr in v["frames"]]
        for az in TARGET_AZ:
            j = azs.index(float(az))
            x0, known, gen = warped_inputs(v, j)
            samples.append((obj, oi, az, x0.permute(2, 0, 1), v["rgb"][j].permute(2, 0, 1), v["alpha"][j], known, gen))

    X0 = torch.stack([s[3] for s in samples])
    GT = torch.stack([s[4] for s in samples])
    KN = torch.stack([s[6] for s in samples]).float()
    HO = 1 - KN
    E = torch.cat([se2.sample_noise(1, torch.Generator(device="cuda").manual_seed(args.seed * 1000 + s[1]))
                   for s in samples])

    def score(x):
        return {
            "lpips": lpips(x, GT).mean().item(),
            "lpips_hole": lpips(x * HO[:, None] + (1 - HO[:, None]), GT * HO[:, None] + (1 - HO[:, None])).mean().item(),
            "psnr": psnr(x, GT).mean().item(),
            "psnr_known": psnr(x, GT, KN).mean().item(),
            "psnr_hole": psnr(x, GT, HO).mean().item(),
            "kid": kid(inception_features(x), f_ref),
        }

    res = {"iter0": score(X0)}
    print("iter0", res["iter0"], flush=True)
    finals = {}
    for a in ALPHAS:
        x = X0.clone()
        curve = []
        for it in range(args.iters):
            with torch.no_grad():
                x = se2.to01(se2.decode(se2.rotate(se2.encode(x * 2 - 1), E, a), None, a))
            curve.append(score(x))
            if it in (0, 2, args.iters - 1):
                finals[(a, it)] = x.clone()
        res[f"alpha{a}"] = curve
        print(f"alpha {a}: " + " | ".join(f"it{i+1} L{c['lpips']:.3f} Lh{c['lpips_hole']:.3f} K{c['kid']*1e3:.1f}"
                                          for i, c in enumerate(curve) if i in (0, 2, 4, 9)), flush=True)

    json.dump({"results": res, "objects": args.objects, "targets": TARGET_AZ, "model": args.model},
              open(osp.join(args.out, "metrics.json"), "w"), indent=1)
    # grid: rows = samples (first 6), cols = GT, warp+prefill, a=20/40/60/75/85 after 1 and 10 its
    cols = [GT[:6], X0[:6]] + [finals[(a, 0)][:6] for a in ALPHAS] + [finals[(a, args.iters - 1)][:6] for a in ALPHAS]
    grid = torch.stack(cols, 1).flatten(0, 1)
    save_image(grid, osp.join(args.out, "grid.png"), nrow=len(cols))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a in ALPHAS:
        c = res[f"alpha{a}"]
        it = np.arange(0, len(c) + 1)
        for k, key in enumerate(["lpips", "lpips_hole", "kid"]):
            ax[k].plot(it, [res["iter0"][key]] + [x[key] for x in c], "o-", ms=3, label=f"a={a}")
    for k, key in enumerate(["LPIPS to GT (image)", "LPIPS to GT (holes only)", "KID vs GSO renders"]):
        ax[k].set_title(key); ax[k].set_xlabel("iteration"); ax[k].grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(osp.join(args.out, "p5_curves.png"), dpi=130)
    print("P5_DONE")


if __name__ == "__main__":
    main()
