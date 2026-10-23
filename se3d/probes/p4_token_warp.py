"""
Probe P4: warp the 16x16 token map with (oracle GT) depth, nearest token.

For target view j (az in {15,30,45,60,90}) and each target token centre on the
object: unproject with target depth, project into the source (az 0), check
visibility with source depth; take the source token there (spherified source
latent). Target background tokens -> mean source background token. Disoccluded
object tokens -> the token of F(e) (shared noise). Re-spherify, decode at
a in {0, 30, 60} with the same e. Baselines: source decoded as-is, pixel-warp
(+hygiene, pre-fill) -> E -> D at a=0, and the ceiling D(F(E(x_tgt))).
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from torchvision.utils import save_image

from se3d.eval.metrics import lpips, psnr
from se3d.geometry.cameras import project_points, unproject_depth
from se3d.gs.gt_ceiling import DATA, load_views
from se3d.probes.p5_projection import DEV_GSO, warped_inputs
from se3d.se2.wrapper import SE2

G, P = 16, 16
TARGETS = [15, 30, 45, 60, 90]


@torch.no_grad()
def token_source_index(v, j, rel_tol=0.03):
    """for each target token: source token index, or -1 (bg) / -2 (disoccluded)."""
    c = torch.arange(G, device="cuda") * P + P // 2
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    yy, xx = yy.flatten(), xx.flatten()
    on = (v["alpha"][j][yy, xx] > 0.5) & (v["depth"][j][yy, xx] > 0)
    idx = torch.full((G * G,), -1, dtype=torch.long, device="cuda")
    dimg = torch.zeros_like(v["depth"][j])
    dimg[yy[on], xx[on]] = v["depth"][j][yy[on], xx[on]]
    pw, valid = unproject_depth(dimg, v["K"], v["w2c"][j])
    uv, z = project_points(pw, v["K"], v["w2c"][0])
    pix = torch.nonzero(valid.flatten()).flatten()
    tok_of = {int(p): t for t, p in enumerate((yy * 256 + xx).tolist())}
    for k, pid in enumerate(pix.tolist()):
        t = tok_of[pid]
        u, vv = uv[k]
        ui, vi = int(u), int(vv)
        if 0 <= ui < 256 and 0 <= vi < 256:
            zs = v["depth"][0][vi, ui]
            if zs > 0 and abs(float(zs) - float(z[k])) < rel_tol * float(z[k]):
                idx[t] = (vi // P) * G + ui // P
                continue
        idx[t] = -2
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="large-fdlite-256")
    p.add_argument("--objects", nargs="*", default=DEV_GSO)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/fs/nexus-scratch/vinayakg/se3d/results/0b_p4_token_warp")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    se2 = SE2(args.model)
    rows, grids = [], []
    for oi, obj in enumerate(args.objects):
        v = load_views(osp.join(DATA, "dev_orbit", obj))
        azs = [fr["azimuth"] for fr in v["frames"]]
        x = v["rgb"].permute(0, 3, 1, 2)
        e = se2.sample_noise(1, torch.Generator(device="cuda").manual_seed(args.seed * 1000 + oi))
        with torch.no_grad():
            z = se2.encode(x * 2 - 1)  # [V,256,384] on the global sphere
        c = torch.arange(G, device="cuda") * P + P // 2
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        src_bg = v["alpha"][0][yy.flatten(), xx.flatten()] < 0.5
        bg_tok = z[0][src_bg].mean(0) if src_bg.any() else z[0].mean(0)
        for az in TARGETS:
            j = azs.index(float(az))
            gt = x[j:j + 1]
            idx = token_source_index(v, j)
            zw = torch.where((idx >= 0)[:, None], z[0][idx.clamp_min(0)], bg_tok[None])
            zw = torch.where((idx == -2)[:, None], e[0], zw)
            zw = se2.spherify(zw[None])
            r = {"obj": obj, "az": az, "frac_disoccluded": float((idx == -2).float().sum() / (idx != -1).float().sum().clamp_min(1)),
                 "frac_fg": float((idx != -1).float().mean())}
            outs = []
            with torch.no_grad():
                for a in [0, 30, 60]:
                    xa = se2.to01(se2.decode(se2.rotate(zw, e, a), None, a))
                    r[f"lpips_tokwarp_a{a}"] = lpips(xa, gt).item()
                    r[f"psnr_tokwarp_a{a}"] = psnr(xa, gt).item()
                    outs.append(xa)
                src_dec = se2.to01(se2.decode(z[0:1], None, 0.0))
                xw, _, _ = warped_inputs(v, j)
                pix = se2.to01(se2.decode(se2.encode(xw.permute(2, 0, 1)[None] * 2 - 1), None, 0.0))
                ceil = se2.to01(se2.decode(z[j:j + 1], None, 0.0))
            r.update({"lpips_source": lpips(src_dec, gt).item(), "lpips_pixwarp_a0": lpips(pix, gt).item(),
                      "lpips_ceiling": lpips(ceil, gt).item(), "psnr_pixwarp_a0": psnr(pix, gt).item(),
                      "psnr_ceiling": psnr(ceil, gt).item(), "psnr_source": psnr(src_dec, gt).item()})
            rows.append(r)
            if az in (30, 60) and oi < 4:
                grids.append(torch.cat([x[0:1], gt, src_dec, pix] + outs + [ceil]))
        print(obj, [(r["az"], round(r["lpips_tokwarp_a0"], 3), round(r["lpips_pixwarp_a0"], 3), round(r["lpips_ceiling"], 3))
                    for r in rows if r["obj"] == obj], flush=True)
    save_image(torch.cat(grids), osp.join(args.out, "token_warp_grid.png"), nrow=8)
    agg = {}
    for az in TARGETS:
        rs = [r for r in rows if r["az"] == az]
        agg[az] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k not in ("obj", "az")}
    json.dump({"by_az": agg, "rows": rows}, open(osp.join(args.out, "metrics.json"), "w"), indent=1)
    print("BY_AZ  (LPIPS: tokwarp a0/a30/a60 | pixwarp a0 | source | ceiling ; disocc frac)")
    for az, a in agg.items():
        print(f"  {az:>3}: {a['lpips_tokwarp_a0']:.3f}/{a['lpips_tokwarp_a30']:.3f}/{a['lpips_tokwarp_a60']:.3f} | "
              f"{a['lpips_pixwarp_a0']:.3f} | {a['lpips_source']:.3f} | {a['lpips_ceiling']:.3f} ; {a['frac_disoccluded']:.2f}")
    print("P4_DONE")


if __name__ == "__main__":
    main()
