"""
R1 on the evaluation protocols.

gso30 : SyncDreamer GSO30 (Liu et al.): input = their 000.png (az 0, elev 30),
        15 targets at 22.5 deg steps (alternating sides, autoregressive), their
        camera-16 poses, f = 280 px @ 256. Metrics vs THEIR GT renders on the 15
        novel views: full white-background image (SyncDreamer's protocol) and
        alpha-masked. Bars: Zero123 18.93/0.166, SyncDreamer 20.05/0.146.
co3d  : real CO3D teddybear dev frames (se3d/data/prep_co3d_dev.py): source +
        5 real target frames with per-frame intrinsics; MoGe-2 depth scaled to the
        CO3D world by LSQ against CO3D's MVS depth on the source.
sv3d21: our GSO-300 orbits (SEVA split ids), P=1: input = SEVA's train id.
Output: <out>/metrics.json (+ per-object grids).
"""
import argparse
import dataclasses
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from se3d.eval.metrics import all_metrics
from se3d.geometry.cameras import syncdreamer_K
from se3d.geometry.depth import align_scale, align_scale_shift
from se3d.gs.gt_ceiling import load_views
from se3d.routes.r1_warp_fill import R1, R1Config, orbit_unitbox_scale

DATA = "/fs/nexus-scratch/vinayakg/se3d/data"


def _rgba(path):
    a = np.array(Image.open(path)).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])
    return torch.tensor(rgb, device="cuda"), torch.tensor(a[..., 3], device="cuda")


def alt_order(n):
    """1, n-1, 2, n-2, ... (alternating sides around index 0)"""
    out = []
    for k in range(1, n // 2 + 1):
        out.append(k)
        if n - k != k:
            out.append(n - k)
    return out


def run_gso30(r1, objects, out):
    cams = np.load(f"{DATA}/syncdreamer_meta/camera-16.npz")
    K = torch.tensor(syncdreamer_K(256), dtype=torch.float32, device="cuda")
    W2C = []
    for i in range(16):
        w = np.eye(4)
        w[:3] = cams["poses"][i]
        W2C.append(torch.tensor(w, dtype=torch.float32, device="cuda"))
    per = {}
    for obj in objects:
        gts = [_rgba(f"{DATA}/gso30_syncdreamer/renders/{obj}/{i:03d}.png") for i in range(16)]
        x0, a0 = gts[0]
        order = alt_order(16)
        targets = [(i, W2C[i]) for i in order]
        t0 = time.time()
        views, inter, info, P = r1.run_cams(x0, a0, K, W2C[0], targets,
                                            scale_fn=lambda d, m: orbit_unitbox_scale(d, m, K, W2C[0], 1.5), src_key=0)
        sec = time.time() - t0
        by_idx = {int(v["az"]): v for v in views}
        pred = torch.stack([by_idx[i]["rgb"].permute(2, 0, 1) for i in range(1, 16)])
        gt = torch.stack([gts[i][0].permute(2, 0, 1) for i in range(1, 16)])
        msk = torch.stack([gts[i][1] for i in range(1, 16)])
        m = all_metrics(pred, gt, msk)
        per[obj] = {**m, "seconds": sec}
        save_image(torch.cat([torch.stack([gts[i][0].permute(2, 0, 1) for i in range(16)]),
                              torch.stack([by_idx[i]["rgb"].permute(2, 0, 1) for i in range(16)])]),
                   osp.join(out, f"{obj}.png"), nrow=16)
        print(obj, {k: round(v, 3) for k, v in per[obj].items()}, flush=True)
    return per


def run_co3d(r1, seqs, out):
    per = {}
    for seq in seqs:
        v = load_views(f"{DATA}/co3d_dev/{seq}")
        meta = json.load(open(f"{DATA}/co3d_dev/{seq}/cameras.json"))
        Ks = [torch.tensor(fr["K"], dtype=torch.float32, device="cuda") for fr in meta["frames"]]
        targets = [(j, v["w2c"][j], Ks[j]) for j in range(1, len(Ks))]
        d_ref = v["depth"][0]
        # scale-only fit (median ratio) to CO3D's MVS depth: the cameras live in CO3D units.
        # No MVS depth on the source (some sequences): use the point all camera axes look at
        # (least-squares ray intersection) as the orbit centre and the q90 limb rule.
        if (d_ref > 0).sum() > 100:
            scale_fn = lambda d, m: align_scale(d, d_ref, m & (d_ref > 0))
        else:
            W = v["w2c"].cpu().double()
            cs = torch.stack([-w[:3, :3].T @ w[:3, 3] for w in W])
            ds = torch.stack([w[2, :3] for w in W])
            A = torch.zeros(3, 3, dtype=torch.float64)
            b = torch.zeros(3, dtype=torch.float64)
            for c_, d_ in zip(cs, ds):
                Pm = torch.eye(3, dtype=torch.float64) - torch.outer(d_, d_)
                A += Pm
                b += Pm @ c_
            centre = torch.linalg.solve(A, b)
            R_orbit = float((cs[0] - centre).norm())
            scale_fn = lambda d, m: R_orbit / torch.quantile(d[m], 0.90).item()
            print(f"{seq}: no MVS depth on source; orbit radius from camera axes = {R_orbit:.3f}", flush=True)
        t0 = time.time()
        views, inter, info, P = r1.run_cams(v["rgb"][0], v["alpha"][0], Ks[0], v["w2c"][0], targets, scale_fn, src_key=0)
        sec = time.time() - t0
        by = {int(x["az"]): x for x in views}
        idx = list(range(1, len(Ks)))
        pred = torch.stack([by[j]["rgb"].permute(2, 0, 1) for j in idx])
        gt = v["rgb"][idx].permute(0, 3, 1, 2)
        m = all_metrics(pred, gt, v["alpha"][idx])
        m["per_view_lpips"] = [all_metrics(pred[k:k + 1], gt[k:k + 1])["lpips_vgg"] for k in range(len(idx))]
        m["angles"] = [meta["frames"][j]["azimuth"] for j in idx]
        per[seq] = {**m, "seconds": sec}
        save_image(torch.cat([v["rgb"].permute(0, 3, 1, 2), torch.stack([by[j]["rgb"].permute(2, 0, 1) for j in range(len(Ks))])]),
                   osp.join(out, f"{seq}.png"), nrow=len(Ks))
        print(seq, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in per[seq].items() if k != "per_view_lpips"}, flush=True)
    return per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=["gso30", "co3d"], required=True)
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--out", required=True)
    for f in dataclasses.fields(R1Config):
        if f.type in (int, float, str):
            p.add_argument(f"--{f.name}", type=f.type, default=None)
        elif f.type is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            p.add_argument(f"--{f.name}", type=float, nargs="+", default=None)
    args = p.parse_args()
    cfg = R1Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(R1Config) if getattr(args, f.name) is not None})
    os.makedirs(args.out, exist_ok=True)
    r1 = R1(cfg)
    if args.protocol == "gso30":
        objs = args.objects or sorted(os.listdir(f"{DATA}/gso30_syncdreamer/renders"))
        per = run_gso30(r1, objs, args.out)
    else:
        objs = args.objects or sorted(os.listdir(f"{DATA}/co3d_dev"))
        per = run_co3d(r1, objs, args.out)
    keys = [k for k, v in next(iter(per.values())).items() if isinstance(v, float)]
    summ = {k: float(np.mean([per[o][k] for o in per])) for k in keys}
    json.dump({"protocol": args.protocol, "config": dataclasses.asdict(cfg), "summary": summ, "per_object": per},
              open(osp.join(args.out, "metrics.json"), "w"), indent=1)
    print("SUMMARY", json.dumps({k: round(v, 4) for k, v in summ.items()}))
    print("PROTOCOL_DONE")


if __name__ == "__main__":
    main()
