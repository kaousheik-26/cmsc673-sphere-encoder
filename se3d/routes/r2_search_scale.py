"""
Latent search at scale (many objects, few fixed camera steps each), for pattern analysis.
Objects: G-Objaverse folders (views 0..23 = azimuth ring every
15 deg at one elevation; view 0 = source). Targets: view indices (default +15/+45/+90 and
-15/-45/-90 deg). Search: from z0 = F(E(x_src)), mu pull to z0 (see r2_latent_search.search).
Saves <out>/part{i}.pt: objs, tgt_views, z0 [O,T,C] fp16, zstar / zgt [O,V,T,C] fp16,
w2c [O,1+V,4,4], K [O,1+V,3,3], lpips_zstar / lpips_ceiling [O,V] (decode at a = 0).
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch

from se3d.eval.metrics import lpips
from se3d.routes.r2_latent_search import search
from se3d.data.images import load_rgba_white
from se3d.se2.wrapper import SE2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="file with one object folder per line")
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--n_parts", type=int, required=True)
    p.add_argument("--targets", type=int, nargs="+", default=[1, 3, 6, 23, 21, 18])
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dst = osp.join(args.out, f"part{args.part}.pt")
    if osp.exists(dst):
        print("exists", dst, "R2S_DONE")
        return
    allo = [l.strip() for l in open(args.list) if l.strip()]
    objs = allo[args.part::args.n_parts]
    keep, W, K = [], [], []
    for o in objs:
        if not osp.exists(osp.join(o, "cameras.json")):  # incomplete download
            continue
        m = json.load(open(osp.join(o, "cameras.json")))
        fr = m["frames"]
        if len(fr) <= max(args.targets):
            continue
        idx = [0] + args.targets
        keep.append(o)
        W.append(np.stack([np.array(fr[i]["w2c"], dtype=np.float64) for i in idx]))
        K.append(np.stack([np.array(fr[i].get("K", m.get("K")), dtype=np.float64) for i in idx]))
    objs = keep
    print(f"part {args.part}: {len(objs)} objects x {len(args.targets)} targets, {args.steps} steps", flush=True)
    se2 = SE2("large-fdlite-256")
    V = len(args.targets)
    pairs = [(oi, vi) for oi in range(len(objs)) for vi in range(V)]
    T_, C_ = se2.latent_shape
    Z0 = torch.zeros(len(objs), T_, C_, dtype=torch.float16)
    ZS = torch.zeros(len(objs), V, T_, C_, dtype=torch.float16)
    ZG = torch.zeros(len(objs), V, T_, C_, dtype=torch.float16)
    LS = torch.zeros(len(objs), V)
    LC = torch.zeros(len(objs), V)
    img = lambda o, k: torch.tensor(load_rgba_white(osp.join(o, json.load(open(osp.join(o, "cameras.json")))["frames"][k]["file"]))[0]).permute(2, 0, 1)
    t0 = time.time()
    for b in range(0, len(pairs), args.batch):
        ch = pairs[b:b + args.batch]
        xs = torch.stack([img(objs[oi], 0) for oi, vi in ch]).cuda()
        xt = torch.stack([img(objs[oi], args.targets[vi]) for oi, vi in ch]).cuda()
        with torch.no_grad():
            z0 = se2.encode(xs * 2 - 1)
            zg = se2.encode(xt * 2 - 1)
        best, _ = search(se2, xt, z0, z0, zg, None, args.steps, 1e-2, args.mu, 0.0, log_every=50)
        with torch.no_grad():
            ls = lpips(se2.to01(se2.decode(best["z"], None, 0.0)), xt).flatten()
            lc = lpips(se2.to01(se2.decode(zg, None, 0.0)), xt).flatten()
        for k, (oi, vi) in enumerate(ch):
            Z0[oi] = z0[k].half().cpu()
            ZS[oi, vi] = best["z"][k].half().cpu()
            ZG[oi, vi] = zg[k].half().cpu()
            LS[oi, vi], LC[oi, vi] = ls[k].item(), lc[k].item()
        if (b // args.batch) % 10 == 0:
            done = b + len(ch)
            print(f"{done}/{len(pairs)} pairs {time.time() - t0:.0f}s  LPIPS z* {ls.mean():.3f} ceiling {lc.mean():.3f}", flush=True)
    torch.save({"objs": objs, "tgt_views": args.targets, "z0": Z0, "zstar": ZS, "zgt": ZG,
                "w2c": torch.tensor(np.stack(W)), "K": torch.tensor(np.stack(K)),
                "lpips_zstar": LS, "lpips_ceiling": LC, "steps": args.steps, "mu": args.mu}, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"saved {dst}  mean LPIPS z* {LS.mean():.3f} ceiling {LC.mean():.3f}  R2S_DONE", flush=True)


if __name__ == "__main__":
    main()
