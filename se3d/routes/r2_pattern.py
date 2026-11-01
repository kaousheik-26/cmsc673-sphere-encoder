"""
Is there a shared, learnable pattern in the latent moves that show a new camera?
Input: R2 searches on GSO-300 orbits (source = frame 0, targets at several azimuth
steps, init z0, mu pull to z0): <runs>/part*/{results.json, zstar/*.pt}.
For every object o and camera step d: z0 = F(E(x_src)), z* (searched), z_gt = F(E(x_tgt)).
Deltas: dS = z* - z0 (the nudge), dG = z_gt - z0 (the encoder's own change).
  P1 size: angle(z*, z0) and angle(z_gt, z0) vs |d|
  P2 shared direction across objects: |mean of unit deltas| (1 = same for all, ~1/sqrt(N) = random)
     and mean pairwise cosine; also for the token-mean (global, 384-d) part
  P3 +d vs -d of the same object: cosine
  P4 2d vs d of the same object: cosine and size ratio
  P5 low rank: variance explained by the top-k PCs of all deltas
  P6 where: share of |delta|^2 on object tokens (alpha > 0.5 on the 16x16 grid) vs token share
  P7 learnable on UNSEEN objects (objects split 160 train / 40 test, fixed seed):
     decode LPIPS vs the target view for: identity z0, z0 + mean train delta(d),
     ridge z0 -> delta (PCA of z0), oracle z*, ceiling z_gt
  Picture: z0 + t * mean_delta(d) on unseen objects for growing t (does it turn the object?)
Outputs: <out>/pattern.json, <out>/*.png
"""
import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from se3d.eval.metrics import lpips
from se3d.gs.gt_ceiling import DATA, load_views
from se3d.se2.wrapper import SE2


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def ang(a, b):
    return torch.rad2deg(torch.acos(F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/fs/nexus-scratch/vinayakg/se3d/results/r2_scale200")
    p.add_argument("--n_test", type=int, default=40)
    p.add_argument("--pca_k", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(osp.join(args.runs, "part*", "results.json"))):
        for r in json.load(open(f))["results"]:
            r["file"] = osp.join(osp.dirname(f), "zstar", f"{r['obj']}_d{int(r['delta'])}_{r['init']}_mu{r['mu']:g}.pt")
            rows.append(r)
    objs = sorted({r["obj"] for r in rows})
    deltas = sorted({int(r["delta"]) for r in rows})
    have = {(r["obj"], int(r["delta"])): r for r in rows}
    objs = [o for o in objs if all((o, d) in have for d in deltas)]
    print(f"{len(objs)} objects with all {len(deltas)} camera steps {deltas}", flush=True)
    se2 = SE2("large-fdlite-256")
    O, Dn = len(objs), len(deltas)
    Z0, ZS, ZG, XT, X0, M, WC, KC = [], [], [], [], [], [], [], []
    for o in objs:
        v = load_views(osp.join(DATA, "sv3d21", o))
        meta = json.load(open(osp.join(DATA, "sv3d21", o, "cameras.json")))
        azs = [int(((fr["azimuth"] + 180) % 360) - 180) for fr in v["frames"]]
        idx = [min(range(len(azs)), key=lambda j: abs(azs[j] - d)) for d in deltas]
        fr = meta["frames"]
        WC.append(np.stack([np.array(fr[j]["w2c"], dtype=np.float64) for j in [0] + idx]))
        KC.append(np.stack([np.array(fr[j].get("K", meta.get("K")), dtype=np.float64) for j in [0] + idx]))
        with torch.no_grad():
            x = v["rgb"].permute(0, 3, 1, 2)
            z = se2.encode(x * 2 - 1)
        Z0.append(z[0].flatten().half())
        ZG.append(z[idx].flatten(1).half())
        ZS.append(torch.stack([torch.load(have[(o, d)]["file"]).cuda().flatten() for d in deltas]))
        XT.append(x[idx].half())
        X0.append(x[0].half())
        M.append(F.adaptive_avg_pool2d(v["alpha"][0][None, None].float(), 16).flatten() > 0.5)
    Z0, ZS, ZG = torch.stack(Z0).float(), torch.stack(ZS).float(), torch.stack(ZG).float()  # [O,D], [O,Dn,D]
    XT, X0, M = torch.stack(XT), torch.stack(X0), torch.stack(M)
    T, C = se2.latent_shape
    torch.save({"objs": objs, "deltas": deltas, "z0": Z0.half().cpu(), "zstar": ZS.half().cpu(), "zgt": ZG.half().cpu(),
                "x_tgt": XT.cpu(), "x_src": X0.cpu(), "w2c": torch.tensor(np.stack(WC)), "K": torch.tensor(np.stack(KC))},
               osp.join(args.out, "latents.pt"))
    dS, dG = ZS - Z0[:, None], ZG - Z0[:, None]
    out = {"objects": O, "deltas": deltas}

    def stats(dl, name):
        res = {}
        for k, d in enumerate(deltas):
            u = unit(dl[:, k])
            g = unit(dl[:, k].view(O, T, C).mean(1))
            cos = u @ u.T
            off = cos[~torch.eye(O, dtype=torch.bool, device=cos.device)]
            res[d] = {"coherence": u.mean(0).norm().item(), "pair_cos": off.mean().item(),
                      "coherence_tokenmean": g.mean(0).norm().item(),
                      "tokenmean_share": (dl[:, k].view(O, T, C).mean(1).norm(dim=-1) ** 2 * T / dl[:, k].norm(dim=-1) ** 2).mean().item(),
                      "obj_energy_share": ((dl[:, k].view(O, T, C).norm(dim=-1) ** 2 * M).sum(1) / (dl[:, k].view(O, T, C).norm(dim=-1) ** 2).sum(1)).mean().item(),
                      "obj_token_share": M.float().mean(1).mean().item()}
        for d in deltas:
            if -d in deltas and d > 0:
                kp, km = deltas.index(d), deltas.index(-d)
                res[d]["cos_plus_minus"] = F.cosine_similarity(dl[:, kp], dl[:, km], dim=-1).mean().item()
        # P4 composition on the +17/+34 and -17/-34 steps (2 steps vs 1 step)
        comp = {}
        for a, b in ((17, 34), (-17, -34)):
            if a in deltas and b in deltas:
                ka, kb = deltas.index(a), deltas.index(b)
                comp[f"{a}->{b}"] = {"cos": F.cosine_similarity(dl[:, ka], dl[:, kb], dim=-1).mean().item(),
                                     "size_ratio": (dl[:, kb].norm(dim=-1) / dl[:, ka].norm(dim=-1)).mean().item()}
        # P5 low rank over all deltas
        A = dl.flatten(0, 1)
        A = A - A.mean(0)
        sv = torch.linalg.eigvalsh(A @ A.T).flip(0).clamp_min(0)  # squared singular values
        ev = (sv.cumsum(0) / sv.sum()).cpu().numpy()
        out[name] = {"by_delta": res, "composition": comp,
                     "pca_explained": {k: float(ev[k - 1]) for k in (1, 5, 10, 50, 100) if k <= len(ev)},
                     "random_coherence": 1 / np.sqrt(O)}
        print(name, json.dumps(out[name], indent=None)[:1500], flush=True)

    # P1 size
    out["angle"] = {d: {"zstar_z0": ang(ZS[:, k], Z0).mean().item(), "zstar_z0_sd": ang(ZS[:, k], Z0).std().item(),
                        "zgt_z0": ang(ZG[:, k], Z0).mean().item(), "zgt_z0_sd": ang(ZG[:, k], Z0).std().item(),
                        "lpips_zstar": float(np.mean([have[(o, d)]["lpips"] for o in objs])),
                        "lpips_ceiling": float(np.mean([have[(o, d)]["lpips_ceiling"] for o in objs]))}
                    for k, d in enumerate(deltas)}
    print("angle", json.dumps(out["angle"]), flush=True)
    stats(dS, "zstar_minus_z0")
    stats(dG, "zgt_minus_z0")

    # P7 learnability on unseen objects
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(O, generator=g)
    te, tr = perm[:args.n_test].cuda(), perm[args.n_test:].cuda()

    @torch.no_grad()
    def dec_lpips(z, x):
        vals = []
        for i in range(0, len(z), 16):
            zz = se2.spherify(z[i:i + 16].view(-1, T, C))
            vals.append(lpips(se2.to01(se2.decode(zz, None, 0.0)), x[i:i + 16].float()))
        return torch.cat(vals).flatten()

    # PCA of z0 on train objects for the ridge input
    mu0 = Z0[tr].mean(0)
    U, S, Vh = torch.linalg.svd(Z0[tr] - mu0, full_matrices=False)
    P = Vh[:args.pca_k].T  # [D,k]
    Xtr, Xte = (Z0[tr] - mu0) @ P, (Z0[te] - mu0) @ P
    Xtr1 = torch.cat([Xtr, torch.ones(len(tr), 1, device=Xtr.device)], 1)
    Xte1 = torch.cat([Xte, torch.ones(len(te), 1, device=Xte.device)], 1)
    p7 = {}
    for k, d in enumerate(deltas):
        xt = XT[te, k]
        res = {"identity": dec_lpips(Z0[te], xt).mean().item(),
               "oracle_zstar": dec_lpips(ZS[te, k], xt).mean().item(),
               "ceiling_zgt": dec_lpips(ZG[te, k], xt).mean().item()}
        for nm, dl in (("zstar", dS), ("zgt", dG)):
            m = dl[tr, k].mean(0)
            res[f"mean_delta_{nm}"] = dec_lpips(Z0[te] + m, xt).mean().item()
            best = None
            for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):  # pick lambda by 4-fold CV on train objects
                err = 0.0
                for f_ in range(4):
                    vi = torch.arange(len(tr), device=Xtr.device) % 4 == f_
                    A, B = Xtr1[~vi], dl[tr, k][~vi]
                    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=A.device), A.T @ B)
                    err += ((Xtr1[vi] @ W - dl[tr, k][vi]) ** 2).sum().item()
                if best is None or err < best[1]:
                    best = (lam, err)
            W = torch.linalg.solve(Xtr1.T @ Xtr1 + best[0] * torch.eye(Xtr1.shape[1], device=Xtr1.device), Xtr1.T @ dl[tr, k])
            res[f"ridge_{nm}"] = dec_lpips(Z0[te] + Xte1 @ W, xt).mean().item()
            res[f"ridge_{nm}_lambda"] = best[0]
        p7[d] = res
        print("P7", d, {a: round(b, 4) for a, b in res.items()}, flush=True)
    out["learnable_unseen"] = p7

    # picture: shared mean delta applied with growing strength to 4 unseen objects
    rows_img = []
    with torch.no_grad():
        for k, d in enumerate(deltas):
            if d not in (34, -34, 86, -86):
                continue
            m = dS[tr, k].mean(0)
            for i in te[:4]:
                imgs = [X0[i].float()]
                for t in (0, 1, 4, 16):
                    zz = se2.spherify((Z0[i] + t * m).view(1, T, C))
                    imgs.append(se2.to01(se2.decode(zz, None, 0.0))[0])
                imgs.append(XT[i, k].float())
                rows_img += imgs
    save_image(torch.stack(rows_img), osp.join(args.out, "mean_delta_strength.png"), nrow=6)
    json.dump(out, open(osp.join(args.out, "pattern.json"), "w"), indent=1)
    print("R2_PATTERN_DONE")


if __name__ == "__main__":
    main()
