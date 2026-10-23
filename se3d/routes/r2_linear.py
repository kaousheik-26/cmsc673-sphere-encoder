"""
R2-A4: is a fixed camera change linear (or low-rank) in SE2 latent space?

Input: r2_latent_search runs on the GSO-300 orbits (protocol sv3d21, init z0)
at d = +1 step (17.14 deg) and +2 steps (34.29 deg). For each object:
z0 = F(E(x_src)), z* (optimised target latent), z_gt = F(E(x_tgt)).
Deltas live in the tangent space of z0: v = log_{z0}(z*) (and v_gt = log_{z0}(z_gt)).

- PCA of v over objects (explained variance of the top components).
- decomposition: fraction of |v|^2 in the token-mean (global, 384-d) component.
- ridge map z0 -> v (dual / kernel form, linear kernel), train on 80% objects,
  test on 20%: decode exp_{z0}(v_pred) at a = 0 / 20 and score LPIPS vs the GT
  target; baselines: identity (z0), mean delta, token-mean delta, oracle z*, and
  the ceiling D(F(E(x_tgt))).
- composition: (+1 map)^2 vs the +2 map vs GT at +2.
Targets: v (from z*) and v_gt (from E(x_tgt)) are both reported.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch

from se3d.eval.metrics import lpips, psnr
from se3d.gs.gt_ceiling import load_views
from se3d.se2.sphere_ops import exp_map, log_map, project_tangent
from se3d.se2.wrapper import SE2

DATA = "/fs/nexus-scratch/vinayakg/se3d/data/gso_renders"
STEP = 360.0 / 21


def kernel_ridge(Xtr, Ytr, Xte, lam):
    K = Xtr @ Xtr.T
    A = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(K.shape[0], device=K.device, dtype=K.dtype), Ytr)
    return (Xte @ Xtr.T) @ A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run1", required=True, help="R2 run dir at +1 step")
    p.add_argument("--run2", required=True, help="R2 run dir at +2 steps")
    p.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    se2 = SE2("large-fdlite-256")
    D1 = [r for r in json.load(open(osp.join(args.run1, "results.json")))["results"] if r["init"] == "z0" and r["mu"] == 0]
    D2 = {r["obj"]: r for r in json.load(open(osp.join(args.run2, "results.json")))["results"] if r["init"] == "z0" and r["mu"] == 0}
    objs = sorted(set(r["obj"] for r in D1) & set(D2))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(objs))
    n_te = max(5, int(len(objs) * args.test_frac))
    te = set(objs[i] for i in perm[:n_te])

    def load(run, r):
        return torch.load(osp.join(run, "zstar", f"{r['obj']}_d{int(r['delta'])}_z0_mu0.pt")).float().cuda()

    Z0, Zs1, Zg1, Zs2, Zg2, X1, X2 = [], [], [], [], [], [], []
    for o in objs:
        r1 = [r for r in D1 if r["obj"] == o][0]
        v = load_views(osp.join(DATA, "sv3d21", o), [0, 1, 2])
        x = v["rgb"].permute(0, 3, 1, 2)
        with torch.no_grad():
            z = se2.encode(x * 2 - 1)
        Z0.append(z[0]); Zg1.append(z[1]); Zg2.append(z[2])
        Zs1.append(load(args.run1, r1)); Zs2.append(load(args.run2, D2[o]))
        X1.append(x[1]); X2.append(x[2])
    Z0, Zs1, Zg1, Zs2, Zg2 = map(torch.stack, (Z0, Zs1, Zg1, Zs2, Zg2))
    X1, X2 = torch.stack(X1), torch.stack(X2)
    is_te = torch.tensor([o in te for o in objs], device="cuda")
    res = {"n_objects": len(objs), "n_test": int(is_te.sum())}

    for tgt_name, Zt in [("zstar", Zs1), ("zgt", Zg1)]:
        V = log_map(Z0, Zt)  # tangent deltas [n, 256, 384]
        n = V.shape[0]
        Vf = V.flatten(1)
        # PCA
        Vc = Vf - Vf.mean(0, keepdim=True)
        S = torch.linalg.svdvals(Vc.double())
        ev = (S ** 2) / (S ** 2).sum()
        res[f"{tgt_name}_pca_top1"] = float(ev[0])
        res[f"{tgt_name}_pca_top5"] = float(ev[:5].sum())
        res[f"{tgt_name}_pca_top10"] = float(ev[:10].sum())
        # token-mean component
        tm = V.mean(1, keepdim=True).expand_as(V)
        res[f"{tgt_name}_frac_token_mean"] = float((tm.flatten(1).norm(dim=1) ** 2 / Vf.norm(dim=1) ** 2).mean())
        res[f"{tgt_name}_mean_delta_angle_deg"] = float(torch.rad2deg(Vf.norm(dim=1) / se2.radius).mean())

        # ridge map z0 -> v, tested on held-out objects
        tr_i, te_i = (~is_te).nonzero().flatten(), is_te.nonzero().flatten()
        preds = {"identity": Z0[te_i]}
        mean_v = V[tr_i].mean(0, keepdim=True)
        preds["mean_delta"] = exp_map(Z0[te_i], project_tangent(Z0[te_i], mean_v.expand(len(te_i), -1, -1)))
        tmv = V[tr_i].mean((0, 1), keepdim=True).expand(len(te_i), V.shape[1], V.shape[2])
        preds["token_mean_delta"] = exp_map(Z0[te_i], project_tangent(Z0[te_i], tmv))
        for lam in args.lams:
            vp = kernel_ridge(Z0[tr_i].flatten(1).double(), V[tr_i].flatten(1).double(), Z0[te_i].flatten(1).double(), lam)
            vp = project_tangent(Z0[te_i], vp.float().view(-1, *V.shape[1:]))
            preds[f"ridge_lam{lam:g}"] = exp_map(Z0[te_i], vp)
        preds["oracle"] = Zt[te_i]
        with torch.no_grad():
            ceil = se2.to01(se2.decode(Zg1[te_i], None, 0.0))
            e = se2.sample_noise(len(te_i), torch.Generator(device="cuda").manual_seed(0))
            for name, zp in preds.items():
                for a in (0, 20):
                    xh = se2.to01(se2.decode(se2.rotate(zp, e, a) if a else zp, None, a))
                    res[f"{tgt_name}|{name}|a{a}_lpips"] = lpips(xh, X1[te_i]).mean().item()
                    res[f"{tgt_name}|{name}|a{a}_psnr"] = psnr(xh, X1[te_i]).mean().item()
            res[f"{tgt_name}|ceiling_lpips"] = lpips(ceil, X1[te_i]).mean().item()

        # composition: (+1)^2 vs +2 (ridge with the best lambda on +1)
        if tgt_name == "zstar":
            best = min(args.lams, key=lambda l: res[f"zstar|ridge_lam{l:g}|a0_lpips"])
            V2 = log_map(Z0, Zs2)
            v1 = kernel_ridge(Z0[tr_i].flatten(1).double(), V[tr_i].flatten(1).double(), Z0[te_i].flatten(1).double(), best)
            z1 = exp_map(Z0[te_i], project_tangent(Z0[te_i], v1.float().view(-1, *V.shape[1:])))
            v11 = kernel_ridge(Z0[tr_i].flatten(1).double(), V[tr_i].flatten(1).double(), z1.flatten(1).double(), best)
            z11 = exp_map(z1, project_tangent(z1, v11.float().view(-1, *V.shape[1:])))
            v2 = kernel_ridge(Z0[tr_i].flatten(1).double(), V2[tr_i].flatten(1).double(), Z0[te_i].flatten(1).double(), best)
            z2 = exp_map(Z0[te_i], project_tangent(Z0[te_i], v2.float().view(-1, *V.shape[1:])))
            with torch.no_grad():
                for name, zp in [("compose_1x2", z11), ("direct_2", z2), ("identity_2", Z0[te_i]), ("oracle_2", Zs2[te_i])]:
                    xh = se2.to01(se2.decode(zp, None, 0.0))
                    res[f"composition|{name}_lpips"] = lpips(xh, X2[te_i]).mean().item()
            res["composition|best_lam"] = best

    json.dump(res, open(osp.join(args.out, "a4_linear.json"), "w"), indent=1)
    for k, v in res.items():
        print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")
    print("A4_DONE")


if __name__ == "__main__":
    main()
