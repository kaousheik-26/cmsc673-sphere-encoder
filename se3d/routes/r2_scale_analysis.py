"""
Pattern analysis of the large latent search (r2_search_scale.py parts; G-Objaverse azimuth steps
+15/+45/+90/-15/-45/-90 deg from view 0). Same questions as r2_pattern.py, at ~10K objects:
  size of the move vs camera step; shared direction across objects (coherence, pairwise cos,
  token-mean part); +d vs -d; composition (15 -> 45, 45 -> 90); low rank (PCA);
  learnable on UNSEEN objects with linear tools fitted on many objects
  (mean move, ridge on PCA of z0), decode LPIPS vs the true view.
Outputs <out>/scale_pattern.json.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F

from se3d.data.images import load_rgba_white
from se3d.eval.metrics import lpips
from se3d.routes.r2_delta_scale import load_parts
from se3d.se2.wrapper import SE2


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def ang(a, b):
    return torch.rad2deg(torch.acos(F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default="/fs/nexus-projects/genwildsplat/vinayakg/se3d/r2_scale10k")
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_stat", type=int, default=2000, help="objects used for coherence / PCA statistics")
    p.add_argument("--pca_k", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    c = load_parts(args.parts)
    O, steps = len(c["objs"]), c["tgt_views"]
    deg = {1: 15, 3: 45, 6: 90, 23: -15, 21: -45, 18: -90}
    D = [deg.get(v, v) for v in steps]
    print(f"{O} objects, camera steps {D}", flush=True)
    se2 = SE2("large-fdlite-256")
    T, C = se2.latent_shape
    out = {"objects": O, "steps_deg": D}
    # size of the move (all objects, chunked)
    A_s, A_g = [], []
    for i in range(0, O, 500):
        z0 = c["z0"][i:i + 500].float().cuda().flatten(1)
        zs = c["zstar"][i:i + 500].float().cuda().flatten(2)
        zg = c["zgt"][i:i + 500].float().cuda().flatten(2)
        A_s.append(ang(zs, z0[:, None]).cpu())
        A_g.append(ang(zg, z0[:, None]).cpu())
    A_s, A_g = torch.cat(A_s), torch.cat(A_g)
    out["angle"] = {d: {"zstar_z0": A_s[:, k].mean().item(), "zgt_z0": A_g[:, k].mean().item(),
                        "lpips_zstar": c["lpips_zstar"][:, k].mean().item(), "lpips_ceiling": c["lpips_ceiling"][:, k].mean().item()}
                    for k, d in enumerate(D)}
    print("angle", json.dumps(out["angle"]), flush=True)
    # structure on a subset
    S = torch.randperm(O, generator=torch.Generator().manual_seed(0))[:args.n_stat]
    z0 = c["z0"][S].float().cuda().flatten(1)
    for name in ("zstar", "zgt"):
        dl = c[name][S].float().cuda().flatten(2) - z0[:, None]  # [n,V,D]
        res = {}
        for k, d in enumerate(D):
            u = unit(dl[:, k])
            g = unit(dl[:, k].view(len(S), T, C).mean(1))
            G = u @ u.T
            off = G[~torch.eye(len(S), dtype=torch.bool, device=G.device)]
            res[d] = {"coherence": u.mean(0).norm().item(), "pair_cos": off.mean().item(),
                      "coherence_tokenmean": g.mean(0).norm().item()}
        for a_, b_ in ((15, -15), (45, -45), (90, -90)):
            res[f"cos({a_},{b_})"] = F.cosine_similarity(dl[:, D.index(a_)], dl[:, D.index(b_)], dim=-1).mean().item()
        for a_, b_ in ((15, 45), (45, 90), (-15, -45), (-45, -90)):
            res[f"cos({a_},{b_})"] = F.cosine_similarity(dl[:, D.index(a_)], dl[:, D.index(b_)], dim=-1).mean().item()
            res[f"size({b_})/size({a_})"] = (dl[:, D.index(b_)].norm(dim=-1) / dl[:, D.index(a_)].norm(dim=-1)).mean().item()
        M = dl.flatten(0, 1)
        M = M - M.mean(0)
        ev = torch.linalg.eigvalsh(M @ M.T).flip(0).clamp_min(0)
        ev = (ev.cumsum(0) / ev.sum()).cpu().numpy()
        res["pca_explained"] = {k: float(ev[k - 1]) for k in (1, 10, 100, 1000) if k <= len(ev)}
        res["random_coherence"] = 1 / np.sqrt(len(S))
        out[name + "_minus_z0"] = res
        print(name, json.dumps(res)[:1500], flush=True)
        del dl, M
    # learnable on unseen objects with linear tools fitted on all training objects
    tr = torch.arange(O - args.n_test)
    ev_o = torch.arange(O - args.n_test, O - args.n_test + args.n_eval)
    Z0 = c["z0"].float().flatten(1)  # CPU
    mu0 = Z0[tr].mean(0)
    sub = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(1))[:4000]]
    _, _, Vh = torch.linalg.svd((Z0[sub] - mu0).cuda(), full_matrices=False)
    P = Vh[:args.pca_k].T  # [D,k]
    X = lambda idx: torch.cat([(Z0[idx].cuda() - mu0.cuda()) @ P, torch.ones(len(idx), 1, device="cuda")], 1)
    frames = {}

    def image(o, k):
        d = c["objs"][o]
        if d not in frames:
            frames[d] = json.load(open(osp.join(d, "cameras.json")))["frames"]
        return torch.tensor(load_rgba_white(osp.join(d, frames[d][k]["file"]))[0]).permute(2, 0, 1)

    @torch.no_grad()
    def dec_lp(z, gt):
        return torch.cat([lpips(se2.to01(se2.decode(se2.spherify(z[q:q + 12].view(-1, T, C)), None, 0.0)), gt[q:q + 12]).flatten()
                          for q in range(0, len(z), 12)]).mean().item()

    learn = {}
    Xte = X(ev_o)
    for k, d in enumerate(D):
        gt = torch.stack([image(int(o), steps[k]) for o in ev_o]).cuda()
        z0e = Z0[ev_o].cuda()
        r = {"identity": dec_lp(z0e, gt), "oracle_zstar": dec_lp(c["zstar"][ev_o, k].float().cuda().flatten(1), gt),
             "ceiling_zgt": dec_lp(c["zgt"][ev_o, k].float().cuda().flatten(1), gt)}
        for name in ("zstar", "zgt"):
            # accumulate X^T X and X^T Y over all training objects in chunks
            XtX = torch.zeros(args.pca_k + 1, args.pca_k + 1, device="cuda")
            XtY = torch.zeros(args.pca_k + 1, T * C, device="cuda")
            msum = torch.zeros(T * C, device="cuda")
            for i in range(0, len(tr), 1000):
                idx = tr[i:i + 1000]
                Xi = X(idx)
                Yi = c[name][idx, k].float().cuda().flatten(1) - Z0[idx].cuda()
                XtX += Xi.T @ Xi
                XtY += Xi.T @ Yi
                msum += Yi.sum(0)
            r[f"mean_move_{name}"] = dec_lp(z0e + msum / len(tr), gt)
            for lam in (1e1, 1e3, 1e5):
                Wr = torch.linalg.solve(XtX + lam * torch.eye(len(XtX), device="cuda"), XtY)
                r[f"ridge_{name}_l{lam:g}"] = dec_lp(z0e + Xte @ Wr, gt)
        learn[d] = r
        print("learn", d, {a_: round(b_, 4) for a_, b_ in r.items()}, flush=True)
    out["learnable_unseen"] = learn
    json.dump(out, open(osp.join(args.out, "scale_pattern.json"), "w"), indent=1)
    print("R2_SCALE_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
