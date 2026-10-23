"""
R2 analyses A1-A6 over a finished r2_latent_search run (results.json + zstar/).

A1 fidelity vs ceiling: LPIPS(z*) vs D(F(E(x_tgt))) per init; failure rate
   (failure = z* worse than its own init, or > ceiling + 0.1).
A2 landing point: angle(z*, z_gt) vs angle(z0, z_gt) vs random (~90 deg).
A3 token transport: for co-visible target tokens (GT depth), is the source token
   that a target token "comes from" recoverable by per-token cosine NN from
   z0 into z*, and into z_gt? recall@1-token; split co-visible / disoccluded
   (mean cosine of the best source match).
A5 angle dependence of A2/A3 (by delta).
A6 re-decoding at a in {10, 20, 40} (taken from results.json).
A4 (linearity / ridge map) is in r2_linear.py (needs >= 50 objects at fixed delta).
"""
import argparse
import json
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F

from se3d.gs.gt_ceiling import load_views
from se3d.probes.p4_token_warp import G, P, token_source_index

DATA = "/fs/nexus-scratch/vinayakg/se3d/data/gso_renders"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--protocol", default="dev_orbit")
    args = p.parse_args()
    R = json.load(open(osp.join(args.run, "results.json")))["results"]
    out = {}

    # A1 / A2 / A6 by init x mu x delta
    groups = {}
    for r in R:
        groups.setdefault((r["init"], r["mu"]), []).append(r)
    a1 = {}
    for (init, mu), rs in sorted(groups.items()):
        fail = [r["lpips"] > r["lpips_init"] + 1e-3 or r["lpips"] > r["lpips_ceiling"] + 0.1 for r in rs]
        a1[f"{init}|mu{mu:g}"] = {
            "n": len(rs), "lpips": float(np.mean([r["lpips"] for r in rs])),
            "lpips_ceiling": float(np.mean([r["lpips_ceiling"] for r in rs])),
            "frac_beats_ceiling": float(np.mean([r["lpips"] < r["lpips_ceiling"] for r in rs])),
            "failure_rate": float(np.mean(fail)),
            "ang_zstar_zgt": float(np.mean([r["ang_zstar_zgt"] for r in rs])),
            "ang_z0_zgt": float(np.mean([r["ang_z0_zgt"] for r in rs])),
            "ang_zstar_z0": float(np.mean([r["ang_zstar_z0"] for r in rs])),
            **{f"lpips_redec_a{a}": float(np.mean([r[f"lpips_redec_a{a}"] for r in rs])) for a in (10, 20, 40)},
        }
    out["A1_A2_A6"] = a1

    # A3 token transport (init z0, mu 0 only)
    a3 = {}
    views = {}
    for r in R:
        if r["init"] != "z0" or r["mu"] != 0:
            continue
        o, d = r["obj"], r["delta"]
        if o not in views:
            views[o] = load_views(osp.join(DATA, args.protocol, o))
        v = views[o]
        azs = [((fr["azimuth"] + 180) % 360) - 180 for fr in v["frames"]]
        j = [k for k, a in enumerate(azs) if abs(a - d) < 1e-3][0]
        sub = {k: (val[[0, j]] if torch.is_tensor(val) and val.ndim >= 3 else val) for k, val in v.items() if k != "frames"}
        idx = token_source_index(sub, 1)  # per target token: source token / -1 bg / -2 disoccluded
        zs = torch.load(osp.join(args.run, "zstar", f"{o}_d{int(d)}_z0_mu0.pt")).float().cuda()
        from se3d.se2.wrapper import SE2  # lazy: only for encoding
        if "se2" not in globals():
            globals()["se2"] = SE2("large-fdlite-256")
        with torch.no_grad():
            z0 = se2.encode(sub["rgb"][0:1].permute(0, 3, 1, 2) * 2 - 1)[0]
            zgt = se2.encode(sub["rgb"][1:2].permute(0, 3, 1, 2) * 2 - 1)[0]
        res = {}
        for name, zt in [("zstar", zs), ("zgt", zgt)]:
            sim = F.normalize(zt, dim=-1) @ F.normalize(z0, dim=-1).T  # [tgt, src]
            best, nn = sim.max(1)
            cov = idx >= 0
            if cov.any():
                dy = (nn[cov] // G - idx[cov] // G).abs()
                dx = (nn[cov] % G - idx[cov] % G).abs()
                res[f"{name}_recall1"] = float((torch.maximum(dx, dy) <= 1).float().mean())
                res[f"{name}_cos_covis"] = float(best[cov].mean())
            dis = idx == -2
            res[f"{name}_cos_disocc"] = float(best[dis].mean()) if dis.any() else float("nan")
            # per-token angle between target-latent token and its true source token
            if cov.any():
                res[f"{name}_tok_cos_to_true_src"] = float(F.cosine_similarity(zt[cov], z0[idx[cov]], dim=-1).mean())
        res["tok_cos_zstar_zgt_covis"] = float(F.cosine_similarity(zs[idx >= 0], zgt[idx >= 0], dim=-1).mean()) if (idx >= 0).any() else float("nan")
        res["tok_cos_zstar_zgt_disocc"] = float(F.cosine_similarity(zs[idx == -2], zgt[idx == -2], dim=-1).mean()) if (idx == -2).any() else float("nan")
        a3.setdefault(int(d), []).append(res)
    out["A3_A5_by_delta"] = {d: {k: float(np.nanmean([x[k] for x in rs])) for k in rs[0]} for d, rs in sorted(a3.items())}
    json.dump(out, open(osp.join(args.run, "analysis.json"), "w"), indent=1)
    print("A1/A2/A6")
    for k, v in a1.items():
        print(f"  {k:14s} n={v['n']:3d} LPIPS {v['lpips']:.3f} (ceil {v['lpips_ceiling']:.3f}, beats ceil {v['frac_beats_ceiling']:.2f}, "
              f"fail {v['failure_rate']:.2f}) ang z*-zgt {v['ang_zstar_zgt']:.1f} vs z0-zgt {v['ang_z0_zgt']:.1f} | "
              f"redec a10/20/40 {v['lpips_redec_a10']:.3f}/{v['lpips_redec_a20']:.3f}/{v['lpips_redec_a40']:.3f}")
    print("A3/A5 (z0 init, mu 0)")
    for d, v in out["A3_A5_by_delta"].items():
        print(f"  d={d:>3}: recall1 z* {v.get('zstar_recall1', float('nan')):.2f} zgt {v.get('zgt_recall1', float('nan')):.2f} | "
              f"tok-cos(z*,zgt) covis {v['tok_cos_zstar_zgt_covis']:.3f} disocc {v['tok_cos_zstar_zgt_disocc']:.3f}")
    print("R2_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
