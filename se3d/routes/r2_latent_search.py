"""
Route R2: latent search. Given a source view and a GT target view of the same
object, optimise only the latent so the frozen SE2 decoder outputs the target:

    z = F(u),  L = LPIPS_vgg(D(z, y, a_dec), x_tgt) + L1 + mu * angle(z, z0)^2

inits: z0 = F(E(x_src)) | z_gt = F(E(x_tgt)) (control) | token-warped z0 (P4) |
random F(e). Optimiser: Adam on u (default) or Riemannian GD (tangent projection
+ exp-map retraction) as a check. Early stopping on LPIPS-alex (not in the loss).
Logged every 50 steps: PSNR / LPIPS to x_tgt, angle(z, z_gt), angle(z, z0),
per-token norm stats. Every optimum is re-decoded at a in {10, 20, 40} with the
object's shared e (A6). z* is saved (fp16) for the A1-A6 analyses.

Pairs come from a render protocol folder: (source frame, target frame) lists.
"""
import argparse
import json
import math
import os
import os.path as osp
import time

import numpy as np
import torch
from torchvision.utils import save_image

from se3d.eval.metrics import _lpips_net, lpips, psnr
from se3d.gs.gt_ceiling import load_views
from se3d.se2.sphere_ops import angle_deg, exp_map, project_tangent, token_norms
from se3d.se2.wrapper import SE2

DATA = "/fs/nexus-scratch/vinayakg/se3d/data/gso_renders"


def token_warp_init(se2, v, i_src, i_tgt, z_src, e):
    """P4-style oracle-depth token warp of z_src into the target view."""
    from se3d.probes.p4_token_warp import G, P, token_source_index

    # token_source_index works with source index 0; build a 2-view dict
    sub = {k: (val[[i_src, i_tgt]] if torch.is_tensor(val) and val.ndim >= 3 else val) for k, val in v.items() if k != "frames"}
    idx = token_source_index(sub, 1)
    c = torch.arange(G, device="cuda") * P + P // 2
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    src_bg = sub["alpha"][0][yy.flatten(), xx.flatten()] < 0.5
    bg_tok = z_src[0][src_bg].mean(0) if src_bg.any() else z_src[0].mean(0)
    zw = torch.where((idx >= 0)[:, None], z_src[0][idx.clamp_min(0)], bg_tok[None])
    zw = torch.where((idx == -2)[:, None], e[0], zw)
    return se2.spherify(zw[None])


def search(se2, x_tgt, u0, z0, z_gt, y, steps, lr, mu, a_dec, optimizer="adam", log_every=50):
    lp_vgg = _lpips_net("vgg", "cuda")
    lp_alex = _lpips_net("alex", "cuda")
    B = x_tgt.shape[0]
    u = u0.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr) if optimizer == "adam" else None
    best = {"lpips_alex": torch.full((B,), 9.0, device="cuda"), "z": se2.spherify(u0.detach()).clone(),
            "step": torch.zeros(B, dtype=torch.long, device="cuda")}
    log = []
    for s in range(steps + 1):
        z = se2.spherify(u)
        xh = se2.decode(z, y, a_dec) * 0.5 + 0.5
        l_lp = lp_vgg(xh * 2 - 1, x_tgt * 2 - 1).flatten()
        l_l1 = (xh - x_tgt).abs().mean((1, 2, 3))
        # squared geodesic distance, smooth form: 2 (1 - cos) ~= theta^2 (acos has an
        # infinite gradient at theta = 0, which made the mu > 0 runs NaN at step 0)
        if mu > 0:
            cosz = (z * z0).flatten(1).sum(1) / (z.flatten(1).norm(dim=1) * z0.flatten(1).norm(dim=1))
            reg = 2 * (1 - cosz)
        else:
            reg = torch.zeros_like(l_l1)
        loss = (l_lp + l_l1 + mu * reg).sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {s}")
        if s % log_every == 0 or s == steps:
            with torch.no_grad():
                xc = xh.clamp(0, 1)
                la = lp_alex(xc * 2 - 1, x_tgt * 2 - 1).flatten()
                better = la < best["lpips_alex"]
                best["lpips_alex"] = torch.where(better, la, best["lpips_alex"])
                best["z"][better] = z.detach()[better]
                best["step"][better] = s
                tn = token_norms(z)
                log.append({"step": s, "psnr": psnr(xc, x_tgt).tolist(), "lpips_vgg": l_lp.tolist(),
                            "lpips_alex": la.tolist(), "ang_gt": angle_deg(z, z_gt).tolist(),
                            "ang_z0": angle_deg(z, z0).tolist(), "tok_norm_std": tn.std(1).tolist(),
                            "tok_norm_max": tn.max(1).values.tolist()})
        if s == steps:
            break
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:  # Riemannian GD on the sphere of radius sqrt(d)
            g, = torch.autograd.grad(loss, u)
            with torch.no_grad():
                rg = project_tangent(u, g)
                # step length in arc units: lr * radius * normalised gradient
                rn = rg.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
                u.copy_(exp_map(u, -lr * se2.radius * rg / rn * 0.05))
    return best, log


def build_pairs(protocol, objects, deltas):
    """(obj, i_src, i_tgt, delta) from a render folder; source = frame 0."""
    pairs = []
    for obj in objects:
        meta = json.load(open(osp.join(DATA, protocol, obj, "cameras.json")))
        azs = [((fr["azimuth"] + 180) % 360) - 180 for fr in meta["frames"]]
        for d in deltas:
            for j, a in enumerate(azs):
                if abs(a - d) < 1e-3 and j != 0:
                    pairs.append((obj, 0, j, d))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="large-fdlite-256")
    p.add_argument("--decoder_ckpt", default=None)
    p.add_argument("--protocol", default="dev_orbit")
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=[15, 30, 45, 60, 90])
    p.add_argument("--inits", nargs="+", default=["z0", "zgt", "warp", "random"])
    p.add_argument("--mus", type=float, nargs="+", default=[0.0])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--a_dec", type=float, default=0.0)
    p.add_argument("--optimizer", default="adam", choices=["adam", "riemannian"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(osp.join(args.out, "zstar"), exist_ok=True)
    se2 = SE2(args.model, decoder_ckpt=args.decoder_ckpt)
    objs = args.objects or sorted(os.listdir(osp.join(DATA, args.protocol)))
    pairs = build_pairs(args.protocol, objs, args.deltas)
    print(f"{len(pairs)} pairs x inits {args.inits} x mu {args.mus}", flush=True)
    views = {obj: load_views(osp.join(DATA, args.protocol, obj)) for obj in objs}
    E = {obj: se2.sample_noise(1, torch.Generator(device="cuda").manual_seed(args.seed * 1000 + i))
         for i, obj in enumerate(objs)}
    results = []
    t0 = time.time()
    for init in args.inits:
        for mu in args.mus:
            for b in range(0, len(pairs), args.batch):
                chunk = pairs[b:b + args.batch]
                xs = torch.stack([views[o]["rgb"][i].permute(2, 0, 1) for o, i, j, d in chunk])
                xt = torch.stack([views[o]["rgb"][j].permute(2, 0, 1) for o, i, j, d in chunk])
                e = torch.cat([E[o] for o, i, j, d in chunk])
                with torch.no_grad():
                    z0 = se2.encode(xs * 2 - 1)
                    zgt = se2.encode(xt * 2 - 1)
                    if init == "z0":
                        u0 = z0
                    elif init == "zgt":
                        u0 = zgt
                    elif init == "random":
                        u0 = e
                    elif init == "warp":
                        u0 = torch.cat([token_warp_init(se2, views[o], i, j, z0[k:k + 1], e[k:k + 1])
                                        for k, (o, i, j, d) in enumerate(chunk)])
                best, log = search(se2, xt, u0, z0, zgt, None, args.steps, args.lr, mu, args.a_dec, args.optimizer)
                zs = best["z"]
                with torch.no_grad():
                    x_star = se2.to01(se2.decode(zs, None, args.a_dec))
                    x_ceil = se2.to01(se2.decode(zgt, None, 0.0))
                    redec = {a: se2.to01(se2.decode(se2.rotate(zs, e, a), None, a)) for a in (10, 20, 40)}
                    m_star_l, m_star_p = lpips(x_star, xt), psnr(x_star, xt)
                    m_ceil_l, m_ceil_p = lpips(x_ceil, xt), psnr(x_ceil, xt)
                    m_init_l = lpips(se2.to01(se2.decode(se2.spherify(u0), None, 0.0)), xt)
                    m_red = {a: (lpips(v, xt), psnr(v, xt)) for a, v in redec.items()}
                    ang_gt, ang_z0 = angle_deg(zs, zgt), angle_deg(zs, z0)
                    ang_z0_gt = angle_deg(z0, zgt)
                    ang_ED = angle_deg(se2.encode(x_star * 2 - 1), zs)  # does E map D(z*) back to z*?
                for k, (o, i, j, d) in enumerate(chunk):
                    tag = f"{o}_d{int(d)}_{init}_mu{mu:g}"
                    torch.save(zs[k].half().cpu(), osp.join(args.out, "zstar", tag + ".pt"))
                    results.append({"obj": o, "delta": d, "init": init, "mu": mu, "best_step": int(best["step"][k]),
                                    "lpips": m_star_l[k].item(), "psnr": m_star_p[k].item(),
                                    "lpips_init": m_init_l[k].item(), "lpips_ceiling": m_ceil_l[k].item(),
                                    "psnr_ceiling": m_ceil_p[k].item(),
                                    **{f"lpips_redec_a{a}": m_red[a][0][k].item() for a in m_red},
                                    **{f"psnr_redec_a{a}": m_red[a][1][k].item() for a in m_red},
                                    "ang_zstar_zgt": ang_gt[k].item(), "ang_zstar_z0": ang_z0[k].item(),
                                    "ang_z0_zgt": ang_z0_gt[k].item(), "ang_ED_zstar": ang_ED[k].item(),
                                    "curve": [{key: (val[k] if isinstance(val, list) else val) for key, val in L.items()} for L in log]})
                if b == 0:
                    save_image(torch.cat([xs[:6], xt[:6], x_ceil[:6], x_star[:6], redec[20][:6]]),
                               osp.join(args.out, f"grid_{init}_mu{mu:g}.png"), nrow=min(6, len(chunk)))
                done = len(results)
                print(f"[{init} mu={mu:g}] {done} runs  {time.time() - t0:.0f}s  "
                      f"last-batch LPIPS z*={m_star_l.mean():.3f} ceil={m_ceil_l.mean():.3f} init={m_init_l.mean():.3f} "
                      f"ang(z*,zgt)={ang_gt.mean():.1f} ang(z0,zgt)={ang_z0_gt.mean():.1f}", flush=True)
    json.dump({"args": vars(args), "results": results}, open(osp.join(args.out, "results.json"), "w"))
    # summary table
    summ = {}
    for r in results:
        key = f"{r['init']}|mu{r['mu']:g}|d{int(r['delta'])}"
        summ.setdefault(key, []).append(r)
    rows = {}
    for key, rs in sorted(summ.items()):
        rows[key] = {m: float(np.mean([r[m] for r in rs])) for m in
                     ["lpips", "psnr", "lpips_init", "lpips_ceiling", "lpips_redec_a10", "lpips_redec_a20", "lpips_redec_a40",
                      "ang_zstar_zgt", "ang_zstar_z0", "ang_z0_zgt", "ang_ED_zstar", "best_step"]}
        rows[key]["n"] = len(rs)
    json.dump(rows, open(osp.join(args.out, "summary.json"), "w"), indent=1)
    for key, v in rows.items():
        print(f"{key:28s} LPIPS {v['lpips']:.3f} (init {v['lpips_init']:.3f}, ceil {v['lpips_ceiling']:.3f}, "
              f"a20 {v['lpips_redec_a20']:.3f}) | ang z*-zgt {v['ang_zstar_zgt']:.1f} z*-z0 {v['ang_zstar_z0']:.1f} "
              f"z0-zgt {v['ang_z0_zgt']:.1f} E(D(z*))-z* {v['ang_ED_zstar']:.1f}")
    print("R2_DONE")


if __name__ == "__main__":
    main()
