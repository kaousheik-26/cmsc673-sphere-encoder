import argparse
import json
import logging
import math
import time
from dataclasses import asdict, replace

import torch

import sphere.rng as rng
from halfsphere.backbone import SE2Backbone
from halfsphere.data import balanced_subset, image_dataset, load_batch
from halfsphere.geometry import angle_deg
from halfsphere.masks import half_mask, prox_mask, token_index_for_half
from halfsphere.metrics import HalfLPIPS, psnr_masked
from halfsphere.records import Run, save_grid
from halfsphere.repaint import RepaintConfig, repaint_complete
from halfsphere.ridgemap import DecoderOps, cca_with_null, completion_modes, contraction_radius, lm_refine, render_mode, slq_rank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Ridge Map local linear algebra experiments")
parser.add_argument("cmd", choices=["h2", "h3", "modes", "lm", "h1", "check"])
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-flowers-256px")
parser.add_argument("--n_images", type=int, default=16)
parser.add_argument("--subset_seed", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split", type=str, default="lr")
parser.add_argument("--class_policy", type=str, default="true", choices=["true", "null"])
parser.add_argument("--repaint_config", type=str, default="{}", help="JSON overrides of RepaintConfig for the solutions analysed")
parser.add_argument("--jvp_mode", type=str, default="dvjp", choices=["dvjp", "fwd", "fd"])
parser.add_argument("--eps", type=str, default="0.01,0.001")
parser.add_argument("--probes", type=int, default=8)
parser.add_argument("--lanczos_k", type=int, default=80)
parser.add_argument("--alphas", type=str, default="10,20,30,40,50,60,70,80")
parser.add_argument("--fixed_point_steps", type=int, default=30)
parser.add_argument("--power_iters", type=int, default=20)
parser.add_argument("--gamma", type=float, default=10.0)
parser.add_argument("--n_modes", type=int, default=4)
parser.add_argument("--mode_degrees", type=str, default="-6,-3,0,3,6")
parser.add_argument("--lm_steps", type=int, default=10)
parser.add_argument("--lm_mu", type=float, default=1e-2)
parser.add_argument("--lm_max_deg", type=float, default=3.0)
parser.add_argument("--h1_split", type=str, default="train+val", help="images whose latents are cached for CCA")
parser.add_argument("--k_pca", type=int, default=512)
parser.add_argument("--n_perm", type=int, default=5)
args = parser.parse_args()


def gen(*tags):
    return torch.Generator().manual_seed(rng.fold_in(args.seed, *tags) % (2**63))


def setup():
    dev = "cuda"
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ds = image_dataset(bb.cfg["dataset_name"], bb.image_size, args.dev_dir)
    idx = balanced_subset(ds, args.n_images, seed=args.subset_seed)
    run = Run(f"ridgemap_{args.cmd}", {"args": vars(args)}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path,
              seeds={"base": args.seed, "subset": args.subset_seed})
    return bb, ds, idx, run, dev


def repaint_solution(bb, gt, m, c, tag):
    cfg = replace(RepaintConfig(split=args.split), **json.loads(args.repaint_config))
    with torch.no_grad():
        x = repaint_complete(bb, m * gt, m, cfg, c, generator=gen("s2", tag))
    return x, cfg


def main():
    bb, ds, idx, run, dev = setup()
    lp = HalfLPIPS(dev) if args.cmd in ("lm", "modes") else None
    H = bb.image_size
    out_summary = []
    if args.cmd == "h1":
        return h1(bb, ds, run, dev)
    if args.cmd == "check":
        return check(bb, ds, idx, run, dev)
    for n, i in enumerate(idx):
        gt, c, ids = load_batch(ds, [i], dev)
        c = c if args.class_policy == "true" else None
        m = half_mask(1, H, H, args.split, device=dev)
        y = m * gt
        t0 = time.time()
        bb.reset_counters()
        with torch.no_grad():
            v_true = bb.encode(gt, c)
        if args.cmd == "h2":
            eps = [float(e) for e in args.eps.split(",")]
            res = {}
            for where, v in (("true", v_true),):
                ops = DecoderOps(bb, v, m, 0.0, c, args.jvp_mode)
                r = slq_rank(ops, eps_rel=eps, probes=args.probes, k=args.lanczos_k, generator=gen("h2", i))
                res[where] = {"lam_max": r["lam_max"], **{str(e): r[e] for e in eps}}
            row = {"image_id": ids[0], "where": res}
        elif args.cmd == "h3":
            alphas = [float(a) for a in args.alphas.split(",")]
            res = []
            for a in alphas:
                cfg = RepaintConfig(K=args.fixed_point_steps, alpha_start=a, alpha_end=a, schedule="constant",
                               final_clean=False, split=args.split, share_noise=True)
                g_ = gen("h3", i, a)
                with torch.no_grad():
                    x, traj = repaint_complete(bb, y, m, cfg, c, generator=g_, return_traj=True)
                xs = traj["x"]
                step = float(((1 - m) * (xs[-1] - xs[-2])).norm() / ((1 - m) * xs[-2]).norm().clamp_min(1e-12))
                rho, trace = contraction_radius(bb, xs[-1], y, m, a, traj["e_hat"], iters=args.power_iters, c=c, generator=gen("h3p", i, a))
                res.append({"alpha": a, "rho": rho, "trace": trace, "last_step_rel": step,
                            "psnr_B": float(psnr_masked(x, gt, 1 - m)[0])})
                logger.info(f"  img {n} alpha {a:4.1f}: rho {rho:.4f}  last step {step:.2e}")
            row = {"image_id": ids[0], "h3": res}
        elif args.cmd == "modes":
            x_repaint, cfg = repaint_solution(bb, gt, m, c, i)
            with torch.no_grad():
                v_repaint = bb.encode(x_repaint, c)
            res = {}
            for where, v in (("repaint", v_repaint), ("true", v_true)):
                ops = DecoderOps(bb, v, m, 0.0, c, args.jvp_mode)
                th, modes, leak, resid = completion_modes(ops, gamma=args.gamma, k=args.lanczos_k, n_modes=args.n_modes, generator=gen("modes", i))
                degs = [float(d) for d in args.mode_degrees.split(",")]
                strips = [render_mode(bb, v, md, degrees=degs, c=c) for md in modes]
                grid = torch.stack(strips, 0)
                save_grid([grid[:, j] for j in range(grid.shape[1])], run.path(f"modes_{n:02d}_{where}.png"))
                obs_change = [float(((m * (s - s[degs.index(0.0)][None])).abs()).mean()) for s in strips]
                miss_change = [float((((1 - m) * (s - s[degs.index(0.0)][None])).abs()).mean()) for s in strips]
                res[where] = {"eig": th.tolist(), "leak": leak.tolist(), "ritz_resid": resid.tolist(),
                              "obs_change_mean_abs": obs_change, "miss_change_mean_abs": miss_change}
                logger.info(f"  img {n} {where}: eig {[round(t, 3) for t in th.tolist()]} leak {[round(t, 3) for t in leak.tolist()]} resid {[round(t, 3) for t in resid.tolist()]}")
            row = {"image_id": ids[0], "repaint_config": asdict(cfg), "modes": res}
        elif args.cmd == "lm":
            x_repaint, cfg = repaint_solution(bb, gt, m, c, i)
            res = {}
            for where, v0 in (("repaint", None), ("true", v_true)):
                with torch.no_grad():
                    v0 = bb.encode(x_repaint, c) if v0 is None else v0
                    x0 = bb.decode(v0, 0.0, c)
                v1, hist = lm_refine(bb, v0, y, m, steps=args.lm_steps, mu=args.lm_mu, c=c, jvp_mode=args.jvp_mode, max_deg=args.lm_max_deg)
                with torch.no_grad():
                    x1 = bb.decode(v1, 0.0, c)
                res[where] = {
                    "resid_A_before": hist[0], "resid_A_after": hist[-1], "hist": hist,
                    "psnr_A_before": float(psnr_masked(x0.clamp(-1, 1), gt, m)[0]), "psnr_A_after": float(psnr_masked(x1.clamp(-1, 1), gt, m)[0]),
                    "lpips_B_before": float(lp(prox_mask(x0, y, m, 0).clamp(-1, 1), gt)[0]), "lpips_B_after": float(lp(prox_mask(x1, y, m, 0).clamp(-1, 1), gt)[0]),
                    "moved_deg": float(angle_deg(v0, v1)[0]),
                }
                if n < 6:
                    save_grid([gt, x0, x1, prox_mask(x1, y, m, 0)], run.path(f"lm_{n:02d}_{where}.png"))
                logger.info(f"  img {n} {where}: resid_A {hist[0]:.4f} -> {hist[-1]:.4f}  moved {res[where]['moved_deg']:.2f} deg")
            row = {"image_id": ids[0], "repaint_config": asdict(cfg), "lm": res}
        row.update(wall_s=round(time.time() - t0, 1), passes=dict(bb.counters))
        run.row(**row)
        out_summary.append(row)
        logger.info(f"[{n + 1}/{len(idx)}] {ids[0]} {row['wall_s']}s")
    run.finish("ok", n_images=len(idx))
    logger.info(f"done: {run.dir}")


def check(bb, ds, idx, run, dev):
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    from halfsphere.ridgemap import lanczos

    gt, c, ids = load_batch(ds, idx[:1], dev)
    c = c if args.class_policy == "true" else None
    m = half_mask(1, bb.image_size, bb.image_size, args.split, device=dev)
    with torch.no_grad():
        v = bb.encode(gt, c)
    ops = DecoderOps(bb, v, m, 0.0, c, args.jvp_mode)
    shape = v.shape
    n_calls = [0]

    def mv(x):
        n_calls[0] += 1
        t = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(shape)).to(dev)
        return ops.GA(ops.P(t)).double().cpu().numpy().reshape(-1)

    t0 = time.time()
    A = LinearOperator((bb.L, bb.L), matvec=mv, dtype=np.float64)
    ev_scipy = np.sort(eigsh(A, k=10, which="LA", tol=1e-4, return_eigenvectors=False))[::-1]
    t_scipy, calls = time.time() - t0, n_calls[0]
    t0 = time.time()
    th, _, _ = lanczos(ops.GA, ops.P(torch.randn(shape, generator=gen("check")).to(dev)), k=80)
    ev_ours = np.sort(th.double().cpu().numpy())[::-1][:10]
    t_ours = time.time() - t0
    rel = np.abs(ev_ours - ev_scipy) / np.abs(ev_scipy)
    logger.info(f"top-10 eig scipy {np.round(ev_scipy, 3).tolist()}")
    logger.info(f"top-10 eig ours  {np.round(ev_ours, 3).tolist()}  max rel err {rel.max():.2e}")
    eps = [float(e) for e in args.eps.split(",")]
    slq = {}
    for k in (80, 200):
        r = slq_rank(ops, eps_rel=eps, probes=4, k=k, lam_max=float(ev_scipy[0]), generator=gen("check_slq"))
        slq[k] = {str(e): r[e] for e in eps}
        logger.info(f"SLQ k={k}: {slq[k]}")
    res = {"image_id": ids[0], "eig_scipy": ev_scipy.tolist(), "eig_ours": ev_ours.tolist(), "max_rel_err": float(rel.max()),
           "scipy_s": round(t_scipy, 1), "scipy_matvecs": calls, "ours_s": round(t_ours, 1), "slq": slq}
    json.dump(res, open(run.path("check.json"), "w"), indent=2, default=float)
    run.finish("ok")


def h1(bb, ds, run, dev):
    dsets = [image_dataset(bb.cfg["dataset_name"], bb.image_size, args.dev_dir, split=s) for s in args.h1_split.split("+")]
    n_total = sum(len(d) for d in dsets)
    iA, iB = token_index_for_half(16 * 16, 16 * 16, 16, args.split)
    VA = torch.empty(n_total, len(iA) * bb.latent_shape[1], dtype=torch.float16)
    VB = torch.empty(n_total, len(iB) * bb.latent_shape[1], dtype=torch.float16)
    k = 0
    t0 = time.time()
    for d in dsets:
        loader = torch.utils.data.DataLoader(d, batch_size=128, num_workers=8)
        for xb, _, _ in loader:
            with torch.no_grad():
                v = bb.encode(xb.to(dev))
            VA[k : k + len(xb)] = v[:, iA].flatten(1).half().cpu()
            VB[k : k + len(xb)] = v[:, iB].flatten(1).half().cpu()
            k += len(xb)
    logger.info(f"cached {k} latents in {time.time() - t0:.0f}s")
    torch.save({"iA": iA, "iB": iB, "n": k}, run.path("h1_meta.pt"))

    def pca(X, q):
        X = X.float().to(dev)
        X = X - X.mean(0)
        torch.manual_seed(rng.fold_in(args.seed, "h1_pca") % (2**63))
        _, _, V = torch.pca_lowrank(X, q=q, center=False, niter=4)
        return (X @ V[:, :q]).double().cpu()

    A = pca(VA, args.k_pca)
    Bm = pca(VB, args.k_pca)
    real, null_m, null_s = cca_with_null(A, Bm, k_pca=args.k_pca, n_perm=args.n_perm, seed=args.seed)
    c = args.k_pca / k
    res = {"n": k, "k_pca": args.k_pca, "wachter_edge": 2 * math.sqrt(c * (1 - c)),
           "real": real.tolist(), "null_mean": null_m.tolist(), "null_sd": null_s.tolist(),
           "z_top20": [float((r - mu) / max(s, 1e-12)) for r, mu, s in zip(real[:20], null_m[:20], null_s[:20])]}
    json.dump(res, open(run.path("h1_cca.json"), "w"), indent=2)
    logger.info(f"H1: top CC {res['real'][:5]} null {res['null_mean'][:5]} wachter {res['wachter_edge']:.3f}")
    run.finish("ok")


if __name__ == "__main__":
    main()
