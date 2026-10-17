import argparse
import json
import logging
import time
from dataclasses import asdict, replace

import torch

import sphere.rng as rng
from cli_utils import str2bool
from halfsphere.backbone import SE2Backbone
from halfsphere.data import balanced_subset, image_dataset, load_batch
from halfsphere.geometry import token_manifold_gap
from halfsphere.masks import half_mask, prox_mask
from halfsphere.metrics import HalfLPIPS, InceptionFeatures, diversity, kid, latent_angle, psnr_masked, real_features, seam_score
from halfsphere.records import Run, config_hash, save_grid
from halfsphere.steer import SteerConfig, steer_optimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Steer Riemannian latent optimisation experiments")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-flowers-256px")
parser.add_argument("--n_images", type=int, default=32)
parser.add_argument("--subset_seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split", type=str, default="lr")
parser.add_argument("--class_policy", type=str, default="true", choices=["true", "null"])
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--configs", type=str, default="[{}]", help="JSON list of SteerConfig overrides")
parser.add_argument("--kid", type=str2bool, default=True)
parser.add_argument("--save_images", type=int, default=8)
args = parser.parse_args()


def main():
    dev = "cuda"
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    ds = image_dataset(bb.cfg["dataset_name"], bb.image_size, args.dev_dir)
    idx = balanced_subset(ds, args.n_images, seed=args.subset_seed)
    cfgs = [replace(SteerConfig(split=args.split), **o) for o in json.loads(args.configs)]
    run = Run("steer", {"args": vars(args), "configs": [asdict(c) for c in cfgs]}, dev_dir=args.dev_dir,
              ckpt_path=bb.ckpt_path, seeds={"base": args.seed, "subset": args.subset_seed})
    lp = HalfLPIPS(dev)
    incep = InceptionFeatures(dev) if args.kid else None
    X, C, IDS = load_batch(ds, idx, "cpu")
    B = X.shape[0]
    feats_real = None
    if incep is not None:
        feats_real = real_features(ds, incep, bb.image_size)

    summary = {}
    for cfg in cfgs:
        ch = config_hash(cfg)
        bb.reset_counters()
        t0 = time.time()
        finals, all_restarts = [], []
        for s in range(0, B, args.batch_size):
            gt, c = X[s : s + args.batch_size].to(dev), C[s : s + args.batch_size].to(dev)
            c = c if args.class_policy == "true" else None
            m = half_mask(gt.shape[0], bb.image_size, bb.image_size, args.split, device=dev)
            y = m * gt
            lpips_fn = (lambda a, b: lp.loss(a, b, args.split, "A")) if cfg.beta_lpips > 0 else None
            Xr, Vr, Lr, curves = steer_optimize(bb, y, m, cfg, c, lpips_fn=lpips_fn, seed=rng.fold_in(args.seed, "s1", s) % (2**63))
            best = Lr.argmin(0)
            ar = torch.arange(gt.shape[0], device=dev)
            x_best, v_best = Xr[best, ar], Vr[best, ar]
            with torch.no_grad():
                v_true = bb.encode(gt, c)
            x_pin = prox_mask(x_best, y, m, 0.0).clamp(-1, 1)
            gap = token_manifold_gap(v_best)
            met = {
                "psnr_A_prepin": psnr_masked(x_best, gt, m),
                "psnr_B": psnr_masked(x_pin, gt, 1 - m),
                "lpips_B": lp(x_pin, gt, args.split, "B"),
                "lpips_A_unpinned": lp(x_best, gt, args.split, "A"),
                "lpips_full": lp.full(x_pin, gt),
                "seam": seam_score(x_pin, args.split),
                "seam_unpinned": seam_score(x_best, args.split),
                "lat_angle_true": latent_angle(v_best, v_true),
                "data_loss": Lr[best, ar],
                "tok_dist_frac": gap["dist_frac"],
            }
            if cfg.n_restarts > 1:
                met["diversity_lpips_B"] = diversity(torch.stack([prox_mask(Xr[r], y, m, 0).clamp(-1, 1) for r in range(cfg.n_restarts)]), args.split, lp)
            for j in range(gt.shape[0]):
                run.row(image_id=IDS[s + j], method="steer", config_hash=ch, config=asdict(cfg), seed=args.seed,
                        class_policy=args.class_policy, split=args.split, best_restart=int(best[j]),
                        metrics={k: float(v[j]) for k, v in met.items()},
                        loss_curve_best=[float(cv[t]) for cv in [curves[int(best[j])][:, j]] for t in range(0, cv.shape[0], 10)])
            finals.append(x_pin.cpu())
            all_restarts.append(Xr.cpu())
        Xf = torch.cat(finals)
        info = {"wall_s": round(time.time() - t0, 1), "passes": dict(bb.counters),
                "per_image_dec_fwd": bb.counters["dec_fwd"] / B, "per_image_dec_bwd": bb.counters["dec_bwd"] / B}
        if incep is not None:
            info["kid"] = kid(incep(Xf), feats_real, subset_size=min(50, B), n_subsets=100)
        summary[ch] = info
        save_grid([X[: args.save_images]] + [torch.cat(all_restarts, 1)[r, : args.save_images] for r in range(cfg.n_restarts)] + [Xf[: args.save_images]],
                  run.path(f"steer_{ch}.png"))
        logger.info(f"{ch} {asdict(cfg)} -> {info}")
    json.dump(summary, open(run.path("summary.json"), "w"), indent=2)
    run.finish("ok")
    logger.info(f"done: {run.dir}")


if __name__ == "__main__":
    main()
