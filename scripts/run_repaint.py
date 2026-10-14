import argparse
import itertools
import json
import logging
import time
from dataclasses import asdict, replace

import torch

from cli_utils import str2bool
from halfsphere.backbone import SE2Backbone
from halfsphere.data import balanced_subset, image_dataset, load_batch
from halfsphere.masks import fill_missing, half_mask, prox_mask
from halfsphere.metrics import HalfLPIPS, InceptionFeatures, kid, real_features, latent_angle, psnr_masked, seam_score
from halfsphere.records import Run, config_hash, save_grid
from halfsphere.masks import token_index_for_half
from halfsphere.repaint import RepaintConfig, repaint_complete, repaint_token_complete

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Repaint projection splitting experiments")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-flowers-256px")
parser.add_argument("--stage", type=str, default="coarse", choices=["coarse", "refine", "samples", "traj", "tok"])
parser.add_argument("--n_images", type=int, default=64)
parser.add_argument("--subset_seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=0, help="noise seed; per image and draw via sphere.rng.fold_in")
parser.add_argument("--split", type=str, default="lr", choices=["lr", "tb"])
parser.add_argument("--observed", type=str, default="first", choices=["first", "second"])
parser.add_argument("--class_policy", type=str, default="true", choices=["true", "null"])
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--configs", type=str, default="", help="JSON list of RepaintConfig overrides (refine, samples, traj stages)")
parser.add_argument("--n_samples", type=int, default=8)
parser.add_argument("--kid", type=str2bool, default=True)
parser.add_argument("--save_images", type=int, default=8, help="images per config saved as a grid")
args = parser.parse_args()


def coarse_grid():
    out = []
    for a0, a1, K, sch in itertools.product([85, 80, 70, 60], [30, 20, 10, 0], [4, 8, 12, 20], ["linear", "cosine"]):
        out.append(dict(alpha_start=a0, alpha_end=a1, K=K, schedule=sch))
    for a0, K in itertools.product([85, 80, 70, 60, 50, 40, 30], [4, 8, 12, 20]):
        out.append(dict(alpha_start=a0, alpha_end=a0, K=K, schedule="constant"))
    return out


def image_seed(batch_start, draw):
    import sphere.rng as rng

    return rng.fold_in(args.seed, "s2", batch_start, draw) % (2**63)


def main():
    dev = "cuda"
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    ds = image_dataset(bb.cfg["dataset_name"], bb.image_size, args.dev_dir)
    idx = balanced_subset(ds, args.n_images, seed=args.subset_seed)

    if args.stage == "coarse":
        overrides = coarse_grid()
    else:
        overrides = json.loads(args.configs)
    base = RepaintConfig(split=args.split)
    tok_cfgs = overrides if args.stage == "tok" else []
    cfgs = [] if args.stage == "tok" else [replace(base, **o) for o in overrides]

    run = Run(f"repaint_{args.stage}", {"args": vars(args), "n_configs": len(cfgs)}, dev_dir=args.dev_dir,
              ckpt_path=bb.ckpt_path, seeds={"noise": args.seed, "subset": args.subset_seed})
    json.dump([{"hash": config_hash(c), **asdict(c)} for c in cfgs], open(run.path("configs.json"), "w"), indent=2)
    lp = HalfLPIPS(dev)
    incep = InceptionFeatures(dev) if args.kid else None

    X, C, IDS = load_batch(ds, idx, "cpu")
    B = X.shape[0]
    mA = half_mask(B, bb.image_size, bb.image_size, args.split, args.observed)
    feats_real = None
    if incep is not None:
        feats_real = real_features(ds, incep, bb.image_size)
        logger.info(f"inception features of {feats_real.shape[0]} real val images")

    def batches():
        for s in range(0, B, args.batch_size):
            sl = slice(s, s + args.batch_size)
            yield sl, X[sl].to(dev), C[sl].to(dev), mA[sl].to(dev)

    def cls(c):
        return c if args.class_policy == "true" else None

    def metrics(x, x_prepin, gt, m, v_true, c):
        mB = 1 - m
        with torch.no_grad():
            v_hat = bb.encode(x, cls(c))
        return {
            "psnr_A_prepin": psnr_masked(x_prepin.clamp(-1, 1), gt, m),
            "psnr_B": psnr_masked(x, gt, mB),
            "lpips_B": lp(x, gt, args.split, "B" if args.observed == "first" else "A"),
            "lpips_full": lp.full(x, gt),
            "seam": seam_score(x, args.split),
            "lat_angle_true": latent_angle(v_hat, v_true),
        }

    def log_rows(method, cfg_d, ch, sl, met, draw=0, extra=None):
        for j in range(met["psnr_B"].shape[0]):
            run.row(image_id=IDS[sl][j] if isinstance(sl, slice) else IDS[j], method=method, config_hash=ch, config=cfg_d,
                    seed=args.seed, draw=draw, class_policy=args.class_policy, split=args.split, observed=args.observed,
                    metrics={k: float(v[j]) for k, v in met.items()}, **(extra or {}))

    ref_imgs = {"mirror": [], "oracle": []}
    with torch.no_grad():
        for sl, gt, c, m in batches():
            y = m * gt
            v_true = bb.encode(gt, cls(c))
            x_m = fill_missing(y, m, "mirror", args.split)
            log_rows("mirror", {}, "mirror", sl, metrics(x_m, x_m, gt, m, v_true, c))
            x_or = bb.decode(v_true, 0.0, cls(c))
            x_o = prox_mask(x_or, y, m, 0.0).clamp(-1, 1)
            log_rows("oracle", {}, "oracle", sl, metrics(x_o, x_or, gt, m, v_true, c))
            ref_imgs["mirror"].append(x_m.cpu())
            ref_imgs["oracle"].append(x_o.cpu())
    summary = {}
    for name in ("mirror", "oracle"):
        xs = torch.cat(ref_imgs[name])
        if incep is not None:
            summary[name] = {"kid": kid(incep(xs), feats_real, subset_size=min(50, B), n_subsets=100)}
    save_grid([X[: args.save_images], torch.cat(ref_imgs["mirror"])[: args.save_images], torch.cat(ref_imgs["oracle"])[: args.save_images]],
              run.path("reference_gt_mirror_oracle.png"))

    draws = range(args.n_samples) if args.stage == "samples" else [0]
    for ci, cfg in enumerate(cfgs):
        ch = config_hash(cfg)
        t0 = time.time()
        bb.reset_counters()
        outs = []
        for draw in draws:
            xs_draw = []
            for sl, gt, c, m in batches():
                y = m * gt
                gen = torch.Generator().manual_seed(image_seed(sl.start, draw))
                with torch.no_grad():
                    v_true = bb.encode(gt, cls(c))
                    x, traj = repaint_complete(bb, y, m, cfg, cls(c), generator=gen, return_traj=True)
                met = metrics(x, traj["x_prepin"], gt, m, v_true, c)
                last = traj["steps"][-1] if traj["steps"] else None
                if last is not None:
                    met["final_lat_move_deg"] = last["lat_move_deg"]
                    met["final_resid_A"] = last["resid_A"]
                log_rows("repaint", asdict(cfg), ch, sl, met, draw=draw)
                xs_draw.append(x.cpu())
                if args.stage == "traj":
                    steps = [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()} for st in traj["steps"]]
                    torch.save(steps, run.path(f"traj_{ch}_b{sl.start}.pt"))
                    save_grid([gt] + [t for t in traj["x"]], run.path(f"traj_{ch}_b{sl.start}.png"))
            outs.append(torch.cat(xs_draw))
        dt_s = time.time() - t0
        info = {"wall_s": round(dt_s, 2), "passes": dict(bb.counters), "per_image_dec_fwd": bb.counters["dec_fwd"] / (B * len(draws))}
        if incep is not None:
            info["kid"] = kid(incep(outs[0]), feats_real, subset_size=min(50, B), n_subsets=100)
        if len(draws) > 1:
            S = torch.stack(outs)
            div = []
            for s in range(0, B, args.batch_size):
                from halfsphere.metrics import diversity

                div.append(diversity(S[:, s : s + args.batch_size].to(dev), args.split, lp).cpu())
            info["diversity_lpips_B"] = torch.cat(div).tolist()
        summary[ch] = info
        if ci < 64 or args.stage != "coarse":
            save_grid([X[: args.save_images], outs[0][: args.save_images]], run.path(f"repaint_{ch}.png"))
        logger.info(f"[{ci + 1}/{len(cfgs)}] {ch} {asdict(cfg)} {info.get('kid', '')} {dt_s:.1f}s")
    iA, iB = token_index_for_half(256, 256, 16, args.split, args.observed)
    for tc in tok_cfgs:
        ch = "tok_" + config_hash(tc)
        bb.reset_counters()
        t0 = time.time()
        xs_all = []
        for sl, gt, c, m in batches():
            y = m * gt
            gen = torch.Generator().manual_seed(image_seed(sl.start, 0))
            with torch.no_grad():
                v_true = bb.encode(gt, cls(c))
                x = repaint_token_complete(bb, y, m, iA.to(dev), iB.to(dev), K=tc["K"], alphaA=tuple(tc["alphaA"]),
                                      alphaB=tuple(tc["alphaB"]), split=args.split, c=cls(c), generator=gen)
            log_rows("repaint_tok", tc, ch, sl, metrics(x, x, gt, m, v_true, c))
            xs_all.append(x.cpu())
        xs_all = torch.cat(xs_all)
        info = {"wall_s": round(time.time() - t0, 2), "passes": dict(bb.counters), "per_image_dec_fwd": bb.counters["dec_fwd"] / B}
        if incep is not None:
            info["kid"] = kid(incep(xs_all), feats_real, subset_size=min(50, B), n_subsets=100)
        summary[ch] = info
        save_grid([X[: args.save_images], xs_all[: args.save_images]], run.path(f"{ch}.png"))
        logger.info(f"tok {ch} {tc} {info.get('kid', '')}")
    json.dump(summary, open(run.path("summary.json"), "w"), indent=2)
    run.finish("ok", n_rows=B * (2 + len(cfgs) * len(draws) + len(tok_cfgs)))
    logger.info(f"done: {run.dir}")


if __name__ == "__main__":
    main()
