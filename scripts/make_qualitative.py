import argparse
import json
import logging
from dataclasses import replace

import torch

import sphere.rng as rng
from halfsphere.backbone import SE2Backbone
from halfsphere.data import balanced_subset, image_dataset, load_batch
from halfsphere.masks import fill_missing, half_mask, prox_mask
from halfsphere.records import Run, save_grid
from halfsphere.repaint import RepaintConfig, repaint_complete
from halfsphere.steer import SteerConfig, steer_optimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="qualitative completion panel")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-flowers-256px")
parser.add_argument("--n_show", type=int, default=4)
parser.add_argument("--show_offset", type=int, default=0, help="first image (within the subset) to show")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repaint_A", type=str, default='{"K": 4, "alpha_start": 50, "alpha_end": 50, "schedule": "constant"}')
parser.add_argument("--repaint_B", type=str, default='{"K": 12, "alpha_start": 80, "alpha_end": 20, "schedule": "linear"}')
parser.add_argument("--steer_config", type=str, default='{"lam_cyc": 0.1, "steps": 300, "n_restarts": 2}')
parser.add_argument("--repaint_batch", type=int, default=32, help="must match run_repaint.py --batch_size to reproduce its noise")
parser.add_argument("--steer_batch", type=int, default=16, help="must match run_steer.py --batch_size")
args = parser.parse_args()


def main():
    dev = "cuda"
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=torch.bfloat16)
    ds = image_dataset(bb.cfg["dataset_name"], bb.image_size, args.dev_dir)
    run = Run("qualitative", {"args": vars(args)}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path, seeds={"base": args.seed})
    lo, hi = args.show_offset, args.show_offset + args.n_show
    idx = balanced_subset(ds, max(args.repaint_batch, hi), seed=0)[: args.repaint_batch]
    gt, c, _ = load_batch(ds, idx, dev)
    m = half_mask(gt.shape[0], bb.image_size, bb.image_size, device=dev)
    y = m * gt
    cols = {"truth": gt, "observed": y + (1 - m) * 0.0, "mirror": fill_missing(y, m, "mirror")}
    for name, cfgj, draw in (("Repaint A", args.repaint_A, 0), ("Repaint B d0", args.repaint_B, 0), ("Repaint B d1", args.repaint_B, 1)):
        g = torch.Generator().manual_seed(rng.fold_in(args.seed, "s2", 0, draw) % (2**63))
        cols[name] = repaint_complete(bb, y, m, replace(RepaintConfig(), **json.loads(cfgj)), c, generator=g)
    steer_b = slice(0, args.steer_batch)
    cfg1 = replace(SteerConfig(), **json.loads(args.steer_config))
    X1, _, L1, _ = steer_optimize(bb, y[steer_b], m[steer_b], cfg1, c[steer_b], seed=rng.fold_in(args.seed, "s1", 0) % (2**63))
    best = L1.argmin(0)
    x1 = X1[best, torch.arange(X1.shape[1], device=dev)]
    cols["Steer"] = torch.cat([prox_mask(x1, y[steer_b], m[steer_b], 0).clamp(-1, 1), cols["mirror"][args.steer_batch :]])
    with torch.no_grad():
        cols["oracle"] = prox_mask(bb.decode(bb.encode(gt, c), 0.0, c), y, m, 0).clamp(-1, 1)
    order = ["truth", "observed", "mirror", "Repaint A", "Repaint B d0", "Repaint B d1", "Steer", "oracle"]
    save_grid([cols[k][lo:hi] for k in order], run.path("panel.png"))
    json.dump({"columns": order, "images": [ds.list[i]["image_path"] for i in idx[lo:hi]]}, open(run.path("panel.json"), "w"), indent=2)
    run.finish("ok")
    logger.info(f"done: {run.dir}")


if __name__ == "__main__":
    main()
