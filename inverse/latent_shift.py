"""
does subtracting a damage's mean latent shift undo it?

for each damage type and strength:
    1. on 64 train images, the damage direction d is the mean shift between
       the latent of the damaged image and the latent of the clean one
       (latents scaled to unit norm). the damaged image is what the
       restoration pipeline encodes in its first pass (op.init(y))
    2. the first pass decodes z - alpha * |z| * d instead of z (alpha = 0 is
       the usual pass). alpha is chosen by lpips of the plain loop on 32 other
       train images, separately for each damage type and strength
    3. on validation images, score against the clean image for every alpha:
       one pass with no anchor, one pass with the anchor, and the task's full
       recommended pipeline (readme) with the correction in its first pass.
       also the latent angle to the clean latent

writes shift_summary.json to <dev_dir>/inverse/<job_dir>/latent/.

usage:
    python inverse/latent_shift.py
"""

import argparse
import json
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run_inverse as ri  # noqa: E402
from latent_analysis import TASKS, angle_deg, latent  # noqa: E402

ANGLE = 10.0
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


@torch.no_grad()
def encode_raw(model, x, ptdtype):
    """the encoder's output as is (bf16 values in float32), so that decoding it
    unchanged reproduces the restoration pipeline exactly"""
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        return model.encode((x * 2 - 1).to(ptdtype)).float()


@torch.no_grad()
def one_pass(model, z, null, ptdtype):
    """decode a latent the way one pass of the loop does (10 deg of noise)"""
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        zz, alpha = model.spherify(z.to(ptdtype), ANGLE)
        x = model.decoder(zz, null, alpha=alpha).float()
    return (x * 0.5 + 0.5).clamp(0, 1)


def damaged_latents(model, ptdtype, clean, op, seed):
    """per batch: the damaged image, the unit-norm clean latent, the raw latent
    of what pass 1 encodes, and the batch's haze map (patchy haze is drawn per
    batch, so the anchor needs the map of the batch it is applied to)"""
    gen = torch.Generator().manual_seed(seed)
    out = {"y": [], "z_clean": [], "z_in": [], "t": []}
    for b in range(0, clean.shape[0], 8):
        y = op.degrade(clean[b:b + 8], gen)
        out["t"].append(op.t)
        out["y"].append(y)
        out["z_clean"].append(latent(model, clean[b:b + 8], ptdtype))
        out["z_in"].append(encode_raw(model, op.init(y, "adjoint"), ptdtype))
    return {k: (v if k == "t" else torch.cat(v)) for k, v in out.items()}


def unit(z):
    return z / z.flatten(1).norm(dim=1).view(-1, 1, 1)


def corrected(z_raw, d, a):
    """subtract a * d at the latent's own scale (a = 0 returns z_raw as is)"""
    return z_raw - a * z_raw.flatten(1).norm(dim=1).view(-1, 1, 1) * d


def score(model, ptdtype, metrics, null, z_raw, data, clean, op, mu, passes, seed):
    """mean psnr and lpips after `passes` passes (anchored when mu > 0), the
    first pass decoding z_raw. with z_raw uncorrected this is exactly
    run_inverse.restore"""
    ps, lp = [], []
    for b in range(0, clean.shape[0], 8):
        op.t = data["t"][b // 8]
        y = data["y"][b:b + 8]
        torch.manual_seed(seed * 100003 + b)
        x = one_pass(model, z_raw[b:b + 8], null, ptdtype)
        for k in range(passes):
            if k > 0:
                x = ri.project(model, x, null, ANGLE, ptdtype)
            if mu > 0:
                x = op.prox(x, y, mu).clamp(0, 1)
        m = metrics(x, clean[b:b + 8])
        ps.append(m[0])
        lp.append(m[2])
    return {"psnr": torch.cat(ps).mean().item(), "lpips": torch.cat(lp).mean().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--dev_dir", default=osp.join(ri.REPO, "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--num_train", type=int, default=64, help="images for the direction")
    p.add_argument("--num_select", type=int, default=32, help="other train images to pick alpha")
    p.add_argument("--num_val", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    model, cfg, ptdtype = ri.load_model(args.dev_dir, args.job_dir, device)
    data_dir = osp.join(args.dev_dir, "datasets", cfg.dataset_name)
    train, _, _ = ri.load_images(data_dir, "train", args.num_train + args.num_select, cfg.image_size)
    train, select = train[:args.num_train].to(device), train[args.num_train:].to(device)
    val, _, _ = ri.load_images(data_dir, "val", args.num_val, cfg.image_size)
    val = val.to(device)
    null = torch.full((8,), model.num_classes, device=device)
    metrics = ri.Metrics(device)

    out_dir = osp.join(args.dev_dir, "inverse", args.job_dir, "latent")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = osp.join(out_dir, "shift_summary.json")
    summary = json.load(open(summary_path)) if osp.exists(summary_path) else {}
    for task in args.tasks:
        spec = TASKS[task]
        rows = []
        # the model's own reconstruction of the clean image: the best a pure
        # latent fix can reach
        ceil = [metrics(ri.project(model, val[b:b + 8], null, 0.0, ptdtype), val[b:b + 8])
                for b in range(0, val.shape[0], 8)]
        ceiling = {"psnr": torch.cat([c[0] for c in ceil]).mean().item(),
                   "lpips": torch.cat([c[2] for c in ceil]).mean().item()}
        for level in spec["levels"]:
            op = ri.Operator(task, level, cfg.image_size, device, spec["noise"], "patchy")
            tr = damaged_latents(model, ptdtype, train, op, args.seed + 1)
            d = (unit(tr["z_in"]) - tr["z_clean"]).mean(0, keepdim=True)  # [1, N, D]
            # pick alpha on held-out train images, by lpips of the plain loop
            se = damaged_latents(model, ptdtype, select, op, args.seed + 2)
            sel = {a: score(model, ptdtype, metrics, null, corrected(se["z_in"], d, a), se,
                            select, op, 0.0, 1, args.seed)["lpips"] for a in ALPHAS}
            alpha = min(sel, key=sel.get)

            va = damaged_latents(model, ptdtype, val, op, args.seed)
            deg = metrics(va["y"], val)
            row = {"level": level, "alpha": alpha, "select_lpips": {f"{a:g}": v for a, v in sel.items()},
                   "degraded": {"psnr": deg[0].mean().item(), "lpips": deg[2].mean().item()},
                   "recommended": {"mu": spec["mu"], "passes": spec["passes"]}, "alphas": {}}
            mu = float(spec["mu"])
            for a in ALPHAS:
                z = corrected(va["z_in"], d, a)
                res = {"angle": angle_deg(va["z_clean"].flatten(1), z.flatten(1), 1).median().item(),
                       "loop": score(model, ptdtype, metrics, null, z, va, val, op, 0.0, 1, args.seed)}
                if mu:
                    res["anchored"] = score(model, ptdtype, metrics, null, z, va, val, op, mu, 1, args.seed)
                if spec["passes"] > 1:
                    res["recommended"] = score(model, ptdtype, metrics, null, z, va, val, op, mu,
                                               spec["passes"], args.seed)
                row["alphas"][f"{a:g}"] = res
            rows.append(row)
            b0, ba = row["alphas"]["0"], row["alphas"][f"{alpha:g}"]
            fmt = lambda k: f" | {k} {b0[k]['psnr']:.2f}/{b0[k]['lpips']:.3f} -> {ba[k]['psnr']:.2f}/{ba[k]['lpips']:.3f}"  # noqa: E731
            print(f"{task} {level:g}: alpha {alpha:g} | angle {b0['angle']:.1f} -> {ba['angle']:.1f} deg"
                  + "".join(fmt(k) for k in ("loop", "anchored", "recommended") if k in b0), flush=True)
        summary[task] = {"name": spec["name"], "ceiling": ceiling, "levels": rows}

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
