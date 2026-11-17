"""
the one-step latent fix when the damage strength is random and unknown.

every validation image gets its own strength, drawn uniformly over the range
tested before (blur 2 to 12px, noise std 0.1 to 0.5, haze densest patch 50%
to 10%, too bright x1.5 to x4). the fix (encode, subtract the damage
direction once with alpha 1, decode) is run with four choices of direction:
    none       plain encode, decode
    random     one direction per damage type, from 64 train images damaged
               at random strengths. needs no strength
    heaviest   the direction for the heaviest strength (the committed method
               used blindly)
    exact      from 64 train images damaged at the image's own strength (an
               oracle that knows the strength)

writes random_strength.csv (per image), random_strength.json (means, overall
and by thirds of strength) and random_<task>.png to
<dev_dir>/inverse/<job_dir>/latent/.

usage:
    python inverse/shift_random.py
"""

import argparse
import csv
import json
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run_inverse as ri  # noqa: E402
from latent_analysis import TASKS, latent  # noqa: E402
from latent_shift import corrected, encode_raw, unit  # noqa: E402
from make_figure import draw_figure  # noqa: E402
from shift_direct import decode  # noqa: E402

METHODS = ["none", "random", "heaviest", "exact"]
LABEL = {  # strength as shown in tables and figures
    "deblur": lambda v: f"{v:.1f}px",
    "denoise": lambda v: f"{v:.2f}",
    "dehaze": lambda v: f"{v:.0%}",
    "overexpose": lambda v: f"x{v:.1f}",
}


def sample_levels(spec, n, seed):
    lo, hi = min(spec["levels"]), max(spec["levels"])
    u = torch.rand(n, generator=torch.Generator().manual_seed(seed))
    return (lo + (hi - lo) * u).tolist()


def severity(task, level):
    """larger is heavier (haze level is the transmission, so lower is heavier)"""
    return -level if task == "dehaze" else level


def degrade_each(task, spec, x, levels, size, device, seed):
    """each image at its own strength"""
    out = []
    for i, lv in enumerate(levels):
        op = ri.Operator(task, lv, size, device, spec["noise"], "patchy")
        out.append(op.degrade(x[i:i + 1], torch.Generator().manual_seed(seed * 1000 + i)))
    return torch.cat(out)


@torch.no_grad()
def raw_latents(model, x, ptdtype, bs=16):
    return torch.cat([encode_raw(model, x[b:b + bs], ptdtype) for b in range(0, len(x), bs)])


def direction(model, ptdtype, damaged, z_clean):
    """mean shift of unit latents, damaged minus clean, [1, N, D]"""
    return (unit(raw_latents(model, damaged, ptdtype)) - z_clean).mean(0, keepdim=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--dev_dir", default=osp.join(ri.REPO, "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--num_train", type=int, default=64)
    p.add_argument("--num_val", type=int, default=96)
    p.add_argument("--tile", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    model, cfg, ptdtype = ri.load_model(args.dev_dir, args.job_dir, device)
    size = cfg.image_size
    data_dir = osp.join(args.dev_dir, "datasets", cfg.dataset_name)
    train = ri.load_images(data_dir, "train", args.num_train, size)[0].to(device)
    val, _, names = ri.load_images(data_dir, "val", args.num_val, size)
    val = val.to(device)
    metrics = ri.Metrics(device)
    z_train = torch.cat([latent(model, train[b:b + 16], ptdtype) for b in range(0, len(train), 16)])

    out_dir = osp.join(args.dev_dir, "inverse", args.job_dir, "latent")
    rows, summary = [], {}
    for task in args.tasks:
        spec, label = TASKS[task], LABEL[task]
        levels = sample_levels(spec, len(val), args.seed)
        y = degrade_each(task, spec, val, levels, size, device, args.seed)
        z_val = raw_latents(model, y, ptdtype)

        # the three directions
        train_levels = sample_levels(spec, len(train), args.seed + 1)
        d = {"random": direction(model, ptdtype, degrade_each(task, spec, train, train_levels, size,
                                                              device, args.seed + 1), z_train)}
        heavy = ri.Operator(task, spec["levels"][-1], size, device, spec["noise"], "patchy")
        d["heaviest"] = direction(model, ptdtype, heavy.degrade(train, torch.Generator().manual_seed(1)),
                                  z_train)
        exact = []
        for lv in levels:
            op = ri.Operator(task, lv, size, device, spec["noise"], "patchy")
            exact.append(direction(model, ptdtype, op.degrade(train, torch.Generator().manual_seed(1)),
                                   z_train))
        d["exact"] = torch.cat(exact)

        # fix with each direction, score every image
        results = {"damaged": y}
        for m in METHODS:
            zs = z_val if m == "none" else corrected(z_val, d[m], 1.0)
            results[m] = torch.cat([
                decode(model, zs[b:b + 8], torch.full((min(8, len(zs) - b),), model.num_classes,
                                                      device=device), ptdtype)
                for b in range(0, len(zs), 8)
            ])
        scores = {k: metrics(v, val) for k, v in results.items()}
        for i in range(len(val)):
            for k, (psnr, _, lp) in scores.items():
                rows.append(dict(task=task, image=names[i], level=round(levels[i], 4), method=k,
                                 psnr=round(float(psnr[i]), 3), lpips=round(float(lp[i]), 4)))

        # means, overall and by thirds of strength (mild, medium, heavy)
        order = sorted(range(len(val)), key=lambda i: severity(task, levels[i]))
        thirds = [order[k * len(order) // 3:(k + 1) * len(order) // 3] for k in range(3)]
        groups = {"all": order, "mild": thirds[0], "medium": thirds[1], "heavy": thirds[2]}
        summary[task] = {
            "name": spec["name"], "range": [label(min(levels)), label(max(levels))],
            "groups": {
                g: {"n": len(ix),
                    "strength": [label(levels[ix[0]]), label(levels[ix[-1]])] if g != "all" else None,
                    **{k: {"psnr": scores[k][0][ix].mean().item(), "lpips": scores[k][2][ix].mean().item()}
                       for k in scores}}
                for g, ix in groups.items()
            },
        }
        for g in ["all", "mild", "medium", "heavy"]:
            s = summary[task]["groups"][g]
            print(f"{task} {g:6s} " + " | ".join(
                f"{k} {s[k]['psnr']:.2f}/{s[k]['lpips']:.3f}" for k in ["damaged"] + METHODS), flush=True)

        # figure: four images spread over the strength range
        pick = [order[round(q * (len(order) - 1))] for q in (0, 1 / 3, 2 / 3, 1)]
        idx = torch.tensor(pick, device=device)
        col_names = {"damaged": "Damaged", "none": "No fix|(encode, decode)",
                     "random": "Fix, random-strength|direction", "heaviest": "Fix, heaviest-strength|direction",
                     "exact": "Fix, exact-strength|direction (oracle)"}
        cols = [("Clean", val[idx], None, None)]
        for k in ["damaged"] + METHODS:
            psnr, _, lp = metrics(results[k][idx], val[idx])
            cols.append((col_names[k], results[k][idx], psnr, lp))
        title = (f"{spec['name']} at a random strength ({label(min(levels))} to {label(max(levels))}): "
                 f"one-step latent fix with different damage directions")
        draw_figure(title, cols, [f"{names[i]}, {label(levels[i])}" for i in pick], args.tile).save(
            osp.join(out_dir, f"random_{task}.png"))

    with open(osp.join(out_dir, "random_strength.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(osp.join(out_dir, "random_strength.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {out_dir}/random_strength.json, .csv and random_<task>.png")


if __name__ == "__main__":
    main()
