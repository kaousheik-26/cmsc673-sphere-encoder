"""
the one-step fix: damaged image -> encode -> subtract the damage direction ->
decode. no latent noise, no anchor, no further passes.

for each damage type at its heaviest strength:
    d      the mean shift between damaged and clean latents (unit norm) over
           64 train images
    alpha  picked by lpips on 32 other train images
    score  32 validation images, decoded with and without the subtraction
figure: 4 validation images, the damaged input and the result for every
damage type. writes direct_summary.json and direct_fix.png to
<dev_dir>/inverse/<job_dir>/latent/.

usage:
    python inverse/shift_direct.py
"""

import argparse
import json
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run_inverse as ri  # noqa: E402
from latent_analysis import TASKS, latent  # noqa: E402
from latent_shift import ALPHAS, corrected, encode_raw, unit  # noqa: E402
from make_figure import draw_figure  # noqa: E402

EXAMPLES = [0, 2, 4, 5]  # images of the first validation batch


@torch.no_grad()
def decode(model, z_raw, null, ptdtype):
    """decode a latent as is: projected onto the sphere, no added noise"""
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        zz, alpha = model.spherify(z_raw.to(ptdtype), 0.0)
        x = model.decoder(zz, null, alpha=alpha).float()
    return (x * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def encode_damaged(model, ptdtype, clean, op, seed):
    """per batch of 8: damaged image, raw latent of it, unit clean latent"""
    gen = torch.Generator().manual_seed(seed)
    ys, zs, zc = [], [], []
    for b in range(0, clean.shape[0], 8):
        y = op.degrade(clean[b:b + 8], gen)
        ys.append(y)
        zs.append(encode_raw(model, y, ptdtype))
        zc.append(latent(model, clean[b:b + 8], ptdtype))
    return torch.cat(ys), torch.cat(zs), torch.cat(zc)


def mean_scores(model, ptdtype, metrics, null, z_raw, clean):
    ps, lp, out = [], [], []
    for b in range(0, clean.shape[0], 8):
        x = decode(model, z_raw[b:b + 8], null, ptdtype)
        m = metrics(x, clean[b:b + 8])
        ps.append(m[0])
        lp.append(m[2])
        out.append(x)
    return {"psnr": torch.cat(ps).mean().item(), "lpips": torch.cat(lp).mean().item()}, torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dev_dir", default=osp.join(ri.REPO, "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    model, cfg, ptdtype = ri.load_model(args.dev_dir, args.job_dir, device)
    data_dir = osp.join(args.dev_dir, "datasets", cfg.dataset_name)
    train, _, _ = ri.load_images(data_dir, "train", 96, cfg.image_size)
    train, select = train[:64].to(device), train[64:].to(device)
    val, _, names = ri.load_images(data_dir, "val", 32, cfg.image_size)
    val = val.to(device)
    null = torch.full((8,), model.num_classes, device=device)
    metrics = ri.Metrics(device)
    idx = torch.tensor(EXAMPLES, device=device)

    summary, cols = {}, [("Clean", val[idx], None, None)]
    for task, spec in TASKS.items():
        level = spec["levels"][-1]
        op = ri.Operator(task, level, cfg.image_size, device, spec["noise"], "patchy")
        _, z_tr, zc_tr = encode_damaged(model, ptdtype, train, op, args.seed + 1)
        d = (unit(z_tr) - zc_tr).mean(0, keepdim=True)
        _, z_se, _ = encode_damaged(model, ptdtype, select, op, args.seed + 2)
        sel = {a: mean_scores(model, ptdtype, metrics, null, corrected(z_se, d, a), select)[0]["lpips"]
               for a in ALPHAS}
        alpha = min(sel, key=sel.get)

        y, z_va, _ = encode_damaged(model, ptdtype, val, op, args.seed)
        without, _ = mean_scores(model, ptdtype, metrics, null, z_va, val)
        with_fix, x_fix = mean_scores(model, ptdtype, metrics, null, corrected(z_va, d, alpha), val)
        deg = metrics(y, val)
        summary[task] = {"level": level, "alpha": alpha,
                         "damaged": {"psnr": deg[0].mean().item(), "lpips": deg[2].mean().item()},
                         "decode_without_subtraction": without, "decode_with_subtraction": with_fix}
        print(f"{task} {level:g}: alpha {alpha:g} | damaged {summary[task]['damaged']['psnr']:.2f}/"
              f"{summary[task]['damaged']['lpips']:.3f} | without {without['psnr']:.2f}/{without['lpips']:.3f}"
              f" | with {with_fix['psnr']:.2f}/{with_fix['lpips']:.3f}", flush=True)

        name = f"{spec['name']} {spec['fmt'](level)}"
        for label, imgs in [(f"{name}|damaged", y), (f"{name}|result", x_fix)]:
            psnr, _, lp = metrics(imgs[idx], val[idx])
            cols.append((label, imgs[idx], psnr, lp))

    out_dir = osp.join(args.dev_dir, "inverse", args.job_dir, "latent")
    with open(osp.join(out_dir, "direct_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    title = "Encode the damaged image, subtract the damage direction once, decode"
    out = osp.join(out_dir, "direct_fix.png")
    draw_figure(title, cols, [names[i] for i in EXAMPLES], args.tile).save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
