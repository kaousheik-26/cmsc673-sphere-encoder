"""
labeled comparison figure: loop vs anchored restoration after k passes.

one row per image, one column per method, with psnr / lpips against the
clean image under every tile:
    clean, degraded
    loop, k passes       encode -> noise -> decode, k times, no anchor
    anchored, k passes   the same loop plus the data prox after each decode
for each k in --passes.

usage (defaults reproduce the figures in the readme, ~30 s each):
    python inverse/make_figure.py --task deblur
    python inverse/make_figure.py --task denoise --level 0.2 --passes 1 2 3
"""

import argparse
import os.path as osp
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run_inverse as ri  # noqa: E402

# task -> (degradation level, data weight mu, measurement noise, passes shown),
# the settings of the figures in the readme. mu is the best value from a sweep
# over train images. dehaze uses patchy haze; its level is the transmission in
# the densest patch
TASKS = {
    "denoise": (0.5, 0.05, 0.0, [1, 2, 3]),
    "deblur": (12.0, 1000.0, 0.0, [1, 3, 5]),
    "dehaze": (0.1, 25.0, 0.02, [1, 3, 5]),
    # too dark: no weight beats the plain loop on lpips; 100 has the best psnr
    "underexpose": (0.05, 100.0, 0.01, [1, 2, 3]),
    "overexpose": (4.0, 10.0, 0.01, [1, 3, 5]),
}
ANGLE = 10.0  # latent noise angle (deg) of every pass

FONT_DIR = osp.join(osp.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
PAD, HEAD, FOOT, TITLE_H = 6, 40, 22, 34


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(osp.join(FONT_DIR, name), size)


def to_pil(img, tile):
    """[3, H, W] in [0, 1] -> tile x tile PIL image"""
    t = F.interpolate(img[None], size=tile, mode="bilinear", antialias=True)[0]
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round()
    return Image.fromarray(arr.astype(np.uint8))


def draw_figure(title, cols, names, tile):
    """cols: [(label, images [N, 3, H, W], psnr [N] or None, lpips [N] or None)];
    a '|' in a label starts a second line"""
    nimg = len(names)
    cell = tile + PAD
    canvas = Image.new(
        "RGB",
        (len(cols) * cell + PAD, TITLE_H + HEAD + nimg * (tile + FOOT + PAD) + PAD),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    f_title, f_text = font(20, bold=True), font(14)

    draw.text((PAD, PAD + 6), title, font=f_title, fill="black")
    y0 = PAD + TITLE_H
    for c, (label, *_) in enumerate(cols):
        for li, line in enumerate(label.split("|")):
            draw.text((PAD + c * cell + tile // 2, y0 + 2 + li * 17), line,
                      font=f_text, fill="black", anchor="mt")
    y0 += HEAD
    for i in range(nimg):
        for c, (_, imgs, psnr, lp) in enumerate(cols):
            x = PAD + c * cell
            canvas.paste(to_pil(imgs[i], tile), (x, y0))
            if psnr is None:
                text, color = names[i], "gray"
            else:
                text, color = f"{psnr[i]:.1f} dB   LPIPS {lp[i]:.3f}", "black"
            draw.text((x + tile // 2, y0 + tile + 3), text, font=f_text,
                      fill=color, anchor="mt")
        y0 += tile + FOOT + PAD
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--dev_dir", default=osp.join(ri.REPO, "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--split", default="val")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--image_offset", type=int, default=0,
                   help="skip this many images (pick different examples)")
    p.add_argument("--level", type=float, default=None,
                   help="degradation strength (default: the task's entry in TASKS)")
    p.add_argument("--mu", type=float, default=None,
                   help="anchor data weight (default: the task's entry in TASKS)")
    p.add_argument("--passes", nargs="+", type=int, default=None,
                   help="passes to show (default: the task's entry in TASKS)")
    p.add_argument("--tile", type=int, default=384,
                   help="tile side in pixels (images are 512px; larger keeps more detail)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="default: <dev_dir>/inverse/<job_dir>/figure_<task>_<level>_passes_<..>.png")
    args = p.parse_args()

    device = "cuda"
    model, cfg, ptdtype = ri.load_model(args.dev_dir, args.job_dir, device)
    data_dir = osp.join(args.dev_dir, "datasets", cfg.dataset_name)
    clean, _, names = ri.load_images(
        data_dir, args.split, args.image_offset + args.num_images, cfg.image_size
    )
    clean, names = clean[args.image_offset:].to(device), names[args.image_offset:]
    null = torch.full((clean.shape[0],), model.num_classes, device=device)
    metrics = ri.Metrics(device)

    level, mu, noise, passes = TASKS[args.task]
    level = level if args.level is None else args.level
    mu = mu if args.mu is None else args.mu
    passes = sorted(passes if args.passes is None else args.passes)
    op = ri.Operator(args.task, level, cfg.image_size, device, noise, "patchy")
    y = op.degrade(clean, torch.Generator().manual_seed(args.seed))

    def run(m):
        """{k: result after k passes} for every k in passes"""
        torch.manual_seed(args.seed)
        return ri.restore(model, y, null, [ANGLE] * passes[-1], op, m,
                          "adjoint", ptdtype, keep_steps=set(passes))[1]

    def plural(k):
        return f"{k} pass{'es' if k > 1 else ''}"

    loop, anchored = run(0.0), run(mu)
    methods = [("Degraded", y)]
    if op.undo(y) is not None:
        # the known degradation inverted with no model
        what = "haze" if args.task == "dehaze" else "exposure"
        methods += [(f"Undo {what}|(no model)", op.undo(y))]
    methods += [(f"Loop, {plural(k)}|(no anchor)", loop[k]) for k in passes]
    methods += [(f"Anchored, {plural(k)}|(data weight {mu:g})", anchored[k])
                for k in passes]
    cols = [("Clean", clean, None, None)]
    for label, imgs in methods:
        psnr, _, lp = metrics(imgs, clean)
        cols.append((label, imgs, psnr, lp))

    title = {
        "denoise": f"Denoising (noise std {level:g})",
        "deblur": f"Deblurring (Gaussian blur std {level:g}px)",
        "dehaze": f"Dehazing (patchy haze, densest patch keeps {level:.0%} "
                  f"of the image, + noise {noise:g})",
        "underexpose": f"Too dark (exposure x{level:g}, + noise {noise:g})",
        "overexpose": f"Too bright (exposure x{level:g}, highlights clipped, "
                      f"+ noise {noise:g})",
    }[args.task]
    out = args.out or osp.join(
        args.dev_dir, "inverse", args.job_dir,
        f"figure_{args.task}_{level:g}_passes_{'-'.join(map(str, passes))}.png",
    )
    draw_figure(title, cols, names, args.tile).save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
