"""
inverse problems with a pretrained sphere encoder: denoising, deblurring,
dehazing. implements SP^3 (Man et al., "Spherical Priors for Plug-and-Play
Restoration", arXiv 2606.16396), algorithm 1:

    x_0 = init(y)                                   # y or A^T y
    for each angle a_k of a schedule:
        z = enc(x)                                  # latent on the sphere
        z = cos(a_k) z + sin(a_k) e                 # noisy spherify
        x_prior = dec(z, label, a_k)
        x = argmin_x mu ||A x - y||^2 + ||x - x_prior||^2      # exact prox

mu weights the data term (mu = 0 turns the prox off, which is the plain
encode -> noise -> decode loop). blur is a circular convolution so the prox
is solved exactly in the fourier domain.

every config is scored (psnr, ssim, lpips) against the clean image at the
steps in --eval_steps, next to two references: the degraded input, and the
clean image passed once through the model (the reconstruction ceiling).

usage (single gpu, no torchrun):
    export HF_TOKEN_PATH=/path/to/hf/token  # dinov3 is a gated repo
    python inverse/run_inverse.py --job_dir sphere2-large-flowers-512px \
        --tasks deblur --run_name deblur_v2

schedules are strings: "30" one step at 30 deg, "10x20" twenty steps at
10 deg, "60-15x4" four steps linearly from 60 down to 15 deg.
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)

from sphere.builder import build_model  # noqa: E402
from sphere.loader import center_crop_arr  # noqa: E402

logger = logging.getLogger("inverse")

# degradation strengths per task:
#   denoise: std of additive gaussian noise, on the [0, 1] pixel scale
#   deblur:  std (pixels, at the model resolution) of a gaussian blur kernel
#   dehaze:  transmission t of homogeneous haze, I = J t + A (1 - t)
DEFAULT_LEVELS = {
    "denoise": [0.1, 0.2, 0.3],
    "deblur": [2.0, 4.0, 8.0],
    "dehaze": [0.7, 0.5, 0.3],
}
DEFAULT_SCHEDULES = ["0", "10x20", "20x20", "30x20"]
HAZE_AIRLIGHT = 0.9


"""
schedules
"""


def parse_schedule(spec):
    """'30' -> [30], '30x3' -> [30, 30, 30], '60-15x4' -> [60, 45, 30, 15]"""
    head, _, steps = spec.partition("x")
    steps = int(steps) if steps else 1
    start, _, end = head.partition("-")
    start = float(start)
    end = float(end) if end else start
    if steps == 1:
        return [start]
    return [start + (end - start) * k / (steps - 1) for k in range(steps)]


"""
forward operators, all on [0, 1] images of shape [B, 3, H, W]
"""


def gaussian_otf(sigma, size, device):
    """fourier transform of a centered, normalized 2d gaussian kernel, for
    circular convolution on a size x size image"""
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k1 = torch.exp(-0.5 * (t / sigma) ** 2)
    k1 = k1 / k1.sum()
    psf = torch.zeros(size, size, device=device)
    psf[: 2 * radius + 1, : 2 * radius + 1] = k1[:, None] * k1[None, :]
    psf = torch.roll(psf, shifts=(-radius, -radius), dims=(0, 1))
    return torch.fft.fft2(psf)


class Operator:
    """y = A(x) + n, with the exact data prox
    argmin_x mu ||A x - y||^2 + ||x - z||^2"""

    def __init__(self, task, level, image_size, device, meas_noise=0.0):
        self.task, self.level = task, level
        # denoise: the level is the noise std; other tasks: optional extra noise
        self.noise = level if task == "denoise" else meas_noise
        if task == "deblur":
            self.otf = gaussian_otf(level, image_size, device)

    def forward(self, x):
        if self.task == "denoise":
            return x
        if self.task == "deblur":
            return torch.fft.ifft2(torch.fft.fft2(x) * self.otf).real
        if self.task == "dehaze":
            return x * self.level + HAZE_AIRLIGHT * (1 - self.level)
        raise ValueError(self.task)

    def adjoint(self, r):
        if self.task == "denoise":
            return r
        if self.task == "deblur":
            return torch.fft.ifft2(torch.fft.fft2(r) * self.otf.conj()).real
        if self.task == "dehaze":
            return r * self.level
        raise ValueError(self.task)

    def init(self, y, mode):
        """starting point x_0: the measurement, or A^T y (paper's choice for
        denoising and deblurring; for haze A^T y only scales y down)"""
        if mode == "adjoint" and self.task != "dehaze":
            return self.adjoint(y).clamp(0, 1)
        return y

    def prox(self, z, y, mu):
        if mu <= 0:
            return z
        if self.task == "denoise":
            return (mu * y + z) / (mu + 1)
        if self.task == "deblur":
            num = mu * self.otf.conj() * torch.fft.fft2(y) + torch.fft.fft2(z)
            den = mu * self.otf.abs() ** 2 + 1
            return torch.fft.ifft2(num / den).real
        if self.task == "dehaze":
            t, c = self.level, HAZE_AIRLIGHT * (1 - self.level)
            return (mu * t * (y - c) + z) / (mu * t * t + 1)
        raise ValueError(self.task)

    def degrade(self, x, generator):
        y = self.forward(x)
        if self.noise > 0:
            n = torch.randn(x.shape, generator=generator, device="cpu")
            y = y + self.noise * n.to(x.device)
        return y.clamp(0, 1)


"""
model
"""


def load_model(dev_dir, job_dir, device, ckpt_fname=None):
    exp_dir = osp.join(dev_dir, "experiments", job_dir)
    with open(osp.join(exp_dir, "cfg.json")) as f:
        cfg = SimpleNamespace(**json.load(f))

    model = build_model(cfg, for_train=False)

    ckpts = sorted(glob.glob(osp.join(exp_dir, "ckpt", "*.pth")))
    ckpt_path = osp.join(exp_dir, "ckpt", ckpt_fname) if ckpt_fname else ckpts[-1]
    logger.info(f"loading ema weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt["ema_model"] if "ema_model" in ckpt else ckpt["model"]
    state = {
        k.replace("module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("._orig_mod", ""): v
        for k, v in state.items()
    }
    msgs = model.load_state_dict(state, strict=False)
    if msgs.missing_keys or msgs.unexpected_keys:
        logger.warning(
            f"missing: {msgs.missing_keys} unexpected: {msgs.unexpected_keys}"
        )

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.dtype]
    model.to(device=device, dtype=ptdtype, memory_format=torch.channels_last)
    model.eval().requires_grad_(False)
    return model, cfg, ptdtype


@torch.no_grad()
def project(model, x, labels, angle, ptdtype):
    """P(x) = dec(spherify(enc(x), angle)), on [0, 1] images"""
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        z = model.encode((x * 2 - 1).to(ptdtype))
        z, alpha = model.spherify(z, angle)
        x = model.decoder(z, labels, alpha=alpha).float()
    return (x * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def restore(model, y, labels, angles, op, mu, init, ptdtype, keep_steps=()):
    """SP^3 from x_0 = init(y). returns the final x and {step: x} for the
    1-indexed steps in keep_steps"""
    x, kept = op.init(y, init), {}
    for k, angle in enumerate(angles, start=1):
        x = op.prox(project(model, x, labels, angle, ptdtype), y, mu).clamp(0, 1)
        if k in keep_steps:
            kept[k] = x
    return x, kept


"""
data and metrics
"""


def load_images(data_dir, split, num_images, image_size):
    """one image per class (the first in the split json), cycling through
    classes until num_images are picked"""
    with open(osp.join(data_dir, f"{split}.json")) as f:
        items = [json.loads(line) for line in f if line.strip()]
    by_class = defaultdict(list)
    for it in sorted(items, key=lambda it: it["image_path"]):
        by_class[it["class_id"]].append(it)
    picked, rnd = [], 0
    while len(picked) < num_images and any(len(v) > rnd for v in by_class.values()):
        for c in sorted(by_class):
            if rnd < len(by_class[c]) and len(picked) < num_images:
                picked.append(by_class[c][rnd])
        rnd += 1

    xs = []
    for it in picked:
        img = Image.open(osp.join(data_dir, it["image_path"])).convert("RGB")
        arr = np.asarray(center_crop_arr(img, image_size), dtype=np.float32)
        xs.append(torch.from_numpy(arr).permute(2, 0, 1) / 255.0)
    names = [osp.splitext(osp.basename(it["image_path"]))[0] for it in picked]
    labels = torch.tensor([it["class_id"] for it in picked], dtype=torch.long)
    return torch.stack(xs), labels, names


class Metrics:
    def __init__(self, device):
        import lpips
        from torchmetrics.functional import structural_similarity_index_measure

        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.ssim = structural_similarity_index_measure

    @torch.no_grad()
    def __call__(self, x, ref):
        """per-image psnr, ssim, lpips of x vs ref, both [B, 3, H, W] in [0, 1]"""
        mse = ((x - ref) ** 2).flatten(1).mean(1)
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
        ssim = self.ssim(x, ref, data_range=1.0, reduction="none")
        lp = self.lpips(x * 2 - 1, ref * 2 - 1).flatten()
        return psnr.cpu(), ssim.cpu(), lp.cpu()


def save_grid(rows, path, tile, mode):
    """rows: list of lists of [3, H, W] tensors in [0, 1]. mode 'crop' keeps a
    tile x tile center crop at full resolution, 'resize' downsamples"""

    def fit(t):
        if mode == "crop":
            H, W = t.shape[-2:]
            top, left = (H - tile) // 2, (W - tile) // 2
            return t[:, top:top + tile, left:left + tile]
        return F.interpolate(t[None], size=tile, mode="bilinear", antialias=True)[0]

    grid = torch.cat([torch.cat([fit(t) for t in row], dim=2) for row in rows], dim=1)
    arr = (grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round()
    Image.fromarray(arr.astype(np.uint8)).save(path)


"""
main
"""


def get_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dev_dir", default=osp.join(REPO, "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--ckpt_fname", default=None)
    p.add_argument("--data_dir", default=None,
                   help="default: <dev_dir>/datasets/<dataset_name of cfg>")
    p.add_argument("--split", default="val")
    p.add_argument("--num_images", type=int, default=102)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_LEVELS),
                   choices=list(DEFAULT_LEVELS))
    p.add_argument("--levels", nargs="+", type=float, default=None,
                   help="override the levels of every task in --tasks")
    p.add_argument("--meas_noise", type=float, default=0.0,
                   help="extra gaussian measurement noise for deblur/dehaze")
    p.add_argument("--schedules", nargs="+", default=DEFAULT_SCHEDULES)
    p.add_argument("--mus", nargs="+", type=float, default=[0.0, 1.0],
                   help="data-prox weights; 0 = no data consistency")
    p.add_argument("--init", choices=["y", "adjoint"], default="adjoint")
    p.add_argument("--label", choices=["null", "true"], default="null",
                   help="decoder class label: null (unconditional) or the true class")
    p.add_argument("--eval_steps", nargs="+", type=int, default=[1, 3, 5, 10, 20],
                   help="also score these intermediate steps (anytime curve)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_grid", type=int, default=6)
    p.add_argument("--grid_tile", type=int, default=256)
    p.add_argument("--grid_mode", choices=["crop", "resize"], default="crop")
    p.add_argument("--out_dir", default=None,
                   help="default: <dev_dir>/inverse/<job_dir>/<run_name>")
    p.add_argument("--run_name", default=time.strftime("%Y%m%d-%H%M%S"))
    return p.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = args.out_dir or osp.join(
        args.dev_dir, "inverse", args.job_dir, args.run_name
    )
    os.makedirs(osp.join(out_dir, "grids"), exist_ok=True)
    with open(osp.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"writing to {out_dir}")

    model, cfg, ptdtype = load_model(args.dev_dir, args.job_dir, device, args.ckpt_fname)
    data_dir = args.data_dir or osp.join(args.dev_dir, "datasets", cfg.dataset_name)
    clean, labels, names = load_images(
        data_dir, args.split, args.num_images, cfg.image_size
    )
    logger.info(f"{len(names)} {args.split} images from {data_dir}")
    if args.label == "null":
        labels = torch.full_like(labels, model.num_classes)
    metrics = Metrics(device)

    configs = [(s, mu) for s in args.schedules for mu in args.mus]
    rows = []
    summary = defaultdict(lambda: defaultdict(list))  # (task, level, col, step)
    n_grid = min(args.num_grid, len(names))

    def record(task, level, col, step, xr, x, b0):
        psnr, ssim, lp = metrics(xr, x)
        for i in range(x.shape[0]):
            rows.append(dict(
                task=task, level=level, config=col, step=step,
                label=args.label, image=names[b0 + i], psnr=float(psnr[i]),
                ssim=float(ssim[i]), lpips=float(lp[i]),
            ))
        s = summary[(task, level, col, step)]
        s["psnr"] += psnr.tolist()
        s["ssim"] += ssim.tolist()
        s["lpips"] += lp.tolist()

    for task in args.tasks:
        for level in args.levels or DEFAULT_LEVELS[task]:
            op = Operator(task, level, cfg.image_size, device, args.meas_noise)
            gen = torch.Generator().manual_seed(args.seed)
            grid_cols = defaultdict(list)
            t0 = time.time()

            for b0 in range(0, len(names), args.batch_size):
                x = clean[b0:b0 + args.batch_size].to(device)
                yb = labels[b0:b0 + args.batch_size].to(device)
                y = op.degrade(x, gen)
                in_grid = max(0, min(n_grid - b0, x.shape[0]))

                torch.manual_seed(args.seed * 100003 + b0)
                ceiling = project(model, x, yb, 0.0, ptdtype)
                outs = [("degraded", 0, y), ("clean-recon", 1, ceiling)]
                for sched, mu in configs:
                    # same rotation noise for every config of a batch
                    torch.manual_seed(args.seed * 100003 + b0)
                    angles = parse_schedule(sched)
                    xr, kept = restore(model, y, yb, angles, op, mu, args.init,
                                       ptdtype, keep_steps=set(args.eval_steps))
                    col = f"{sched}|mu={mu:g}"
                    for step, xs in kept.items():
                        if step < len(angles):
                            record(task, level, col, step, xs, x, b0)
                    outs.append((col, len(angles), xr))

                for col, step, xr in outs:
                    record(task, level, col, step, xr, x, b0)
                    grid_cols[col] += list(xr[:in_grid].cpu())
                grid_cols["clean"] += list(x[:in_grid].cpu())

            cols = ["clean", "degraded", "clean-recon"] + [
                f"{s}|mu={mu:g}" for s, mu in configs
            ]
            tag = f"{task}_{level:g}"
            save_grid(
                [[grid_cols[c][i] for c in cols] for i in range(n_grid)],
                osp.join(out_dir, "grids", f"{tag}.png"), args.grid_tile,
                args.grid_mode,
            )
            with open(osp.join(out_dir, "grids", f"{tag}_columns.txt"), "w") as f:
                f.write("\n".join(cols) + "\n")

            # final step of every config of this level -> lowest mean lpips
            final = {}
            for (t, lv, c, step), s in sorted(summary.items(), key=lambda kv: kv[0][3]):
                if (t, lv) == (task, level) and c != "clean-recon":
                    final[c] = s  # later steps overwrite earlier ones
            best = min(final, key=lambda c: np.mean(final[c]["lpips"]))
            deg, bst = final["degraded"], final[best]
            logger.info(
                f"{tag}: degraded psnr {np.mean(deg['psnr']):.2f} lpips {np.mean(deg['lpips']):.3f}"
                f" | best-lpips {best}: psnr {np.mean(bst['psnr']):.2f}"
                f" lpips {np.mean(bst['lpips']):.3f} ({time.time() - t0:.0f}s)"
            )

    with open(osp.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "| task | level | config | step | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    with open(osp.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "level", "config", "step", "n", "psnr", "ssim", "lpips"])
        for (task, level, col, step), s in sorted(
            summary.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
        ):
            m = {k: float(np.mean(v)) for k, v in s.items()}
            w.writerow([task, level, col, step, len(s["psnr"]),
                        f"{m['psnr']:.3f}", f"{m['ssim']:.4f}", f"{m['lpips']:.4f}"])
            lines.append(
                f"| {task} | {level:g} | {col} | {step} | {m['psnr']:.2f} | "
                f"{m['ssim']:.3f} | {m['lpips']:.3f} |"
            )
    with open(osp.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"done: {osp.join(out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
