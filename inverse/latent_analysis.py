"""
how the sphere encoder's latent responds to each kind of damage.

the latent of an image is 16 x 16 tokens of 384 channels, rms-normalized as
one vector, so it lives on a sphere. for every damage type and strength:
    angle         between the clean and the damaged latent, with two scale
                  references: two different clean flowers, and a clean image
                  vs its own reconstruction re-encoded
    restored      the same angle after restoration (the task's recommended
                  setting from the readme)
    consistency   does damage shift every latent the same way? cosine between
                  one image's shift and the mean shift of the other images
                  (0 for unrelated directions, 1 for one shared direction)
    shift map     each image's shift (damaged minus clean latent) in 2d, from
                  an uncentered pca of all shifts, so every path starts at 0
    token maps    how far each of the 16 x 16 tokens moved, and its pearson
                  correlation with how much the pixels under it changed

writes latent_<task>.png and latent_summary.json to
<dev_dir>/inverse/<job_dir>/latent/.

usage:
    python inverse/latent_analysis.py --tasks deblur denoise dehaze overexpose
"""

import argparse
import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run_inverse as ri  # noqa: E402

# task -> strengths (mild to heavy), measurement noise, and the recommended
# restoration (anchor weight, passes) from the readme
TASKS = {
    "deblur": dict(name="Blur", levels=[2, 4, 6, 8, 12], noise=0.0, mu=1000, passes=3,
                   fmt=lambda v: f"{v:g}px"),
    "denoise": dict(name="Noise", levels=[0.1, 0.2, 0.3, 0.4, 0.5], noise=0.0, mu=0, passes=1,
                    fmt=lambda v: f"{v:g}"),
    "dehaze": dict(name="Haze", levels=[0.5, 0.3, 0.2, 0.1], noise=0.02, mu=25, passes=3,
                   fmt=lambda v: f"{v:.0%}"),
    "overexpose": dict(name="Too bright", levels=[1.5, 2, 3, 4], noise=0.01, mu=10, passes=2,
                       fmt=lambda v: f"x{v:g}"),
}
ANGLE = 10.0
EXAMPLES = [0, 2, 5]  # validation images shown in the token maps

# colors: orange = damaged, blue = restored, grays = references
ORANGE, BLUE, GRAY, DARK = "#eb6834", "#2a78d6", "#8a8a8a", "#555555"
ORANGE_RAMP = ["#f7c3a8", "#f19a6f", "#eb6834", "#c24d1f", "#8f3512"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#fdf3ee", "#f19a6f", "#c24d1f", "#5e220b"])


@torch.no_grad()
def latent(model, x, ptdtype):
    """[B, 3, H, W] in [0, 1] -> unit-norm latents [B, N, D] in float32"""
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        z = model.encode((x * 2 - 1).to(ptdtype)).float()
    return z / z.flatten(1).norm(dim=1).view(-1, 1, 1)


def angle_deg(a, b, dim):
    """angle between a and b along dim, stable for small angles"""
    a = a / a.norm(dim=dim, keepdim=True)
    b = b / b.norm(dim=dim, keepdim=True)
    return torch.rad2deg(2 * torch.asin(((a - b).norm(dim=dim) / 2).clamp(max=1)))


def pearson(a, b):
    """per-row correlation of [B, K] tensors"""
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def analyze(model, cfg, ptdtype, clean, null, task, args):
    spec = TASKS[task]
    device = clean.device
    n, H = clean.shape[0], cfg.image_size
    side = H // cfg.patch_size  # 16 tokens per side

    z_clean = torch.cat([latent(model, clean[b:b + 8], ptdtype) for b in range(0, n, 8)])
    flat_c = z_clean.flatten(1)
    # references: two different flowers, and a clean image through the model
    g = (flat_c @ flat_c.T).clamp(-1, 1)
    pairs = torch.rad2deg(torch.acos(g[torch.triu(torch.ones_like(g), 1) > 0]))
    recon = torch.cat([
        latent(model, ri.project(model, clean[b:b + 8], null[b:b + 8], 0.0, ptdtype), ptdtype)
        for b in range(0, n, 8)
    ])
    refs = {"between_flowers": pairs.median().item(),
            "through_model": angle_deg(flat_c, recon.flatten(1), 1).median().item()}

    out = {"name": spec["name"], "levels": [], "refs": refs}
    shifts, maps = [], {}
    for level in spec["levels"]:
        op = ri.Operator(task, level, H, device, spec["noise"], "patchy")
        gen = torch.Generator().manual_seed(args.seed)  # same draws as the analysis runs
        ys, z_deg, z_res = [], [], []
        for b in range(0, n, 8):
            y = op.degrade(clean[b:b + 8], gen)
            torch.manual_seed(args.seed * 100003 + b)
            restored = ri.restore(model, y, null[b:b + 8], [ANGLE] * spec["passes"], op,
                                  spec["mu"], "adjoint", ptdtype)[0]
            ys.append(y)
            z_deg.append(latent(model, y, ptdtype))
            z_res.append(latent(model, restored, ptdtype))
        y, z_deg, z_res = torch.cat(ys), torch.cat(z_deg), torch.cat(z_res)

        ang = angle_deg(flat_c, z_deg.flatten(1), 1)
        ang_res = angle_deg(flat_c, z_res.flatten(1), 1)
        # consistency: each image's shift vs the mean shift of the others
        delta = z_deg.flatten(1) - flat_c
        others = (delta.sum(0, keepdim=True) - delta) / (n - 1)
        consistency = F.cosine_similarity(delta, others, dim=1)
        # token maps: angle per token, pixel change per 32 x 32 patch
        tok = angle_deg(z_clean, z_deg, 2).view(n, side, side)
        pix = F.avg_pool2d((clean - y).abs().mean(1, keepdim=True), cfg.patch_size).view(n, -1)
        corr = pearson(tok.view(n, -1), pix)

        q = lambda t: [t.quantile(0.25).item(), t.median().item(), t.quantile(0.75).item()]  # noqa: E731
        out["levels"].append({
            "level": level, "angle": q(ang), "restored": q(ang_res),
            "consistency": consistency.mean().item(), "token_pixel_corr": corr.mean().item(),
        })
        shifts.append(delta)
        maps = {"y": y[EXAMPLES].cpu(), "tok": tok[EXAMPLES].cpu(), "corr": corr[EXAMPLES].cpu()}

    # 2d map of the shifts. uncentered, so a direction shared by all images
    # shows up as the first component. row 0 is the clean latent at the origin
    M = torch.cat(shifts)
    U, S, V = torch.pca_lowrank(M, q=2, center=False)
    coords = (M @ V[:, :2]).view(len(shifts), n, 2)
    coords = torch.cat([torch.zeros_like(coords[:1]), coords]).cpu().numpy()
    out["shift_pca_explained"] = (S[:2] ** 2 / M.square().sum()).tolist()
    return out, coords, maps


def plot(task, res, coords, maps, clean, path):
    spec = TASKS[task]
    lv = res["levels"]
    xs = np.arange(len(lv))
    labels = [spec["fmt"](d["level"]) for d in lv]
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c8c8c8", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    fig = plt.figure(figsize=(15, 15), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.1], hspace=0.42, top=0.87, bottom=0.03,
                          left=0.06, right=0.93)
    top = gs[0].subgridspec(1, 3, wspace=0.3)

    def clean_axes(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#ececec", linewidth=0.8)
        ax.set_axisbelow(True)

    # 1) how far the latent moves
    ax = fig.add_subplot(top[0])
    clean_axes(ax)
    for key, col, name in [("angle", ORANGE, "damaged"), ("restored", BLUE, "restored")]:
        lo, mid, hi = (np.array([d[key][i] for d in lv]) for i in range(3))
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15, linewidth=0)
        ax.plot(xs, mid, color=col, linewidth=2, marker="o", markersize=6, label=f"{name} vs clean")
    ax.axhline(res["refs"]["between_flowers"], color=GRAY, linestyle="--", linewidth=1.5,
               label="two different flowers")
    ax.axhline(res["refs"]["through_model"], color=GRAY, linestyle=":", linewidth=1.5,
               label="clean image through the model")
    ax.set_xticks(xs, labels)
    ax.set_xlabel("damage strength")
    ax.set_ylabel("angle to the clean latent (degrees)")
    ax.set_ylim(bottom=0)
    ax.set_title("How far the latent moves", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    # 2) is the shift the same for every image
    ax = fig.add_subplot(top[1])
    clean_axes(ax)
    ax.plot(xs, [d["consistency"] for d in lv], color=ORANGE, linewidth=2, marker="o", markersize=6)
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.5)
    ax.text(xs[-1], 0.03, "unrelated directions", ha="right", va="bottom", fontsize=9.5, color=GRAY)
    ax.set_ylim(-0.1, 1)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("damage strength")
    ax.set_ylabel("cosine with the other images' mean shift")
    ax.set_title("Same shift for every image?", loc="left", fontweight="bold")

    # 3) shift paths, all starting at the clean latent
    ax = fig.add_subplot(top[2])
    clean_axes(ax)
    ax.grid(False)
    ramp = ORANGE_RAMP[-len(lv):] if len(lv) < len(ORANGE_RAMP) else ORANGE_RAMP
    for i in range(coords.shape[1]):
        ax.plot(coords[:, i, 0], coords[:, i, 1], color="#d0d0d0", linewidth=0.8, zorder=1)
    for k in range(len(lv)):
        ax.scatter(coords[k + 1, :, 0], coords[k + 1, :, 1], s=22, color=ramp[k],
                   label=labels[k], zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter([0], [0], s=60, color=DARK, label="clean (start)", zorder=4)
    ev = res["shift_pca_explained"]
    ax.set_xlabel(f"direction 1 ({ev[0]:.0%} of the shift)")
    ax.set_ylabel(f"direction 2 ({ev[1]:.0%} of the shift)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Each image's shift in 2D", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    # 4) token maps for a few images at the heaviest strength
    bottom = gs[1].subgridspec(len(EXAMPLES), 4, width_ratios=[1, 1, 1, 0.06], wspace=0.06, hspace=0.16)
    vmax = float(maps["tok"].max())
    for r in range(len(EXAMPLES)):
        tiles = [(clean[EXAMPLES[r]].cpu(), "clean"),
                 (maps["y"][r], f"damaged ({labels[-1]})")]
        for c, (img, title) in enumerate(tiles):
            ax = fig.add_subplot(bottom[r, c])
            ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_axis_off()
            if r == 0:
                ax.set_title(title, fontsize=11)
        ax = fig.add_subplot(bottom[r, 2])
        im = ax.imshow(maps["tok"][r].numpy(), cmap=HEAT, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_axis_off()
        ax.text(0.02, 0.02, f"r = {maps['corr'][r]:.2f} with pixel change", transform=ax.transAxes,
                fontsize=10, color="#1a1a1a", bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
        if r == 0:
            ax.set_title("how far each latent token moved", fontsize=11)
    cax = fig.add_subplot(bottom[:, 3])
    fig.colorbar(im, cax=cax).set_label("token angle to clean (degrees)")

    corr = lv[-1]["token_pixel_corr"]
    setting = ("anchor weight " + str(spec["mu"]) + ", " if spec["mu"] else "no anchor, ") + \
        f"{spec['passes']} pass{'es' if spec['passes'] > 1 else ''}"
    fig.suptitle(f"{spec['name']}: how the latent responds to the damage", x=0.06, y=0.975,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.915,
             f"32 validation images. Restored uses the recommended setting ({setting}).\n"
             f"Token maps at the heaviest strength. Over all 32 images there, the correlation "
             f"between token movement and pixel change is r = {corr:.2f}.",
             fontsize=11.5, color="#555555", linespacing=1.5)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--dev_dir", default=osp.join(ri.REPO, "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--num_images", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    model, cfg, ptdtype = ri.load_model(args.dev_dir, args.job_dir, device)
    data_dir = osp.join(args.dev_dir, "datasets", cfg.dataset_name)
    clean, _, _ = ri.load_images(data_dir, "val", args.num_images, cfg.image_size)
    clean = clean.to(device)
    null = torch.full((clean.shape[0],), model.num_classes, device=device)

    out_dir = osp.join(args.dev_dir, "inverse", args.job_dir, "latent")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = osp.join(out_dir, "latent_summary.json")
    summary = json.load(open(summary_path)) if osp.exists(summary_path) else {}
    for task in args.tasks:
        res, coords, maps = analyze(model, cfg, ptdtype, clean, null, task, args)
        path = osp.join(out_dir, f"latent_{task}.png")
        plot(task, res, coords, maps, clean, path)
        summary[task] = res
        print(f"saved {path}")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
