import argparse
import json
import logging

import torch

from halfsphere.backbone import SE2Backbone
from halfsphere.records import Run
from spherevideo.clips import video_clips
from spherevideo.trajectory import analyze, gram_pca, shared_directions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Trajectory analysis of video latent paths on the SE2 sphere")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-imagenet-256px")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--videos", type=str, default="workspace/videos/wan21_t2v_1.3b/*/")
parser.add_argument("--searched_pt", type=str, default="")
parser.add_argument("--n_frames", type=int, default=81)
parser.add_argument("--fps", type=float, default=16.0)
parser.add_argument("--src_fps", type=float, default=16.0)
args = parser.parse_args()


def rnd(o, d=3):
    if isinstance(o, float):
        return round(o, d)
    if isinstance(o, list):
        return [rnd(x, d) for x in o]
    if isinstance(o, dict):
        return {k: rnd(v, 1 if k == "similarity" else d) for k, v in o.items()}
    return o


def sphere_figure(per, names, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(14, 6.6), dpi=130)
    cmap = plt.get_cmap("tab20")
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    sx, sy, sz = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    for j, zoom in enumerate((False, True)):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.plot_wireframe(sx, sy, sz, color="0.82", linewidth=0.4)
        for n, p in enumerate(per):
            P = np.array(p["sphere3"])
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=cmap(n % 20), linewidth=1.3)
            ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], color=cmap(n % 20), s=14)
        ax.scatter([0], [0], [1], color="black", s=30)
        if zoom:
            r = max(max(abs(x) for q in per for pt in q["sphere3"] for x in pt[:2]), 0.05) * 1.1
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(1 - 1.6 * r * r - 0.02, 1.0)
            ax.set_title(f"{title} paths, zoomed on the north pole (frame 0)")
        else:
            ax.set_box_aspect((1, 1, 1))
            ax.set_title(f"{title} paths on the sphere (all clips start at the north pole)")
        ax.view_init(elev=28, azim=35)
        ax.set_axis_off()
    handles = [plt.Line2D([0], [0], color=cmap(n % 20), lw=2) for n in range(len(names))]
    fig.legend(handles, names, loc="lower center", ncol=min(8, len(names)), fontsize=7, frameon=False)
    fig.text(0.5, 0.955, "Distance from the pole = true angle from frame 0; direction = the clip's top-2 movement directions", ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    X, meta = video_clips(args.videos, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
    N, T = X.shape[:2]
    with torch.no_grad():
        flat = X.flatten(0, 1)
        Z = torch.cat([bb.encode(flat[i:i + 32].to(dev)) for i in range(0, flat.shape[0], 32)]).view(N, T, *bb.latent_shape)
    paths = {"encoder": Z}
    if args.searched_pt:
        S = torch.load(args.searched_pt)["latents"].float().to(dev)
        if S.shape[0] == N and S.shape[1] == T:
            paths["searched"] = S
        else:
            logger.warning(f"searched latents {tuple(S.shape)} do not match {N}x{T}; skipped")
    grid = int(round(bb.latent_shape[0] ** 0.5))
    run = Run("trajectories", {"args": vars(args), "clips": meta}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path)
    out = {"clips": [m["kind"] for m in meta], "fps": args.fps, "n_frames": T, "paths": {}}
    with torch.no_grad():
        for name, P in paths.items():
            per = [analyze(P[n], grid) for n in range(N)]
            coords, ev = gram_pca(P.flatten(0, 1), 2)
            out["paths"][name] = {
                "per_clip": per,
                "common_pca": coords.view(N, T, 2).tolist(), "common_pca_var": ev[:5].tolist(),
                "shared": shared_directions([P[n] for n in range(N)], k=10),
            }
            Hs = [p["growth_H"] for p in per]
            logger.info(f"{name}: H mean {sum(Hs) / N:.2f}; straightness {sum(p['straightness'] for p in per) / N:.3f}; shared {out['paths'][name]['shared']}")
    for name, v in out["paths"].items():
        sphere_figure(v["per_clip"], out["clips"], name, run.path("figs", f"sphere_{name}.png"))
    with open(run.path("trajectories.json"), "w") as fh:
        json.dump(rnd(out), fh)
    summary = {name: {"H_mean": sum(p["growth_H"] for p in v["per_clip"]) / N,
                      "straightness_mean": sum(p["straightness"] for p in v["per_clip"]) / N,
                      "pca2_var_mean": sum(sum(p["pca_var"][:2]) for p in v["per_clip"]) / N,
                      "shared": v["shared"],
                      "per_clip": {k: {"H": p["growth_H"], "straight": p["straightness"], "pca2": sum(p["pca_var"][:2])} for k, p in zip(out["clips"], v["per_clip"])}}
               for name, v in out["paths"].items()}
    print(json.dumps(rnd(summary), indent=2))
    run.finish("ok", summary=rnd(summary))
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
