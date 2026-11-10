import argparse
import json
import logging
import time

import torch

from halfsphere.backbone import SE2Backbone
from halfsphere.geometry import angle_deg
from halfsphere.metrics import HalfLPIPS
from halfsphere.records import Run, save_grid
from spherevideo.clips import synthetic_clips, video_clips, write_mp4
from spherevideo.predict import VelocityMap, extrapolate, search, snap, turning_deg, velocities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Feasibility probe: next-frame video prediction on the SE2 latent sphere")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-imagenet-256px")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "video"])
parser.add_argument("--images", type=str, default="workspace/datasets/flowers-102/jpg/*.jpg")
parser.add_argument("--videos", type=str, default="")
parser.add_argument("--n_clips", type=int, default=16)
parser.add_argument("--n_frames", type=int, default=40)
parser.add_argument("--fps", type=float, default=8.0)
parser.add_argument("--src_fps", type=float, default=16.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--map_rank", type=int, default=32)
parser.add_argument("--snap_alpha", type=float, default=0.0)
parser.add_argument("--search_clips", type=int, default=4)
parser.add_argument("--search_frames", type=int, default=12)
parser.add_argument("--search_steps", type=int, default=150)
parser.add_argument("--save_videos", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()


def chunked(fn, x, bs):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, x.shape[0], bs)])


def mean(t):
    return float(torch.as_tensor(t, dtype=torch.float32).mean())


def main():
    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    lp = HalfLPIPS(dev)
    if args.source == "synthetic":
        X, meta = synthetic_clips(args.images, args.n_clips, args.n_frames, bb.image_size, seed=args.seed)
    else:
        X, meta = video_clips(args.videos, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
    N, T = X.shape[:2]
    run = Run(f"videoprobe_{args.source}", {"args": vars(args), "clips": meta}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path, seeds={"base": args.seed})
    t0 = time.time()

    with torch.no_grad():
        Xf = X.flatten(0, 1).to(dev)
        Z = chunked(bb.encode, Xf, args.batch_size)
        R = chunked(lambda z: bb.decode(z, 0.0), Z, args.batch_size)
        recon = chunked(lambda i: lp.full(R[i], Xf[i]), torch.arange(N * T, device=dev), args.batch_size).view(N, T)
        Z = Z.view(N, T, *Z.shape[1:])
        R = R.view(N, T, *R.shape[1:])
        Xd = Xf.view(N, T, *Xf.shape[1:])

    train = list(range(0, N, 2))
    test = list(range(1, N, 2))
    with torch.no_grad():
        gmap = VelocityMap.fit(Z[train], rank=0)
        rmap = VelocityMap.fit(Z[train], rank=args.map_rank)
    logger.info(f"fitted gain {gmap.gain:.3f}; rank-{args.map_rank} map on {len(train)} clips")

    def dec(z):
        return bb.decode(z, 0.0)

    per_clip = []
    with torch.no_grad():
        for n in range(N):
            z = Z[n]
            step = angle_deg(z[:-1], z[1:])
            turn = turning_deg(z)
            preds = {
                "copy": z[1:-1],
                "extrap": extrapolate(z[:-2], z[1:-1]),
                "gain": torch.cat([gmap.step(z[t - 1:t], z[t:t + 1]) for t in range(1, T - 1)]),
                "map": torch.cat([rmap.step(z[t - 1:t], z[t:t + 1]) for t in range(1, T - 1)]),
            }
            tgt_z, tgt_x = z[2:], Xd[n, 2:]
            row = {
                "clip": n, "kind": meta[n]["kind"], "split": "train" if n in train else "test",
                "recon_lpips": mean(recon[n]),
                "step_deg": mean(step), "step_deg_sd": float(step.std()),
                "turn_deg": mean(turn),
                "oracle_lpips": mean(lp.full(R[n, 2:], tgt_x)),
            }
            for k, zp in preds.items():
                row[f"onestep_{k}_err_deg"] = mean(angle_deg(zp, tgt_z))
                row[f"onestep_{k}_lpips"] = mean(chunked(lambda i: lp.full(dec(zp[i]), tgt_x[i]), torch.arange(zp.shape[0], device=dev), args.batch_size))
            per_clip.append(row)
            run.row(stage="encode_onestep", **row)

    e_hat = None
    if args.snap_alpha > 0:
        g = torch.Generator(device=dev).manual_seed(args.seed)
        e_hat = bb.random_latent(1, dev, generator=g)

    variants = {
        "copy": lambda zp, zc: zc,
        "extrap_raw": lambda zp, zc: extrapolate(zp, zc),
        "extrap_snap": lambda zp, zc: snap(bb, extrapolate(zp, zc), args.snap_alpha, e_hat),
        "map_snap": lambda zp, zc: snap(bb, rmap.step(zp, zc), args.snap_alpha, e_hat),
    }
    roll_rows = []
    with torch.no_grad():
        idx = torch.tensor(test, device=dev)
        for name, fn in variants.items():
            zp, zc = Z[idx, 0], Z[idx, 1]
            frames = [dec(zp), dec(zc)]
            for _ in range(2, T):
                zn = fn(zp, zc)
                zp, zc = zc, zn
                frames.append(dec(zc))
            G = torch.stack(frames, 1)
            lp_t = torch.stack([lp.full(G[:, t], Xd[idx, t]) for t in range(T)], 1)
            flick = torch.stack([lp.full(G[:, t], G[:, t - 1]) for t in range(1, T)], 1)
            true_flick = torch.stack([lp.full(Xd[idx, t], Xd[idx, t - 1]) for t in range(1, T)], 1)
            r = {
                "variant": name,
                "lpips_mean": mean(lp_t[:, 2:]),
                "lpips_by_second": [mean(lp_t[:, int(s * args.fps):int((s + 1) * args.fps)]) for s in range(int(T / args.fps))],
                "lpips_last": mean(lp_t[:, -1]),
                "flicker": mean(flick), "true_flicker": mean(true_flick),
            }
            roll_rows.append(r)
            run.row(stage="rollout", **r)
            torch.save(G[: args.save_videos].half().cpu(), run.path("rollouts", f"{name}.pt"))
        k = min(args.save_videos, len(test))
        if k:
            cols = [Xd[idx[:k]], R[idx[:k]]] + [torch.load(run.path("rollouts", f"{v}.pt")).to(dev).float() for v in ("extrap_raw", "extrap_snap", "map_snap")]
            for j in range(k):
                grid = torch.cat([c[j] for c in cols], dim=-1)
                write_mp4(grid, run.path("videos", f"clip{test[j]:02d}_{meta[test[j]]['kind']}.mp4"), args.fps)
            save_grid([c[0, :: max(1, T // 8)] for c in cols], run.path("figs", "rollout_strip.png"), nrow=len(cols))

    search_rows = []
    if args.search_clips > 0 and args.search_frames > 1:
        sidx = torch.arange(min(args.search_clips, N), device=dev)
        M = min(args.search_frames, T)
        zs, ls = [], []
        z_cur, l_cur = search(bb, Xd[sidx, 0], Z[sidx, 0], lp, steps=args.search_steps, mu=0.0)
        zs.append(z_cur)
        ls.append(l_cur)
        for t in range(1, M):
            z_cur, l_cur = search(bb, Xd[sidx, t], z_cur, lp, steps=args.search_steps, mu=1.0)
            zs.append(z_cur)
            ls.append(l_cur)
        S = torch.stack(zs, 1)
        Ls = torch.stack(ls, 1)
        torch.save({"latents": S.half().cpu(), "lpips": Ls.cpu()}, run.path("search", "searched.pt"))
        with torch.no_grad():
            Ds = torch.stack([dec(S[:, t]) for t in range(M)], 1)
            for j in range(S.shape[0]):
                n = int(sidx[j])
                s = S[j]
                ex = extrapolate(s[:-2], s[1:-1])
                r = {
                    "clip": n, "kind": meta[n]["kind"],
                    "search_lpips": mean(Ls[j]),
                    "search_lpips_by_second": [mean(Ls[j, int(q * args.fps):int((q + 1) * args.fps)]) for q in range(max(1, int(M / args.fps)))],
                    "search_lpips_max": float(Ls[j].max()),
                    "encoder_lpips": mean(recon[n, :M]),
                    "search_step_deg": mean(angle_deg(s[:-1], s[1:])),
                    "encoder_step_deg": mean(angle_deg(Z[n, :M - 1], Z[n, 1:M])),
                    "search_from_first_deg": float(angle_deg(s[:1], s[-1:])),
                    "encoder_from_first_deg": float(angle_deg(Z[n, :1], Z[n, M - 1:M])),
                    "search_vs_encoder_deg": mean(angle_deg(s, Z[n, :M])),
                    "search_turn_deg": mean(turning_deg(s)),
                    "encoder_turn_deg": mean(turning_deg(Z[n, :M])),
                    "search_copy_err_deg": mean(angle_deg(s[1:-1], s[2:])),
                    "search_extrap_err_deg": mean(angle_deg(ex, s[2:])),
                    "search_extrap_lpips": mean(lp.full(dec(ex), Xd[n, 2:M])),
                }
                search_rows.append(r)
                run.row(stage="search", **r)
                if j < args.save_videos:
                    grid = torch.cat([Xd[n, :M], R[n, :M], Ds[j]], dim=-1)
                    write_mp4(grid, run.path("videos", f"search_clip{n:02d}_{meta[n]['kind']}.mp4"), args.fps)

    test_rows = [r for r in per_clip if r["split"] == "test"]
    summary = {
        "n_clips": N, "n_frames": T, "fps": args.fps, "map_gain": gmap.gain,
        "recon_lpips": mean([r["recon_lpips"] for r in per_clip]),
        "step_deg": mean([r["step_deg"] for r in per_clip]),
        "turn_deg": mean([r["turn_deg"] for r in per_clip]),
        "onestep_test": {k: {"err_deg": mean([r[f"onestep_{k}_err_deg"] for r in test_rows]),
                             "lpips": mean([r[f"onestep_{k}_lpips"] for r in test_rows])} for k in ("copy", "extrap", "gain", "map")},
        "oracle_lpips": mean([r["oracle_lpips"] for r in test_rows]),
        "rollout": roll_rows,
        "search": {k: (mean([r[k] for r in search_rows]) if not isinstance(search_rows[0][k], list) else [mean([r[k][q] for r in search_rows]) for q in range(len(search_rows[0][k]))]) for k in (search_rows[0] if search_rows else {}) if k not in ("clip", "kind")},
        "search_per_clip": [{"clip": r["clip"], "kind": r["kind"], "search_lpips": r["search_lpips"], "encoder_lpips": r["encoder_lpips"], "search_step_deg": r["search_step_deg"], "encoder_step_deg": r["encoder_step_deg"]} for r in search_rows],
        "wall_s": round(time.time() - t0, 1),
    }
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    run.finish("ok", summary=summary)
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
