import argparse
import json
import logging
import math
import time

import torch

from halfsphere.backbone import SE2Backbone
from halfsphere.geometry import angle_deg
from halfsphere.metrics import HalfLPIPS
from halfsphere.records import Run
from spherevideo.clips import video_clips, write_mp4
from spherevideo.framenet import FrameNet, sphere_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train a frame predictor z_t = g(z_0, t) on the SE2 sphere")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-imagenet-256px")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--train_videos", type=str, default="workspace/videos/wan21_t2v_1.3b/*/")
parser.add_argument("--test_videos", type=str, default="workspace/videos/wan21_t2v_1.3b_test/*/")
parser.add_argument("--n_frames", type=int, default=81)
parser.add_argument("--fps", type=float, default=16.0)
parser.add_argument("--src_fps", type=float, default=16.0)
parser.add_argument("--target", type=str, default="enc", choices=["enc", "search"])
parser.add_argument("--search_pt", type=str, default="")
parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "mlp"])
parser.add_argument("--width", type=int, default=384)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--heads", type=int, default=6)
parser.add_argument("--init", type=str, default="")
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lpips_w", type=float, default=0.0)
parser.add_argument("--lpips_batch", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_videos", type=int, default=16)
args = parser.parse_args()


def mean(t):
    return float(torch.as_tensor(t, dtype=torch.float32).mean())


def encode_clips(bb, X, dev, bs=32):
    N, T = X.shape[:2]
    flat = X.flatten(0, 1)
    with torch.no_grad():
        Z = torch.cat([bb.encode(flat[i:i + bs].to(dev)) for i in range(0, flat.shape[0], bs)])
    return Z.view(N, T, *Z.shape[1:])


def predict_all(net, z0, T, dev, bs=81):
    t = torch.arange(T, device=dev).float() / (T - 1)
    out = []
    for i in range(0, T, bs):
        tt = t[i:i + bs]
        out.append(net(z0[None].expand(tt.shape[0], *z0.shape), tt))
    return torch.cat(out)


def evaluate(bb, lp, net, X, Z, meta, run, split, dev, save_k):
    N, T = X.shape[:2]
    secs = max(1, int(T / args.fps))
    rows = []
    net.eval()
    with torch.no_grad():
        for n in range(N):
            P = predict_all(net, Z[n, 0], T, dev)
            x = X[n].to(dev)
            dp = torch.cat([bb.decode(P[i:i + 27], 0.0) for i in range(0, T, 27)])
            dc = bb.decode(Z[n, :1], 0.0).expand(T, -1, -1, -1)
            de = torch.cat([bb.decode(Z[n, i:i + 27], 0.0) for i in range(0, T, 27)])
            lp_pred = torch.cat([lp.full(dp[i:i + 27], x[i:i + 27]) for i in range(0, T, 27)])
            lp_copy = torch.cat([lp.full(dc[i:i + 27], x[i:i + 27]) for i in range(0, T, 27)])
            lp_enc = torch.cat([lp.full(de[i:i + 27], x[i:i + 27]) for i in range(0, T, 27)])
            ang_pred = angle_deg(P, Z[n])
            ang_copy = angle_deg(Z[n, :1].expand_as(Z[n]), Z[n])
            by = lambda v: [mean(v[1:][int(s * args.fps):int((s + 1) * args.fps)]) for s in range(secs)]
            r = {"split": split, "clip": n, "kind": meta[n]["kind"],
                 "pred_lpips": mean(lp_pred[1:]), "copy_lpips": mean(lp_copy[1:]), "redraw_lpips": mean(lp_enc[1:]),
                 "pred_lpips_by_second": by(lp_pred), "copy_lpips_by_second": by(lp_copy),
                 "pred_err_deg": mean(ang_pred[1:]), "copy_err_deg": mean(ang_copy[1:]),
                 "pred_flicker": mean(torch.stack([lp.full(dp[t:t + 1], dp[t - 1:t]) for t in range(1, T)])),
                 "true_flicker": mean(torch.stack([lp.full(x[t:t + 1], x[t - 1:t]) for t in range(1, T)]))}
            rows.append(r)
            run.row(stage=f"eval_{split}", **r)
            if n < save_k:
                write_mp4(torch.cat([x, dc, dp], dim=-1), run.path("videos", f"{split}_clip{n:02d}_{meta[n]['kind']}.mp4"), args.fps)
    keys = ["pred_lpips", "copy_lpips", "redraw_lpips", "pred_err_deg", "copy_err_deg", "pred_flicker", "true_flicker"]
    agg = {k: mean([r[k] for r in rows]) for k in keys}
    for k in ("pred_lpips_by_second", "copy_lpips_by_second"):
        agg[k] = [mean([r[k][s] for r in rows]) for s in range(len(rows[0][k]))]
    agg["pred_beats_copy"] = sum(r["pred_lpips"] < r["copy_lpips"] for r in rows)
    agg["n"] = len(rows)
    agg["per_clip"] = [{k: r[k] for k in ("kind", "pred_lpips", "copy_lpips", "redraw_lpips", "pred_err_deg", "copy_err_deg")} for r in rows]
    return agg


def main():
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    lp = HalfLPIPS(dev)
    Xtr, mtr = video_clips(args.train_videos, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
    Xte, mte = video_clips(args.test_videos, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
    Ztr = encode_clips(bb, Xtr, dev)
    Zte = encode_clips(bb, Xte, dev)
    Ytr = Ztr
    if args.target == "search":
        S = torch.load(args.search_pt)["latents"].float().to(dev)
        assert S.shape[:2] == Ztr.shape[:2], f"searched latents {tuple(S.shape)} do not match train clips {tuple(Ztr.shape)}"
        Ytr = S
    N, T = Ztr.shape[:2]
    run = Run("framenet", {"args": vars(args), "train_clips": mtr, "test_clips": mte}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path, seeds={"base": args.seed})
    net = FrameNet(token_dim=Ztr.shape[-1], n_tokens=Ztr.shape[-2], width=args.width, depth=args.depth, heads=args.heads, arch=args.arch).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["net"])
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"FrameNet {args.arch}: {n_params / 1e6:.2f}M params; {N} train clips x {T - 1} frames; {Xte.shape[0]} test clips")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / max(1, args.steps))))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    log = []
    for step in range(args.steps):
        net.train()
        i = torch.randint(0, N, (args.batch_size,), generator=g)
        t = torch.randint(1, T, (args.batch_size,), generator=g)
        z0 = Ztr[i, 0]
        y = Ytr[i, t]
        pred = net(z0, t.to(dev).float() / (T - 1))
        l_sph = sphere_loss(pred, y).mean()
        loss = l_sph
        l_lp = torch.zeros((), device=dev)
        if args.lpips_w > 0:
            k = args.lpips_batch
            x_hat = bb.decode(pred[:k], 0.0)
            l_lp = lp.fn(x_hat, Xtr[i[:k], t[:k]].to(dev)).mean()
            loss = loss + args.lpips_w * l_lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                err = mean(angle_deg(pred, y))
            rec = {"step": step, "loss": float(loss), "sphere": float(l_sph), "lpips": float(l_lp), "err_deg": err, "s": round(time.time() - t0, 1)}
            log.append(rec)
            run.row(stage="train", **rec)
            logger.info(json.dumps(rec))
    torch.save({"net": net.state_dict(), "args": vars(args)}, run.path("framenet.pt"))
    train_eval = evaluate(bb, lp, net, Xtr, Ztr, mtr, run, "train", dev, min(4, args.save_videos))
    test_eval = evaluate(bb, lp, net, Xte, Zte, mte, run, "test", dev, args.save_videos)
    summary = {"params_m": round(n_params / 1e6, 2), "train_seconds": log[-1]["s"] if log else 0.0, "final_train": log[-1] if log else None, "train": train_eval, "test": test_eval}
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    short = {s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[s].items() if k != "per_clip"} for s in ("train", "test")}
    print(json.dumps(short, indent=2))
    run.finish("ok", summary={s: short[s] for s in short})
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
