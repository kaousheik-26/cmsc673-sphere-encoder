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
from spherevideo.rotnet import RotNet, compose, dir_cos, split_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train a rotation regressor z_t = rotate(z_0, theta(z_0, t), d(z_0, t)) on the SE2 sphere")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-imagenet-256px")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--train_videos", type=str, default="workspace/videos/wan21_t2v_1.3b/*/")
parser.add_argument("--test_videos", type=str, default="workspace/videos/wan21_t2v_1.3b_test/*/")
parser.add_argument("--n_frames", type=int, default=81)
parser.add_argument("--fps", type=float, default=16.0)
parser.add_argument("--src_fps", type=float, default=16.0)
parser.add_argument("--hint", type=str, default="none", choices=["none", "two"])
parser.add_argument("--width", type=int, default=384)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--angle_w", type=float, default=1.0)
parser.add_argument("--dir_w", type=float, default=1.0)
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


def decode(bb, Z, bs=27):
    return torch.cat([bb.decode(Z[i:i + bs], 0.0) for i in range(0, Z.shape[0], bs)])


def lpips(lp, a, b, bs=27):
    return torch.cat([lp.full(a[i:i + bs], b[i:i + bs]) for i in range(0, a.shape[0], bs)])


def evaluate(bb, lp, net, X, Z, meta, curve, run, split, dev, save_k):
    N, T = X.shape[:2]
    t0 = 2
    tt = torch.arange(t0, T, device=dev)
    secs = max(1, int(T / args.fps))
    rows = []
    net.eval()
    with torch.no_grad():
        for n in range(N):
            k = tt.shape[0]
            z0 = Z[n, 0][None].expand(k, *Z.shape[2:])
            z1 = Z[n, 1][None].expand(k, *Z.shape[2:])
            zt = Z[n, t0:]
            th_true, d_true = split_target(z0, zt)
            th_pred, d_pred = net(z0, tt.float() / (T - 1), z1)
            th_curve = curve[t0:]
            pred = compose(z0, th_pred, d_pred)
            o_amount = compose(z0, th_true, d_pred)
            o_dir = compose(z0, th_pred, d_true)
            cosd = dir_cos(d_pred, d_true).clamp(-1, 1)
            rnd = torch.rad2deg(torch.acos(torch.cos(torch.deg2rad(th_true)) ** 2))
            x = X[n, t0:].to(dev)
            dp, dc, do = decode(bb, pred), bb.decode(Z[n, :1], 0.0).expand(k, -1, -1, -1), decode(bb, o_dir)
            lp_pred, lp_copy, lp_odir = lpips(lp, dp, x), lpips(lp, dc, x), lpips(lp, do, x)
            by = lambda v: [mean(v[max(0, int(s * args.fps) - t0 + 1):int((s + 1) * args.fps) - t0 + 1]) for s in range(secs)]
            r = {"split": split, "clip": n, "kind": meta[n]["kind"],
                 "true_angle": mean(th_true), "amount_err": mean((th_pred - th_true).abs()), "curve_amount_err": mean((th_curve - th_true).abs()),
                 "dir_err_deg": mean(torch.rad2deg(torch.acos(cosd))), "dir_cos": mean(cosd),
                 "pred_err_deg": mean(angle_deg(pred, zt)), "copy_err_deg": mean(th_true), "random_dir_err_deg": mean(rnd),
                 "oracle_amount_err_deg": mean(angle_deg(o_amount, zt)), "oracle_dir_err_deg": mean(angle_deg(o_dir, zt)),
                 "pred_lpips": mean(lp_pred), "copy_lpips": mean(lp_copy), "oracle_dir_lpips": mean(lp_odir),
                 "true_angle_by_second": by(th_true), "pred_angle_by_second": by(th_pred), "dir_err_by_second": by(torch.rad2deg(torch.acos(cosd))),
                 "pred_lpips_by_second": by(lp_pred), "copy_lpips_by_second": by(lp_copy), "oracle_dir_lpips_by_second": by(lp_odir)}
            rows.append(r)
            run.row(stage=f"eval_{split}", **r)
            if n < save_k:
                write_mp4(torch.cat([x, dc, dp, do], dim=-1), run.path("videos", f"{split}_clip{n:02d}_{meta[n]['kind']}.mp4"), args.fps)
    keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    agg = {k: mean([r[k] for r in rows]) for k in keys}
    for k in [k for k, v in rows[0].items() if isinstance(v, list)]:
        agg[k] = [mean([r[k][s] for r in rows]) for s in range(len(rows[0][k]))]
    agg["pred_beats_copy"] = sum(r["pred_lpips"] < r["copy_lpips"] for r in rows)
    agg["oracle_dir_beats_copy"] = sum(r["oracle_dir_lpips"] < r["copy_lpips"] for r in rows)
    agg["n"] = len(rows)
    agg["per_clip"] = [{k: r[k] for k in ("kind", "true_angle", "amount_err", "curve_amount_err", "dir_err_deg", "pred_err_deg", "copy_err_deg",
                                          "oracle_dir_err_deg", "pred_lpips", "copy_lpips", "oracle_dir_lpips")} for r in rows]
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
    N, T = Ztr.shape[:2]
    with torch.no_grad():
        TH = torch.stack([split_target(Ztr[n, :1].expand_as(Ztr[n]), Ztr[n])[0] for n in range(N)])
    curve = TH.mean(0)
    run = Run("rotnet", {"args": vars(args), "train_clips": mtr, "test_clips": mte}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path, seeds={"base": args.seed})
    net = RotNet(token_dim=Ztr.shape[-1], n_tokens=Ztr.shape[-2], width=args.width, depth=args.depth, hint=args.hint).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"RotNet hint={args.hint}: {n_params / 1e6:.2f}M params; {N} train clips x {T - 1} frames; {Xte.shape[0]} test clips")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    log = []
    for step in range(args.steps):
        net.train()
        i = torch.randint(0, N, (args.batch_size,), generator=g)
        t = torch.randint(1, T, (args.batch_size,), generator=g)
        z0, z1, zt = Ztr[i, 0], Ztr[i, 1], Ztr[i, t]
        with torch.no_grad():
            th_true, d_true = split_target(z0, zt)
        th, d = net(z0, t.to(dev).float() / (T - 1), z1)
        l_ang = ((th - th_true).abs() / net.angle_scale).mean()
        l_dir = (1 - dir_cos(d, d_true)).mean()
        loss = args.angle_w * l_ang + args.dir_w * l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                full = mean(angle_deg(compose(z0, th, d), zt))
            rec = {"step": step, "loss": float(loss), "amount_err": float(l_ang) * net.angle_scale, "dir_cos": 1 - float(l_dir), "err_deg": full, "s": round(time.time() - t0, 1)}
            log.append(rec)
            run.row(stage="train", **rec)
            logger.info(json.dumps(rec))
    torch.save({"net": net.state_dict(), "args": vars(args)}, run.path("rotnet.pt"))
    train_eval = evaluate(bb, lp, net, Xtr, Ztr, mtr, curve, run, "train", dev, min(4, args.save_videos))
    test_eval = evaluate(bb, lp, net, Xte, Zte, mte, curve, run, "test", dev, args.save_videos)
    summary = {"params_m": round(n_params / 1e6, 2), "hint": args.hint, "train_seconds": log[-1]["s"], "final_train": log[-1], "train": train_eval, "test": test_eval}
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    short = {s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[s].items() if k != "per_clip" and not isinstance(v, list)} for s in ("train", "test")}
    print(json.dumps(short, indent=2))
    run.finish("ok", summary=short)
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
