import argparse
import json
import logging
import math
import os
import os.path as osp
import signal
import sys
import time

import torch

from halfsphere.backbone import SE2Backbone
from halfsphere.geometry import angle_deg
from halfsphere.records import Run
from spherevideo.clips import video_clips
from spherevideo.rotnet import RotNet, compose, dir_cos, split_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Long, resumable rotation model training with periodic train/test angle, distance and direction tracking")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-imagenet-256px")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--train_videos", type=str, default="workspace/videos/wan21_t2v_1.3b/*/")
parser.add_argument("--test_videos", type=str, default="workspace/videos/wan21_t2v_1.3b_test/*/")
parser.add_argument("--n_frames", type=int, default=81)
parser.add_argument("--fps", type=float, default=16.0)
parser.add_argument("--src_fps", type=float, default=16.0)
parser.add_argument("--n_train", type=int, default=0)
parser.add_argument("--init", type=str, default="")
parser.add_argument("--hint", type=str, default="none", choices=["none", "two"])
parser.add_argument("--width", type=int, default=384)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--heads", type=int, default=6)
parser.add_argument("--amp", type=int, default=0)
parser.add_argument("--angle_w", type=float, default=1.0)
parser.add_argument("--dir_w", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=200000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup", type=int, default=500)
parser.add_argument("--eval_every", type=int, default=1000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--keep_every", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--signal", type=str, default="")
parser.add_argument("--signal_every", type=float, default=30.0)
args = parser.parse_args()

FLAGS = {"save": False, "stop": False, "yield": False}


def gpu_allowed():
    if not args.signal:
        return True
    try:
        return bool(json.load(open(args.signal)).get("gpu_allowed", False))
    except Exception:
        return False


def on_signal(sig, frame):
    FLAGS["save"] = True
    FLAGS["stop"] = FLAGS["stop"] or sig == signal.SIGTERM


def encode_clips(bb, X, dev, bs=32):
    N, T = X.shape[:2]
    flat = X.flatten(0, 1)
    with torch.no_grad():
        Z = torch.cat([bb.encode(flat[i:i + bs].to(dev)) for i in range(0, flat.shape[0], bs)])
    return Z.view(N, T, *Z.shape[1:])


def latents(dev):
    p = osp.join(args.out, "latents.pt")
    if osp.exists(p):
        d = torch.load(p, map_location="cpu")
        return d["train"].to(dev), d["test"].to(dev), d["train_meta"], d["test_meta"], d["ckpt_path"]
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    Xtr, mtr = video_clips(args.train_videos, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
    Xte, mte = video_clips(args.test_videos, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
    Ztr, Zte = encode_clips(bb, Xtr, dev), encode_clips(bb, Xte, dev)
    torch.save({"train": Ztr.cpu(), "test": Zte.cpu(), "train_meta": mtr, "test_meta": mte, "ckpt_path": bb.ckpt_path}, p)
    ck = bb.ckpt_path
    del bb, Xtr, Xte
    torch.cuda.empty_cache()
    return Ztr, Zte, mtr, mte, ck


def track(net, Z, dev):
    T = Z.shape[1]
    t = torch.arange(T, device=dev).float() / (T - 1)
    net.eval()
    per, amt, off = [], [], []
    with torch.no_grad():
        for n in range(Z.shape[0]):
            z0 = Z[n, 0][None].expand(T, *Z.shape[2:])
            z1 = Z[n, 1][None].expand(T, *Z.shape[2:])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                th, d = net(z0, t, z1)
            th, d = th.float(), d.float()
            th_true, d_true = split_target(z0[1:], Z[n, 1:])
            per.append(float(angle_deg(compose(z0, th, d), Z[n])[1:].mean()))
            amt.append(float((th[1:] - th_true).abs().mean()))
            off.append(float(torch.rad2deg(torch.acos(dir_cos(d[1:], d_true).clamp(-1, 1))).mean()))
    net.train()
    m = lambda v: sum(v) / len(v)
    return m(per), per, m(amt), m(off)


def save_state(net, opt, sched, g, step, best):
    p = osp.join(args.out, "state.pt")
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": g.get_state(), "step": step, "best": best, "args": vars(args)}, p + ".tmp")
    os.replace(p + ".tmp", p)


def save_model(net, name):
    d = osp.join(args.out, "ckpt", name)
    os.makedirs(d, exist_ok=True)
    torch.save({"net": net.state_dict(), "args": vars(args)}, osp.join(d, "rotnet.pt"))


def main():
    os.makedirs(args.out, exist_ok=True)
    if not gpu_allowed():
        logger.info("resource signal off: not starting")
        sys.exit(3)
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    Ztr, Zte, mtr, mte, ck = latents(dev)
    if args.n_train:
        Ztr, mtr = Ztr[:args.n_train], mtr[:args.n_train]
    N, T = Ztr.shape[:2]
    run = Run("launches", {"args": vars(args), "train_clips": mtr, "test_clips": mte}, dev_dir=args.out, ckpt_path=ck, seeds={"base": args.seed})
    net = RotNet(token_dim=Ztr.shape[-1], n_tokens=Ztr.shape[-2], width=args.width, depth=args.depth, heads=args.heads, hint=args.hint).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    step, best = 0, {"test_err": 1e9, "step": -1}
    sp = osp.join(args.out, "state.pt")
    if osp.exists(sp):
        s = torch.load(sp, map_location="cpu", weights_only=False)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        g.set_state(s["gen"])
        step, best = s["step"], s["best"]
        logger.info(f"resumed at step {step}")
    elif args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["net"])
        logger.info(f"initialised from {args.init}")
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"RotNet hint {args.hint}: {n_params / 1e6:.2f}M params; {N} train clips x {T - 1} frames; {Zte.shape[0]} test clips; step {step}/{args.steps}")
    log_path = osp.join(args.out, "log.jsonl")
    t0, s0 = time.time(), step
    last_sig = time.time()
    while step < args.steps:
        i = torch.randint(0, N, (args.batch_size,), generator=g)
        t = torch.randint(1, T, (args.batch_size,), generator=g)
        z0, z1, zt = Ztr[i, 0], Ztr[i, 1], Ztr[i, t]
        with torch.no_grad():
            th_true, d_true = split_target(z0, zt)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            th, d = net(z0, t.to(dev).float() / (T - 1), z1)
        l_ang = ((th.float() - th_true).abs() / net.angle_scale).mean()
        l_dir = (1 - dir_cos(d.float(), d_true)).mean()
        loss = args.angle_w * l_ang + args.dir_w * l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            tr, _, tr_amt, tr_off = track(net, Ztr, dev)
            te, per, te_amt, te_off = track(net, Zte, dev)
            if te < best["test_err"]:
                best = {"test_err": te, "train_err": tr, "step": step}
                save_model(net, "best_test")
            rec = {"step": step, "loss": loss.item(), "train_err": tr, "test_err": te, "train_amount_err": tr_amt, "test_amount_err": te_amt, "train_dir_off": tr_off, "test_dir_off": te_off, "test_per_clip": [round(v, 2) for v in per],
                   "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1), "steps_per_s": round((step - s0) / (time.time() - t0), 2)}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            logger.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "test_per_clip"}))
        if step % args.keep_every == 0 or step == args.steps:
            save_model(net, f"step_{step:07d}")
        if args.signal and time.time() - last_sig >= args.signal_every:
            last_sig = time.time()
            if not gpu_allowed():
                FLAGS["save"] = FLAGS["stop"] = FLAGS["yield"] = True
        if osp.exists(osp.join(args.out, "SAVE")):
            os.remove(osp.join(args.out, "SAVE"))
            FLAGS["save"] = True
        if osp.exists(osp.join(args.out, "STOP")):
            os.remove(osp.join(args.out, "STOP"))
            FLAGS["save"] = FLAGS["stop"] = True
        if step % args.ckpt_every == 0 or FLAGS["save"] or step == args.steps:
            save_state(net, opt, sched, g, step, best)
            if FLAGS["save"]:
                save_model(net, f"step_{step:07d}")
                logger.info(f"saved at step {step}")
            FLAGS["save"] = False
        if FLAGS["stop"]:
            run.finish("paused_signal" if FLAGS["yield"] else "paused", step=step, best=best)
            logger.info(f"paused at step {step}" + (" (resource signal off)" if FLAGS["yield"] else ""))
            sys.exit(3 if FLAGS["yield"] else 0)
    save_model(net, "final")
    run.finish("ok", step=step, best=best)
    print(json.dumps({"step": step, "best": best}))


if __name__ == "__main__":
    main()
