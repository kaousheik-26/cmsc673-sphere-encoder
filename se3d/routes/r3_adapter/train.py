"""
Train the R3 latent adapter.

Stage 1 (latent only): targets are the frozen encoder's tokens E(x_tgt) (per-token
LayerNorm, zero mean / unit variance, so already normalised):
    L = SmoothL1_{beta=0.1}(pred, E(x_tgt)) + (1 - cos(pred, E(x_tgt)))   (per token)
Stage 2 (+ image): additionally decode F(pred) with the frozen SE2 decoder at
a ~ U[0, a_max] with a shared e per object, + LPIPS-vgg + L1 to x_tgt.
Validation: held-out objects, token cosine and (stage 2) decoded LPIPS; plus the
"copy source" baseline (the identity the model is initialised to).
"""
import argparse
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from se3d.eval.metrics import _lpips_net, lpips
from se3d.routes.r3_adapter.data import ObjaverseViews
from se3d.routes.r3_adapter.model import build_adapter
from se3d.se2.wrapper import SE2


def token_loss(pred, tgt):
    l1 = F.smooth_l1_loss(pred, tgt, beta=0.1)
    cos = 1 - F.cosine_similarity(pred, tgt, dim=-1).mean()
    return l1 + cos, cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/fs/nexus-scratch/vinayakg/se3d/data/objaverse_ccby")
    p.add_argument("--n_objects", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--init", default=None, help="stage-1 checkpoint for stage 2")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_src", type=int, default=1)
    p.add_argument("--n_tgt", type=int, default=3)
    p.add_argument("--arch", default="transformer", choices=["transformer", "mixer", "token"],
                   help="transformer = LatentAdapter (PRoPE); mixer / token = small LatentMLP")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--no_prope", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--a_max", type=float, default=40.0)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--img_weight", type=float, default=1.0)
    p.add_argument("--tok_weight", type=float, default=1.0)
    p.add_argument("--a_min", type=float, default=0.0)
    p.add_argument("--l2_weight", type=float, default=0.0)
    p.add_argument("--model", default="large-fdlite-256")
    p.add_argument("--decoder_ckpt", default=None)
    p.add_argument("--train_objects", nargs="*", default=None, help="explicit object folder names under --root")
    p.add_argument("--val_root", default=None, help="UNSEEN objects for validation (e.g. GSO30 ring32 renders)")
    p.add_argument("--val_objects", nargs="*", default=None)
    p.add_argument("--overfit_object", default=None,
                   help="ONE object: views %% 7 == 3 are held-out targets, the rest train")
    p.add_argument("--train_decoder", action="store_true", help="also fine-tune the SE2 decoder (per-scene stage 3)")
    p.add_argument("--dec_lr", type=float, default=1e-5)
    p.add_argument("--val_alphas", type=int, nargs="+", default=[0, 20])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/resume.pt if present (adapter, optimiser, schedule, step, history)")
    p.add_argument("--train_list", default=None, help="file with one object folder (relative to --root, or absolute) per line")
    p.add_argument("--all_views", action="store_true",
                   help="multi-object: train on every view of every object (no per-object held-out views; objects may "
                        "have different view counts); seen-object validation then uses random view pairs")
    p.add_argument("--cam_norm", default="all_views", choices=["all_views", "src_dist"])
    p.add_argument("--save_best", action="store_true",
                   help="also keep adapter_best.pt: lowest LPIPS (best alpha) on the unseen set if given, else on held-out views")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    all_objs = sorted(o for o in os.listdir(args.root) if osp.exists(osp.join(args.root, o, "cameras.json")))
    val_objs = all_objs[:args.n_val]  # the prep order is already a fixed random shuffle
    tr_objs = all_objs[args.n_val:args.n_val + args.n_objects]
    print(f"train objects {len(tr_objs)} val {len(val_objs)}", flush=True)
    json.dump({"args": vars(args), "val_objects": val_objs}, open(osp.join(args.out, "config.json"), "w"), indent=1)

    held_out_rule = lambda nv: [v for v in range(nv) if v % 7 == 3]
    if args.overfit_object:
        args.train_objects = [args.overfit_object]
    if args.train_list:
        args.train_objects = [l.strip() for l in open(args.train_list) if l.strip()]
    if args.train_objects and args.all_views:
        tr_objs = args.train_objects
        ds = ObjaverseViews(args.root, tr_objs, args.n_src, args.n_tgt, seed=args.seed,
                            length=max(len(tr_objs), args.batch * 50), cam_norm=args.cam_norm)
        vds = ObjaverseViews(args.root, tr_objs[::max(1, len(tr_objs) // 16)][:16], args.n_src, args.n_tgt, seed=123,
                             deterministic=True, length=16, cam_norm=args.cam_norm)
        print(f"train objects {len(tr_objs)}: all views", flush=True)
    elif args.train_objects:
        tr_objs = args.train_objects
        nv = min(len(json.load(open(osp.join(args.root, o, "cameras.json")))["frames"]) for o in tr_objs)
        held = held_out_rule(nv)
        train_v = [v for v in range(nv) if v not in held]
        ds = ObjaverseViews(args.root, tr_objs, args.n_src, args.n_tgt, seed=args.seed,
                            src_views=train_v, tgt_views=train_v, length=max(len(tr_objs), args.batch * 50), cam_norm=args.cam_norm)
        # validation A: held-out target views of the TRAINING objects
        vds = ObjaverseViews(args.root, tr_objs[:16], args.n_src, min(args.n_tgt, len(held)), seed=123, deterministic=True,
                             src_views=train_v, tgt_views=held, length=max(16, min(len(tr_objs), 16)), cam_norm=args.cam_norm)
        print(f"train objects {len(tr_objs)}: {nv} views each, held-out target views {held}", flush=True)
    else:
        ds = ObjaverseViews(args.root, tr_objs, args.n_src, args.n_tgt, seed=args.seed, cam_norm=args.cam_norm)
    # validation B: objects never seen in training
    vds_unseen = None
    if args.val_root:
        vo = args.val_objects or sorted(o for o in os.listdir(args.val_root) if osp.exists(osp.join(args.val_root, o, "cameras.json")))
        vds_unseen = ObjaverseViews(args.val_root, vo, args.n_src, args.n_tgt, seed=321, deterministic=True, length=len(vo), cam_norm=args.cam_norm)
        print(f"unseen validation objects: {len(vo)}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    if not args.train_objects:
        vds = ObjaverseViews(args.root, val_objs, args.n_src, args.n_tgt, seed=123, deterministic=True, cam_norm=args.cam_norm)

    se2 = SE2(args.model, decoder_ckpt=args.decoder_ckpt)
    net = build_adapter(vars(args)).cuda()
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    print(f"adapter params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M", flush=True)
    groups = [{"params": list(net.parameters()), "lr": args.lr, "weight_decay": 0.05}]
    if args.train_decoder:
        assert args.stage == 2, "decoder training needs the image loss (stage 2)"
        se2.model.decoder.train().requires_grad_(True)
        groups.append({"params": list(se2.model.decoder.parameters()), "lr": args.dec_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps, pct_start=0.05)
    lp = _lpips_net("vgg", "cuda") if args.stage == 2 else None

    def encode(x):  # x [B,V,3,H,W] in [0,1] -> tokens [B,V,256,384]
        B, V = x.shape[:2]
        with torch.no_grad():
            return se2.encode_tokens(x.flatten(0, 1) * 2 - 1).view(B, V, *se2.latent_shape)

    def step_loss(batch, train=True):
        x = batch["x"].cuda(non_blocking=True)
        K, w2c = batch["K"].cuda(), batch["w2c"].cuda()
        S = args.n_src
        tok = encode(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(tok[:, :S], K[:, :S], w2c[:, :S], K[:, S:], w2c[:, S:])
        pred = pred.float()
        tgt = tok[:, S:]
        loss, cos = token_loss(pred, tgt)
        logs = {"tok_loss": loss.item(), "tok_cos": 1 - cos.item(),
                "copy_cos": F.cosine_similarity(tok[:, :1].expand_as(tgt), tgt, dim=-1).mean().item()}
        if args.stage == 2:
            B, T = tgt.shape[:2]
            z = se2.spherify(pred.flatten(0, 1))
            a = args.a_min + torch.rand(B * T, device="cuda") * (args.a_max - args.a_min)
            e = se2.sample_noise(B).repeat_interleave(T, 0)
            xh = se2.decode(se2.rotate(z, e, a), None, a) * 0.5 + 0.5
            xt = x[:, S:].flatten(0, 1)
            li = lp(xh * 2 - 1, xt * 2 - 1).mean() + (xh - xt).abs().mean()
            if args.l2_weight > 0:
                li = li + args.l2_weight * (xh - xt).square().mean()
            loss = args.tok_weight * loss + args.img_weight * li
            logs["img_loss"] = li.item()
        return loss, logs

    @torch.no_grad()
    def validate_all():
        out = validate(vds, "val_grid.png")
        if vds_unseen is not None:
            out.update({"unseen_" + k: v for k, v in validate(vds_unseen, "val_grid_unseen.png").items()})
        return out

    @torch.no_grad()
    def validate(vset, grid_name):
        net.eval()
        se2.model.decoder.eval()
        acc = {}
        for i in range(0, len(vset), args.batch):
            items = [vset[j] for j in range(i, min(i + args.batch, len(vset)))]
            batch = {k: torch.stack([it[k] for it in items]) for k in ("x", "K", "w2c")}
            _, logs = step_loss(batch, False)
            x = batch["x"].cuda()
            S = args.n_src
            tok = encode(x)
            pred = net(tok[:, :S], batch["K"][:, :S].cuda(), batch["w2c"][:, :S].cuda(),
                       batch["K"][:, S:].cuda(), batch["w2c"][:, S:].cuda()).float()
            dec = {}
            for a in args.val_alphas:
                z = se2.spherify(pred.flatten(0, 1))
                if a:
                    z = se2.rotate(z, se2.sample_noise(z.shape[0], torch.Generator(device="cuda").manual_seed(i)), a)
                xh = se2.to01(se2.decode(z, None, a))
                dec[a] = xh
                logs[f"lpips_a{a}"] = lpips(xh, x[:, S:].flatten(0, 1)).mean().item()
            if i == 0:  # grid: source | pred a0 | pred a20 | ceiling | GT  (one row per target)
                T = x.shape[1] - S
                src = x[:, :1].expand(-1, T, -1, -1, -1).flatten(0, 1)
                ceil = se2.to01(se2.decode(se2.spherify(tok[:, S:].flatten(0, 1)), None, 0.0))
                best_a = min(args.val_alphas, key=lambda a: logs[f"lpips_a{a}"])
                g = torch.stack([src, dec[args.val_alphas[0]], dec[best_a], ceil, x[:, S:].flatten(0, 1)], 1)[:12].flatten(0, 1)
                save_image(g, osp.join(args.out, grid_name), nrow=5)
            zc = se2.spherify(tok[:, :1].expand(-1, tok.shape[1] - S, -1, -1).flatten(0, 1))
            logs["lpips_copy_src"] = lpips(se2.to01(se2.decode(zc, None, 0.0)), x[:, S:].flatten(0, 1)).mean().item()
            # ceiling: decode the target's own encoding (best any latent predictor of E(x_tgt) can do)
            zt = se2.spherify(tok[:, S:].flatten(0, 1))
            logs["lpips_ceiling"] = lpips(se2.to01(se2.decode(zt, None, 0.0)), x[:, S:].flatten(0, 1)).mean().item()
            for k, v in logs.items():
                acc.setdefault(k, []).append((v, len(items)))  # weight by batch size (last batch is smaller)
        net.train()
        if args.train_decoder:
            se2.model.decoder.train()
        return {k: float(sum(a * n for a, n in v) / sum(n for _, n in v)) for k, v in acc.items()}

    rpath = osp.join(args.out, "resume.pt")
    if args.resume and osp.exists(rpath):  # SLURM requeue after preemption: continue where we stopped
        ck = torch.load(rpath, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, hist, best_score = ck["step"], ck["hist"], ck["best_score"]
        print(f"resumed from step {step}", flush=True)
    else:
        hist = [{"step": 0, **validate_all()}]
        print("val@0", {k: round(v, 4) for k, v in hist[-1].items()}, flush=True)
        step = 0
        best_score = float("inf")
    t0 = time.time()
    while step < args.steps:
        for batch in dl:
            loss, logs = step_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items()) + f" {time.time() - t0:.0f}s", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                hist.append({"step": step, **validate_all()})
                print(f"val@{step}", {k: round(v, 4) for k, v in hist[-1].items()}, flush=True)
                torch.save(net.state_dict(), osp.join(args.out, "adapter_last.pt"))
                if args.save_best:
                    pre = "unseen_" if vds_unseen is not None else ""
                    score = min(hist[-1][f"{pre}lpips_a{a}"] for a in args.val_alphas)
                    if score < best_score:
                        best_score = score
                        torch.save(net.state_dict(), osp.join(args.out, "adapter_best.pt"))
                        json.dump({"step": step, "lpips": score}, open(osp.join(args.out, "best.json"), "w"))
                if args.train_decoder:
                    torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in se2.model.decoder.state_dict().items()},
                               osp.join(args.out, "decoder_last.pt"))
                json.dump(hist, open(osp.join(args.out, "val_history.json"), "w"), indent=1)
                if args.resume:
                    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "step": step, "hist": hist, "best_score": best_score}, rpath + ".tmp")
                    os.replace(rpath + ".tmp", rpath)
            if step >= args.steps:
                break
    print("R3_TRAIN_DONE")


if __name__ == "__main__":
    main()
