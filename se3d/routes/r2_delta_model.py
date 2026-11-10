"""
Learn the latent move for a camera change and apply it to UNSEEN objects.
Data: the cache of se3d/routes/r2_pattern.py (GSO-300 orbits: z0 = F(E(x_src)), searched
z* and the encoder's z_gt = F(E(x_tgt)) for 11 azimuth steps, cameras, target images).
Model: LatentMoveNet (r2_move_net.py: PRoPE, Plucker rays, zero-init head = starts as "copy z0"),
input z0 tokens + source/target cameras (relative to the source, source at distance 1),
output target tokens -> spherify. Frozen SE2 decoder for the image loss / evaluation.
Targets: --target zstar (the searched nudge) | zgt (the encoder's latent of the true view).
Split: the same 40 unseen test objects as r2_pattern P7 (seed 0); --overfit_one: train
and test on ONE object first (fit check).
Reports decode LPIPS vs the true target view per azimuth step: model, identity (copy),
oracle z*, ceiling z_gt.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F

from se3d.eval.metrics import _lpips_net, lpips
from se3d.geometry.cameras import normalise_cameras
from se3d.routes.r2_move_net import LatentMoveNet
from se3d.se2.wrapper import SE2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/fs/nexus-scratch/vinayakg/se3d/results/r2_pattern200/latents.pt")
    p.add_argument("--target", choices=["zstar", "zgt"], default="zstar")
    p.add_argument("--img_weight", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=4, help="objects per step (all azimuth steps each)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_test", type=int, default=40)
    p.add_argument("--overfit_one", action="store_true")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    c = torch.load(args.cache)
    O, deltas = len(c["objs"]), c["deltas"]
    se2 = SE2("large-fdlite-256")
    T, C = se2.latent_shape
    z0 = c["z0"].float().view(O, T, C).cuda()
    tgt = c[args.target].float().view(O, len(deltas), T, C).cuda()
    zs = c["zstar"].float().view(O, len(deltas), T, C).cuda()
    zg = c["zgt"].float().view(O, len(deltas), T, C).cuda()
    xt = c["x_tgt"].cuda()  # [O,Dn,3,H,W] half
    W, K = c["w2c"].numpy(), c["K"].float().cuda()  # [O,1+Dn,4,4], [O,1+Dn,3,3]
    rel = torch.tensor(np.stack([normalise_cameras(W[o], 0, mode="src_dist")[0] for o in range(O)]), dtype=torch.float32).cuda()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(O, generator=g).tolist()
    te, tr = perm[:args.n_test], perm[args.n_test:]
    if args.overfit_one:
        te = tr = [tr[0]]
    net = LatentMoveNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    lp = _lpips_net("vgg", "cuda") if args.img_weight > 0 else None

    def forward(idx):
        idx = torch.tensor(idx, device="cuda")
        return net(z0[idx][:, None], K[idx, :1], rel[idx, :1], K[idx, 1:], rel[idx, 1:]).float()  # [B,Dn,T,C]

    @torch.no_grad()
    def dec_lp(z, gt):  # [N,T,C] latents vs [N,3,H,W] images -> per-image LPIPS, in chunks (memory)
        return torch.cat([lpips(se2.to01(se2.decode(se2.spherify(z[i:i + 16]), None, 0.0)), gt[i:i + 16]).flatten()
                          for i in range(0, len(z), 16)])

    @torch.no_grad()
    def evaluate(idx, with_baselines=False):
        net.eval()
        pr = torch.cat([forward(idx[i:i + 4]) for i in range(0, len(idx), 4)])  # [n,Dn,T,C]
        gt = xt[idx].flatten(0, 1).float()
        res = {"model": dec_lp(pr.flatten(0, 1), gt).view(len(idx), -1).mean(0)}
        if with_baselines:
            res["identity"] = dec_lp(z0[idx][:, None].expand(-1, len(deltas), -1, -1).flatten(0, 1), gt).view(len(idx), -1).mean(0)
            res["oracle_zstar"] = dec_lp(zs[idx].flatten(0, 1), gt).view(len(idx), -1).mean(0)
            res["ceiling_zgt"] = dec_lp(zg[idx].flatten(0, 1), gt).view(len(idx), -1).mean(0)
        net.train()
        return {k: {"mean": v.mean().item(), "by_delta": dict(zip(deltas, v.tolist()))} for k, v in res.items()}

    base = evaluate(te, True)
    print("baselines (test):", {k: round(v["mean"], 4) for k, v in base.items()}, flush=True)
    hist = [{"step": 0, "test": base["model"]["mean"]}]
    best = (9.0, 0)
    for step in range(1, args.steps + 1):
        idx = [tr[i] for i in torch.randint(len(tr), (min(args.batch, len(tr)),)).tolist()]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = forward(idx)
        pr = pr.float()
        y = tgt[torch.tensor(idx, device="cuda")]
        ps = se2.spherify(pr.flatten(0, 1))
        loss = F.smooth_l1_loss(ps, y.flatten(0, 1), beta=0.1) + (1 - F.cosine_similarity(ps, y.flatten(0, 1), dim=-1).mean())
        if lp is not None:
            sub = torch.randperm(ps.shape[0], device="cuda")[:8]  # decode a subset (memory)
            xh = se2.decode(ps[sub], None, 0.0) * 0.5 + 0.5
            xg = xt[torch.tensor(idx, device="cuda")].flatten(0, 1)[sub].float()
            loss = loss + args.img_weight * (lp(xh * 2 - 1, xg * 2 - 1).mean() + (xh - xg).abs().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.val_every == 0 or step == args.steps:
            ev = evaluate(te)
            trv = evaluate(tr[:40])
            hist.append({"step": step, "loss": loss.item(), "test": ev["model"]["mean"], "train": trv["model"]["mean"]})
            print(f"step {step} loss {loss.item():.4f} train-obj LPIPS {trv['model']['mean']:.4f} test-obj LPIPS {ev['model']['mean']:.4f}", flush=True)
            if ev["model"]["mean"] < best[0]:
                best = (ev["model"]["mean"], step)
                torch.save(net.state_dict(), osp.join(args.out, "model_best.pt"))
                json.dump({"step": step, "test": ev, "baselines": base}, open(osp.join(args.out, "best.json"), "w"), indent=1)
            json.dump(hist, open(osp.join(args.out, "history.json"), "w"), indent=1)
    print("BEST", best, "baselines", {k: round(v["mean"], 4) for k, v in base.items()})
    print("R2_DELTA_DONE")


if __name__ == "__main__":
    main()
