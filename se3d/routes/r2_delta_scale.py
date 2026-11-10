"""
Learn the latent move for a camera change from the large latent-search run (r2_search_scale.py
parts: z0, searched z*, encoder z_gt, cameras) and apply it to UNSEEN objects.
Model: build_move_net (r2_move_net.py: mixer / token MLP / transformer) on z0 tokens + source/target cameras
(relative to the source, source at distance 1) -> target tokens -> spherify.
Loss: token SmoothL1 + cosine to the target latent (z* or z_gt), optional image loss through
the frozen decoder against the true target view. Split: last --n_test objects are unseen.
Eval: decode LPIPS vs the true target view on --n_eval unseen objects (all camera steps):
model, identity z0, oracle z*, ceiling z_gt.
"""
import argparse
import glob
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F

from se3d.data.images import load_rgba_white
from se3d.eval.metrics import _lpips_net, lpips
from se3d.geometry.cameras import normalise_cameras
from se3d.routes.r2_move_net import build_move_net
from se3d.se2.wrapper import SE2


def load_parts(d):
    P = [torch.load(f, weights_only=False) for f in sorted(glob.glob(osp.join(d, "part*.pt")), key=lambda f: int(f.split("part")[-1][:-3]))]
    cat = lambda k: torch.cat([p[k] for p in P])
    objs = sum([p["objs"] for p in P], [])
    return {"objs": objs, "tgt_views": P[0]["tgt_views"], "z0": cat("z0"), "zstar": cat("zstar"), "zgt": cat("zgt"),
            "w2c": cat("w2c"), "K": cat("K"), "lpips_zstar": cat("lpips_zstar"), "lpips_ceiling": cat("lpips_ceiling")}


def load_many(dirs):
    """merge several search runs over the SAME objects (same list / part split): camera steps concatenated."""
    cs = [load_parts(d) for d in dirs]
    c = cs[0]
    for o in cs[1:]:
        assert o["objs"] == c["objs"], "search runs must cover the same objects in the same order"
        c["tgt_views"] = c["tgt_views"] + o["tgt_views"]
        for k in ("zstar", "zgt", "lpips_zstar", "lpips_ceiling"):
            c[k] = torch.cat([c[k], o[k]], 1)
        for k in ("w2c", "K"):
            c[k] = torch.cat([c[k], o[k][:, 1:]], 1)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["/fs/nexus-projects/genwildsplat/vinayakg/se3d/r2_scale10k"])
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--resume", action="store_true", help="continue from <out>/resume.pt if present")
    p.add_argument("--arch", default="mixer", choices=["mixer", "token", "transformer"])
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--target", choices=["zstar", "zgt"], default="zstar")
    p.add_argument("--img_weight", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=8, help="objects per step (all camera steps each)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--max_train", type=int, default=0, help=">0: use only this many training objects (data curve)")
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--init", default=None, help="start from this state dict")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    c = load_many(args.parts)
    O, V = len(c["objs"]), len(c["tgt_views"])
    print(f"{O} objects x {V} camera steps", flush=True)
    se2 = SE2("large-fdlite-256")
    T, C = se2.latent_shape
    W = c["w2c"].numpy()
    rel = torch.tensor(np.stack([normalise_cameras(W[o], 0, mode="src_dist")[0] for o in range(O)]), dtype=torch.float32)
    K = c["K"].float()
    te = list(range(O - args.n_test, O))
    tr = list(range(O - args.n_test))
    if args.max_train:
        tr = tr[:args.max_train]
    ev = te[:args.n_eval]
    tgt = c[args.target]
    a = {"arch": args.arch, "width": args.width, "depth": args.depth, "heads": args.heads}
    net = build_move_net(a).cuda()
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    json.dump({"args": {**vars(args), **a, "cam_norm": "src_dist"}}, open(osp.join(args.out, "config.json"), "w"), indent=1)
    print(f"model {args.arch} {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M, train objects {len(tr)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    lp = _lpips_net("vgg", "cuda") if args.img_weight > 0 else None
    frames = {}

    def image(o, k):
        d = c["objs"][o]
        if d not in frames:
            frames[d] = json.load(open(osp.join(d, "cameras.json")))["frames"]
        return torch.tensor(load_rgba_white(osp.join(d, frames[d][k]["file"]))[0]).permute(2, 0, 1)

    def forward(idx):
        i = torch.tensor(idx)
        z0 = c["z0"][i].float().cuda()[:, None]
        return net(z0, K[i, :1].cuda(), rel[i, :1].cuda(), K[i, 1:].cuda(), rel[i, 1:].cuda()).float()  # [B,V,T,C]

    @torch.no_grad()
    def evaluate(with_base=False):
        net.eval()
        res = {"model": []}
        if with_base:
            res.update({"identity": [], "oracle_zstar": [], "ceiling_zgt": []})
        for j in range(0, len(ev), 4):
            idx = ev[j:j + 4]
            gt = torch.stack([image(o, c["tgt_views"][v]) for o in idx for v in range(V)]).cuda()
            cand = {"model": forward(idx).flatten(0, 1)}
            if with_base:
                i = torch.tensor(idx)
                cand["identity"] = c["z0"][i].float().cuda()[:, None].expand(-1, V, -1, -1).flatten(0, 1)
                cand["oracle_zstar"] = c["zstar"][i].float().cuda().flatten(0, 1)
                cand["ceiling_zgt"] = c["zgt"][i].float().cuda().flatten(0, 1)
            for k, z in cand.items():
                x = torch.cat([se2.to01(se2.decode(se2.spherify(z[q:q + 12]), None, 0.0)) for q in range(0, len(z), 12)])
                res[k].append(lpips(x, gt).flatten().view(len(idx), V).cpu())
        net.train()
        out = {}
        for k, v in res.items():
            m = torch.cat(v)
            out[k] = {"mean": m.mean().item(), "by_step": dict(zip([str(t) for t in c["tgt_views"]], m.mean(0).tolist()))}
        return out

    rpath = osp.join(args.out, "resume.pt")
    if args.resume and osp.exists(rpath):
        ck = torch.load(rpath, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, base, hist, best = ck["step"] + 1, ck["base"], ck["hist"], tuple(ck["best"])
        print(f"resumed at step {start}", flush=True)
    else:
        base = evaluate(True)
        print("unseen baselines:", {k: round(v["mean"], 4) for k, v in base.items()}, flush=True)
        hist, best, start = [{"step": 0, "unseen": base["model"]["mean"]}], (9.0, 0), 1
    t0 = time.time()
    for step in range(start, args.steps + 1):
        idx = [tr[i] for i in np.random.randint(len(tr), size=args.batch)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = forward(idx)
        ps = se2.spherify(pr.float().flatten(0, 1))
        y = tgt[torch.tensor(idx)].float().cuda().flatten(0, 1)
        loss = F.smooth_l1_loss(ps, y, beta=0.1) + (1 - F.cosine_similarity(ps, y, dim=-1).mean())
        if lp is not None:
            sub = np.random.choice(len(ps), 8, replace=False)
            xh = se2.decode(ps[sub], None, 0.0) * 0.5 + 0.5
            xg = torch.stack([image(idx[s // V], c["tgt_views"][s % V]) for s in sub]).cuda()
            loss = loss + args.img_weight * (lp(xh * 2 - 1, xg * 2 - 1).mean() + (xh - xg).abs().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            e = evaluate()
            hist.append({"step": step, "unseen": e["model"]["mean"], "by_step": e["model"]["by_step"]})
            print(f"val@{step} unseen LPIPS {e['model']['mean']:.4f} by step {({k: round(v, 3) for k, v in e['model']['by_step'].items()})}", flush=True)
            torch.save(net.state_dict(), osp.join(args.out, "model_last.pt"))
            if e["model"]["mean"] < best[0]:
                best = (e["model"]["mean"], step)
                torch.save(net.state_dict(), osp.join(args.out, "model_best.pt"))
            json.dump({"baselines": base, "hist": hist, "best": best}, open(osp.join(args.out, "history.json"), "w"), indent=1)
            if args.resume:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                            "base": base, "hist": hist, "best": list(best)}, rpath + ".tmp")
                os.replace(rpath + ".tmp", rpath)
    print("BEST", best, "baselines", {k: round(v["mean"], 4) for k, v in base.items()})
    print("R2_DELTA_SCALE_DONE")


if __name__ == "__main__":
    main()
