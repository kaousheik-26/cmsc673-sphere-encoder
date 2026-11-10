"""
Pictures + short videos for the learned latent move (r2_delta_scale.py model) on UNSEEN objects.
Source = view 0; the model predicts every camera turn it was trained for in one pass (same
context as training); frozen SE2 decoder; compared with the true views and with copy.
Picks the --show best of --candidates unseen objects (largest gain over copy).
Outputs <out>/<name>.png (rows: true / ours, ring order -120..180 deg then low/close views),
<out>/<name>.mp4 (input | true | ours, ring order), <out>/summary.json.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from torchvision.utils import save_image

from se3d.data.images import load_rgba_white
from se3d.eval.gs_video import write_panels
from se3d.eval.metrics import lpips
from se3d.geometry.cameras import normalise_cameras
from se3d.routes.r2_move_net import build_move_net
from se3d.se2.wrapper import SE2

DEG = {1: 15, 2: 30, 3: 45, 4: 60, 6: 90, 8: 120, 12: 180, 23: -15, 22: -30, 21: -45, 20: -60, 18: -90, 16: -120}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--weights", default=None, help="state dict (default <model_dir>/model_best.pt)")
    p.add_argument("--part", required=True, help="a search part whose objects are in the unseen split")
    p.add_argument("--targets", type=int, nargs="+", default=[1, 3, 6, 23, 21, 18, 2, 4, 8, 12, 22, 20, 16, 27, 30])
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--show", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(osp.join(args.model_dir, "config.json")))["args"]
    net = build_move_net(cfg).cuda().eval()
    net.load_state_dict(torch.load(args.weights or osp.join(args.model_dir, "model_best.pt"), map_location="cpu", weights_only=True))
    se2 = SE2("large-fdlite-256")
    objs = torch.load(args.part, weights_only=False)["objs"][-args.candidates:]
    res = []
    for o in objs:
        fr = json.load(open(osp.join(o, "cameras.json")))
        meta, fr = fr, fr["frames"]
        idx = [0] + args.targets
        W = np.stack([np.array(fr[i]["w2c"], dtype=np.float64) for i in idx])
        K = torch.tensor(np.stack([np.array(fr[i].get("K", meta.get("K")), dtype=np.float64) for i in idx]), dtype=torch.float32).cuda()
        rel = torch.tensor(normalise_cameras(W, 0, mode="src_dist")[0], dtype=torch.float32).cuda()
        x = torch.stack([torch.tensor(load_rgba_white(osp.join(o, fr[i]["file"]))[0]).permute(2, 0, 1) for i in idx]).cuda()
        with torch.no_grad():
            z0 = se2.encode(x[:1] * 2 - 1)
            pred = net(z0[None], K[None, :1], rel[None, :1], K[None, 1:], rel[None, 1:])[0].float()
            ours = se2.to01(se2.decode(se2.spherify(pred), None, 0.0))
            copy = se2.to01(se2.decode(z0, None, 0.0)).expand_as(ours)
            lo, lc = lpips(ours, x[1:]).flatten(), lpips(copy, x[1:]).flatten()
        res.append({"obj": o, "ours": lo.mean().item(), "copy": lc.mean().item(), "x": x.cpu(), "y": ours.cpu()})
    res.sort(key=lambda r: r["ours"] - r["copy"])
    ring = sorted([t for t in args.targets if t in DEG], key=lambda t: DEG[t])
    extra = [t for t in args.targets if t not in DEG]
    order = ring[:ring.index(next(t for t in ring if DEG[t] > 0))] + [0] + [t for t in ring if DEG[t] > 0] + extra
    summ = []
    for r in res[:args.show]:
        name = osp.basename(r["obj"])
        col = lambda t: (r["x"][0], r["x"][0]) if t == 0 else (r["x"][1 + args.targets.index(t)], r["y"][args.targets.index(t)])
        true_row = torch.stack([col(t)[0] for t in order])
        ours_row = torch.stack([col(t)[1] for t in order])
        save_image(torch.cat([true_row, ours_row]), osp.join(args.out, f"{name}.png"), nrow=len(order))
        write_panels(osp.join(args.out, f"{name}.mp4"),
                     [("input photo", r["x"][:1]), ("true view", true_row), ("ours: learned latent move", ours_row)], fps=3)
        summ.append({"obj": name, "lpips_ours": r["ours"], "lpips_copy": r["copy"], "order": [DEG.get(t, f"view{t}") for t in order]})
        print(name, f"ours {r['ours']:.3f} copy {r['copy']:.3f}", flush=True)
    json.dump({"shown": summ, "all": [{"obj": osp.basename(r["obj"]), "ours": r["ours"], "copy": r["copy"]} for r in res]},
              open(osp.join(args.out, "summary.json"), "w"), indent=1)
    print("R2_VIS_DONE")


if __name__ == "__main__":
    main()
