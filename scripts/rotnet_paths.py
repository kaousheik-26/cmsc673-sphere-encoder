import argparse
import json
import logging

import torch

from halfsphere.backbone import SE2Backbone
from halfsphere.geometry import angle_deg
from halfsphere.records import Run
from spherevideo.clips import video_clips, write_mp4
from spherevideo.predict import log_map
from spherevideo.rotnet import RotNet, compose, dir_cos, split_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Map true, rotation-model and true-direction paths of each clip onto a 2D sphere chart")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-imagenet-256px")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--runs", type=str, nargs="+", required=True)
parser.add_argument("--tags", type=str, nargs="+", required=True)
parser.add_argument("--train_videos", type=str, default="workspace/videos/wan21_t2v_1.3b/*/")
parser.add_argument("--test_videos", type=str, default="workspace/videos/wan21_t2v_1.3b_test/*/")
parser.add_argument("--n_frames", type=int, default=81)
parser.add_argument("--fps", type=float, default=16.0)
parser.add_argument("--src_fps", type=float, default=16.0)
parser.add_argument("--save_videos", type=int, default=1)
args = parser.parse_args()


def encode_clips(bb, X, dev, bs=32):
    N, T = X.shape[:2]
    flat = X.flatten(0, 1)
    with torch.no_grad():
        Z = torch.cat([bb.encode(flat[i:i + bs].to(dev)) for i in range(0, flat.shape[0], bs)])
    return Z.view(N, T, *Z.shape[1:])


def joint_map(z0, paths):
    P = torch.cat(paths)
    V = log_map(z0[None].expand_as(P), P).flatten(1).double()
    w, U = torch.linalg.eigh(V @ V.T)
    w, U = w.flip(0).clamp_min(0), U.flip(1)
    c = U[:, :2] * w[:2].sqrt()
    d = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = angle_deg(z0[None].expand_as(P), P).double()
    xy = (d * r[:, None]).float()
    return list(xy.split([p.shape[0] for p in paths])), float(w[:2].sum() / w.sum().clamp_min(1e-12))


def r2(t):
    return [round(float(v), 2) for v in t.flatten()] if t.ndim == 1 else [[round(float(v), 2) for v in row] for row in t]


def main():
    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=dt)
    sets = {}
    for split, glob_ in (("train", args.train_videos), ("test", args.test_videos)):
        X, meta = video_clips(glob_, args.n_frames, args.fps, bb.image_size, src_fps=args.src_fps)
        sets[split] = (encode_clips(bb, X, dev), meta, X)
    T = args.n_frames
    run = Run("rotnet_paths", {"args": vars(args)}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path)
    out = {"fps": args.fps, "n_frames": T, "models": {}}
    t = torch.arange(T, device=dev).float() / (T - 1)
    for tag, rd in zip(args.tags, args.runs):
        ck = torch.load(f"{rd}/rotnet.pt", map_location=dev)
        a = ck["args"]
        Zs = sets["train"][0]
        net = RotNet(token_dim=Zs.shape[-1], n_tokens=Zs.shape[-2], width=a["width"], depth=a["depth"], hint=a["hint"]).to(dev)
        net.load_state_dict(ck["net"])
        net.eval()
        clips = []
        with torch.no_grad():
            for split, (Z, meta, X) in sets.items():
                for n in range(Z.shape[0]):
                    z0 = Z[n, 0][None].expand(T, *Z.shape[2:])
                    z1 = Z[n, 1][None].expand(T, *Z.shape[2:])
                    th_true, d_true = split_target(z0[1:], Z[n, 1:])
                    th_pred, d_pred = net(z0, t, z1)
                    P = compose(z0, th_pred, d_pred)
                    d_true = torch.cat([d_pred[:1], d_true])
                    C = compose(z0, th_pred, d_true)
                    C[0] = Z[n, 0]
                    (xt, xp, xc), cap = joint_map(Z[n, 0], [Z[n], P, C])
                    off = torch.rad2deg(torch.acos(dir_cos(d_pred, d_true).clamp(-1, 1)))
                    off[0] = 0
                    clips.append({"split": split, "clip": n, "kind": meta[n]["kind"],
                                  "true_xy": r2(xt), "pred_xy": r2(xp), "cheat_xy": r2(xc), "captured": round(cap, 3),
                                  "true_angle": r2(angle_deg(z0, Z[n])), "pred_angle": r2(th_pred),
                                  "err_deg": r2(angle_deg(P, Z[n])), "cheat_err_deg": r2(angle_deg(C, Z[n])), "dir_off_deg": r2(off)})
                    if args.save_videos:
                        dp = torch.cat([bb.decode(P[i:i + 27], 0.0) for i in range(0, T, 27)]).float().cpu()
                        dc = torch.cat([bb.decode(C[i:i + 27], 0.0) for i in range(0, T, 27)]).float().cpu()
                        write_mp4(torch.cat([X[n].float(), dp, dc], dim=-1), run.path("videos", f"{tag}_{split}_clip{n:02d}_{meta[n]['kind']}.mp4"), args.fps)
        out["models"][tag] = {"run": rd, "hint": a["hint"], "clips": clips}
        e = {s: [c for c in clips if c["split"] == s] for s in ("train", "test")}
        m = lambda cs, k: sum(sum(c[k][1:]) / (T - 1) for c in cs) / len(cs)
        logger.info(f"{tag}: err train {m(e['train'], 'err_deg'):.2f} test {m(e['test'], 'err_deg'):.2f}; true-direction err test {m(e['test'], 'cheat_err_deg'):.2f}; direction off test {m(e['test'], 'dir_off_deg'):.1f} deg")
    with open(run.path("paths.json"), "w") as fh:
        json.dump(out, fh)
    run.finish("ok")
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
