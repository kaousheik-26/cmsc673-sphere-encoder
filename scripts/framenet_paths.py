import argparse
import json
import logging

import torch

from halfsphere.backbone import SE2Backbone
from halfsphere.geometry import angle_deg
from halfsphere.records import Run
from spherevideo.clips import video_clips, write_mp4
from spherevideo.framenet import FrameNet
from spherevideo.predict import log_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="True vs predicted latent paths of trained frame predictors, in a shared map around frame 0")
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


def joint_map(z0, A, B):
    P = torch.cat([A, B])
    V = log_map(z0[None].expand_as(P), P).flatten(1).double()
    w, U = torch.linalg.eigh(V @ V.T)
    w, U = w.flip(0).clamp_min(0), U.flip(1)
    c = U[:, :2] * w[:2].sqrt()
    d = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = angle_deg(z0[None].expand_as(P), P).double()
    xy = (d * r[:, None]).float()
    return xy[:A.shape[0]], xy[A.shape[0]:], float(w[:2].sum() / w.sum().clamp_min(1e-12))


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
    run = Run("framenet_paths", {"args": vars(args)}, dev_dir=args.dev_dir, ckpt_path=bb.ckpt_path)
    out = {"fps": args.fps, "n_frames": T, "models": {}}
    t = torch.arange(T, device=dev).float() / (T - 1)
    for tag, rd in zip(args.tags, args.runs):
        ck = torch.load(f"{rd}/framenet.pt", map_location=dev)
        a = ck["args"]
        Zs = sets["train"][0]
        net = FrameNet(token_dim=Zs.shape[-1], n_tokens=Zs.shape[-2], width=a["width"], depth=a["depth"], heads=a.get("heads", 6), arch=a["arch"]).to(dev)
        net.load_state_dict(ck["net"])
        net.eval()
        clips = []
        with torch.no_grad():
            for split, (Z, meta, X) in sets.items():
                for n in range(Z.shape[0]):
                    z0 = Z[n, 0]
                    P = net(z0[None].expand(T, *z0.shape), t)
                    xt, xp, cap = joint_map(z0, Z[n], P)
                    clips.append({"split": split, "clip": n, "kind": meta[n]["kind"],
                                  "true_xy": r2(xt), "pred_xy": r2(xp), "captured": round(cap, 3),
                                  "true_angle": r2(angle_deg(z0[None].expand_as(Z[n]), Z[n])),
                                  "pred_angle": r2(angle_deg(z0[None].expand_as(P), P)),
                                  "err_deg": r2(angle_deg(P, Z[n]))})
                    if args.save_videos:
                        dp = torch.cat([bb.decode(P[i:i + 27], 0.0) for i in range(0, T, 27)]).float().cpu()
                        write_mp4(torch.cat([X[n].float(), dp], dim=-1), run.path("videos", f"{tag}_{split}_clip{n:02d}_{meta[n]['kind']}.mp4"), args.fps)
        out["models"][tag] = {"run": rd, "arch": a["arch"], "lpips_w": a.get("lpips_w", 0.0), "clips": clips}
        e = {s: sum(sum(c["err_deg"][1:]) / (T - 1) for c in clips if c["split"] == s) / sum(c["split"] == s for c in clips) for s in ("train", "test")}
        logger.info(f"{tag}: mean err train {e['train']:.2f} deg, test {e['test']:.2f} deg; mean captured {sum(c['captured'] for c in clips) / len(clips):.2f}")
    with open(run.path("paths.json"), "w") as fh:
        json.dump(out, fh)
    run.finish("ok")
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
