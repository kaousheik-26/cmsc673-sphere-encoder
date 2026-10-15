"""
Real-photo part of the dev set: CO3Dv2 teddybear single-sequence frames
(CC BY-NC 4.0, research use only).

Per sequence: the source frame (default: first frame with a well-centred mask)
and the frames whose relative camera rotation is closest to {15,30,45,60,90}
deg on ONE side of the trajectory. Each frame: square crop around the object
mask (1.3x bbox), resize to 256, composite on white with the mask; depth from
CO3D's MVS depth (float16-in-uint16 PNG) cropped/resized (nearest); cameras
converted from PyTorch3D NDC to OpenCV w2c + pixel K.
Output: <out>/<seq>/{NNN.png RGBA, NNN_depth.npy, cameras.json} (same layout as
the GSO render folders; "azimuth" = signed relative rotation angle in deg).
"""
import argparse
import gzip
import json
import os
import os.path as osp

import numpy as np
from PIL import Image

ROOT = "/fs/nexus-scratch/vinayakg/se3d/data/co3d_teddybear"
OUT = "/fs/nexus-scratch/vinayakg/se3d/data/co3d_dev"
FLIP = np.diag([-1.0, -1.0, 1.0])  # PyTorch3D camera (x left, y up) -> OpenCV (x right, y down)


def opencv_camera(vp, H, W):
    R = np.array(vp["R"], dtype=np.float64)  # X_cam = X_world @ R + T
    T = np.array(vp["T"], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = FLIP @ R.T
    w2c[:3, 3] = FLIP @ T
    f = np.array(vp["focal_length"], dtype=np.float64)
    pp = np.array(vp["principal_point"], dtype=np.float64)
    if vp.get("intrinsics_format", "ndc_norm_image_bounds") == "ndc_isotropic":
        s = min(H, W) / 2.0
        fx, fy = f * s
    else:
        fx, fy = f[0] * W / 2.0, f[1] * H / 2.0
        s = None
    if s is not None:
        cx, cy = W / 2.0 - pp[0] * s, H / 2.0 - pp[1] * s
    else:
        cx, cy = W / 2.0 - pp[0] * W / 2.0, H / 2.0 - pp[1] * H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, w2c


def load_depth(path, scale):
    raw = np.array(Image.open(path))
    d = raw.astype(np.uint16).view(np.float16).astype(np.float32) * scale
    d[~np.isfinite(d)] = 0
    return d


def rot_angle(w2c_a, w2c_b):
    Rr = w2c_b[:3, :3] @ w2c_a[:3, :3].T
    return np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))


def crop_frame(fr, size=256, margin=1.3):
    H, W = fr["image"]["size"]
    img = np.array(Image.open(osp.join(ROOT, fr["image"]["path"])).convert("RGB")).astype(np.float32) / 255.0
    mask = np.array(Image.open(osp.join(ROOT, fr["mask"]["path"]))).astype(np.float32) / 255.0
    if mask.ndim == 3:
        mask = mask[..., 0]
    depth = load_depth(osp.join(ROOT, fr["depth"]["path"]), fr["depth"]["scale_adjustment"])
    dmask = np.array(Image.open(osp.join(ROOT, fr["depth"]["mask_path"]))) > 0
    depth = np.where(dmask, depth, 0)
    K, w2c = opencv_camera(fr["viewpoint"], H, W)
    ys, xs = np.nonzero(mask > 0.5)
    if ys.size < 500:
        return None
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * margin)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = lambda a, v: np.pad(a, ((side, side), (side, side)) + ((0, 0),) * (a.ndim - 2), constant_values=v)
    img_p, m_p, d_p = pad(img, 1.0), pad(mask, 0.0), pad(depth, 0.0)
    sl = (slice(y0 + side, y0 + 2 * side), slice(x0 + side, x0 + 2 * side))
    img_c, m_c, d_c = img_p[sl], m_p[sl], d_p[sl]
    sc = size / side
    img_r = np.array(Image.fromarray((img_c * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)).astype(np.float32) / 255
    m_r = np.array(Image.fromarray((m_c * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)).astype(np.float32) / 255
    d_r = np.array(Image.fromarray(d_c).resize((size, size), Image.NEAREST))
    rgb = img_r * m_r[..., None] + (1 - m_r[..., None])
    Kc = K.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    Kc[:2] *= sc
    return rgb, m_r, d_r * (m_r > 0.5), Kc, w2c, (ys.size / (H * W))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=float, nargs="+", default=[15, 30, 45, 60, 90])
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    fa = json.load(gzip.open(osp.join(ROOT, "teddybear", "frame_annotations.jgz")))
    seqs = sorted(set(f["sequence_name"] for f in fa))
    for seq in seqs:
        frames = sorted([f for f in fa if f["sequence_name"] == seq], key=lambda f: f["frame_number"])
        mass = np.array([f["mask"].get("mass", 0) or 0 for f in frames], dtype=np.float64)
        frames = [f for f, m in zip(frames, mass) if m > 0.25 * np.median(mass[mass > 0])]
        # source must have a usable object mask AND valid MVS depth on the object (for the scale)
        while True:
            c = crop_frame(frames[0])
            if c is not None and ((c[2] > 0) & (c[1] > 0.5)).sum() > 2000:
                break
            frames = frames[1:]
        src = frames[0]
        H, W = src["image"]["size"]
        _, w2c_src = opencv_camera(src["viewpoint"], H, W)
        # signed angle: sign from the direction of travel along the trajectory (first half)
        angs = [rot_angle(w2c_src, opencv_camera(f["viewpoint"], *f["image"]["size"])[1]) for f in frames]
        chosen = [0]
        for t in args.targets:
            half = len(frames) // 2  # one-sided sweep: prefer the first half of the orbit
            cand = sorted([i for i, a in enumerate(angs) if abs(a - t) <= 5 and i not in chosen],
                          key=lambda i: (i >= half, abs(angs[i] - t), i))
            for i in cand:
                if crop_frame(frames[i]) is not None:
                    chosen.append(i)
                    break
        od = osp.join(args.out, seq)
        os.makedirs(od, exist_ok=True)
        meta = {"width": 256, "height": 256, "convention": "opencv_w2c_co3d_world", "frames": [], "license": "CC BY-NC 4.0"}
        Ks = []
        for n, i in enumerate(chosen):
            rgb, m, d, K, w2c, frac = crop_frame(frames[i])
            Image.fromarray(np.concatenate([(rgb * 255 + .5).astype(np.uint8), (m[..., None] * 255 + .5).astype(np.uint8)], -1)).save(osp.join(od, f"{n:03d}.png"))
            np.save(osp.join(od, f"{n:03d}_depth.npy"), d.astype(np.float32))
            meta["frames"].append({"file": f"{n:03d}.png", "depth": f"{n:03d}_depth.npy", "azimuth": float(angs[i]),
                                   "elevation": 0.0, "w2c": w2c.tolist(), "K": K.tolist(), "frame_number": frames[i]["frame_number"]})
            Ks.append(K)
        meta["K"] = Ks[0].tolist()  # per-frame K is in frames[i]["K"]
        json.dump(meta, open(osp.join(od, "cameras.json"), "w"))
        print(seq, [(f["frame_number"], round(f["azimuth"], 1)) for f in meta["frames"]])
    print("CO3D_DEV_DONE")


if __name__ == "__main__":
    main()
