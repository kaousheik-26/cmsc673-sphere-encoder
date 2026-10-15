"""
Find, per GSO30 object, the mesh orientation that reproduces SyncDreamer's
renders: search the 24 axis-aligned rotations x yaw (5 deg, then 0.5 deg
refinement) for max mean mask IoU over the 16 sd16 views. Writes
<data>/gso30_syncdreamer/alignment.json: {obj: {"R": 3x3, "iou_mean", "iou_min"}}.
"""
import itertools
import json

import numpy as np
import torch
from PIL import Image

from se3d.data.render_gso import SD_MESHES, SD_RENDERS, protocol_cams
from se3d.geometry.cameras import syncdreamer_K
from se3d.geometry.render_mesh import Renderer, TexturedMesh


def rot24():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                R[i, j] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def yaw(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def main():
    import os

    rend = Renderer()
    K = syncdreamer_K(256)
    cams = protocol_cams("sd16")
    objs = sorted(os.listdir(SD_RENDERS))
    gt = {o: torch.tensor(np.stack([np.array(Image.open(f"{SD_RENDERS}/{o}/{i:03d}.png"))[..., 3] > 127
                                    for i in range(16)]), device="cuda") for o in objs}
    res = {}
    for obj in objs:
        base = TexturedMesh(f"{SD_MESHES}/{obj}/meshes/model.obj", axis=np.eye(3), normalize=None)
        v0 = base.v.clone()

        def score(R, views=(0, 4, 8, 12)):
            v = v0 @ torch.tensor(R, dtype=torch.float32, device="cuda").T
            lo, hi = v.min(0).values, v.max(0).values
            base.v = (v - (lo + hi) / 2) / (hi - lo).max()
            ious = []
            for i in views:
                a = rend.render(base, K, cams[i][2], 256, 256, ssaa=1)["alpha"] > 0.5
                b = gt[obj][i]
                ious.append(((a & b).sum() / (a | b).sum().clamp_min(1)).item())
            return float(np.mean(ious)), ious

        best = (-1, None)
        for R in rot24():
            for y in range(0, 360, 5):
                s, _ = score(yaw(y) @ R)
                if s > best[0]:
                    best = (s, (R, y))
        R0, y0 = best[1]
        for y in np.arange(y0 - 5, y0 + 5.01, 0.5):
            s, _ = score(yaw(y) @ R0)
            if s > best[0]:
                best = (s, (R0, float(y)))
        R0, y0 = best[1]
        s, ious = score(yaw(y0) @ R0, views=range(16))
        Rf = yaw(y0) @ R0
        res[obj] = {"R": Rf.tolist(), "yaw": y0, "iou_mean": s, "iou_min": float(min(ious))}
        print(f"{obj:15s} yaw {y0:6.1f} IoU mean {s:.3f} min {min(ious):.3f}", flush=True)
    out = "/fs/nexus-scratch/vinayakg/se3d/data/gso30_syncdreamer/alignment.json"
    json.dump(res, open(out, "w"), indent=1)
    print("MEAN IoU", np.mean([r["iou_mean"] for r in res.values()]), "MIN", min(r["iou_min"] for r in res.values()))
    print("ALIGN_DONE")


if __name__ == "__main__":
    main()
