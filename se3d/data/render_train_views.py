"""
Training views for the reconstruction-regime decoder fine-tune (P1 gate) and
other trained parts: GSO objects that are NOT in any evaluation set (GSO30,
SEVA GSO-300), split list in data/gso_train_split.json.

Per object: N random cameras, half in the SyncDreamer framing (f=280 @256,
r=1.5) and half in the SV3D framing (FOV 33.8, r=2.0); azimuth U[0,360),
elevation U[-10, 60]. White background, view-independent shading (same renderer
as the evaluation renders). Output: <out>/<obj>/NNN.png (RGBA) + cameras.json.
"""
import argparse
import json
import os
import os.path as osp
import zlib

import numpy as np

from se3d.data.render_gso import GSO_RAW, render_object
from se3d.geometry.cameras import intrinsics_from_fov, orbit_w2c, syncdreamer_K
from se3d.geometry.render_mesh import Renderer, TexturedMesh

DATA = "/fs/nexus-scratch/vinayakg/se3d/data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--out", default=f"{DATA}/gso_train_renders")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    split = json.load(open(osp.join(osp.dirname(__file__), "gso_train_split.json")))["train"]
    objs = [o for i, o in enumerate(split) if i % args.nshards == args.shard]
    rend = Renderer()
    K_sd, K_sv = syncdreamer_K(256), intrinsics_from_fov(33.8, 256)
    n_ok = 0
    for oi, obj in enumerate(objs):
        od = osp.join(args.out, obj)
        if osp.exists(osp.join(od, "cameras.json")):
            n_ok += 1
            continue
        try:
            mesh = TexturedMesh(f"{GSO_RAW}/{obj}/meshes/model.obj", axis=np.eye(3))
        except Exception as ex:  # a few GSO assets have no texture / broken files
            print("skip", obj, ex, flush=True)
            continue
        rng = np.random.default_rng(args.seed * 100003 + zlib.crc32(obj.encode()) % 100003)
        cams_sd, cams_sv = [], []
        for k in range(args.views):
            az, el = float(rng.uniform(0, 360)), float(rng.uniform(-10, 60))
            if k % 2 == 0:
                cams_sd.append((az, el, orbit_w2c(az, el, 1.5)))
            else:
                cams_sv.append((az, el, orbit_w2c(az, el, 2.0)))
        render_object(rend, mesh, cams_sd, osp.join(od, "sd"), K_sd, 256)
        render_object(rend, mesh, cams_sv, osp.join(od, "sv"), K_sv, 256)
        json.dump({"done": True}, open(osp.join(od, "cameras.json"), "w"))
        n_ok += 1
        if oi % 50 == 0:
            print(f"{oi}/{len(objs)} {obj}", flush=True)
    print("TRAIN_RENDER_DONE", n_ok)


if __name__ == "__main__":
    main()
