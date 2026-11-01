"""
Render one GSO object from N random cameras (azimuth U[0,360), elevation
U[-10,60], radius 1.5, SyncDreamer intrinsics) into
<out>/<obj>/{NN.png RGBA, NN_depth.npy, cameras.json with per-frame w2c + K}.
The camera sequence is seeded by the object name, so the first N cameras are the
same for any larger N.
"""
import argparse
import json
import os
import os.path as osp
import zlib

import numpy as np

from se3d.data.render_gso import GSO_RAW
from se3d.geometry.cameras import orbit_w2c, syncdreamer_K
from se3d.geometry.render_mesh import Renderer, TexturedMesh, save_view


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", default="ASICS_GEL1140V_WhiteBlackSilver")
    p.add_argument("--views", type=int, default=64)
    p.add_argument("--out", default="/fs/nexus-scratch/vinayakg/se3d/data/object_views")
    args = p.parse_args()
    od = osp.join(args.out, args.object)
    os.makedirs(od, exist_ok=True)
    rend = Renderer()
    mesh = TexturedMesh(f"{GSO_RAW}/{args.object}/meshes/model.obj", axis=np.eye(3))
    K = syncdreamer_K(256)
    rng = np.random.default_rng(zlib.crc32(args.object.encode()))
    frames = []
    for v in range(args.views):
        az, el = float(rng.uniform(0, 360)), float(rng.uniform(-10, 60))
        w2c = orbit_w2c(az, el, 1.5)
        save_view(osp.join(od, f"{v:02d}"), rend.render(mesh, K, w2c, 256, 256))
        frames.append({"file": f"{v:02d}.png", "depth": f"{v:02d}_depth.npy", "view": v, "azimuth": az,
                       "elevation": el, "w2c": w2c.tolist(), "K": K.tolist()})
    json.dump({"object": args.object, "frames": frames, "K": K.tolist(), "width": 256, "height": 256,
               "convention": "opencv_w2c_zup_world"}, open(osp.join(od, "cameras.json"), "w"))
    print("OBJECT_VIEWS_DONE", od, len(frames))


if __name__ == "__main__":
    main()
