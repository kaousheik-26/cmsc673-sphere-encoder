"""
Render GSO objects with nvdiffrast for the se3d harness.

Protocols (all at 256 px, SyncDreamer intrinsics f=280, radius 1.5, Z-up world):
  sd16      : SyncDreamer GSO30 cameras (elev 30, az = k * 22.5)
  ring32    : elev 30, az = k * 11.25 (even = sd16 train views, odd = interleaved holdout)
  dev_orbit : elev 30, az in {0, +-5, +-15, +-30, +-45, +-60, +-75, +-90}
  ext8      : extrapolated holdout, elev in {0, 60}, az = k * 90 (+ 45 offset at 60)
Output: <out>/<protocol>/<obj>/{NNN.png (RGBA, white-composited RGB), NNN_depth.npy}, cameras.json

--axis_check renders sd16 with several OBJ->world axis candidates and reports mask
IoU against SyncDreamer's own renders, to validate camera + axis conventions.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image

from se3d.geometry.cameras import intrinsics_from_fov, orbit_w2c, syncdreamer_K
from se3d.geometry.render_mesh import OBJ_TO_ZUP, Renderer, TexturedMesh, save_view

DATA = "/fs/nexus-scratch/vinayakg/se3d/data"
SD_RENDERS = f"{DATA}/gso30_syncdreamer/renders"
SD_MESHES = f"{DATA}/gso30_syncdreamer/meshes/gso"
SD_CAMS = f"{DATA}/syncdreamer_meta/camera-16.npz"
GSO_RAW = f"{DATA}/gso_raw"  # meshes for the SEVA GSO-300 split
SEVA_GSO = f"{DATA}/seva_bench/weka/home-jensen/reve/datasets/reconfusion_export/gso"
# SV3D/SEVA GSO-300 static orbit, our reconstruction (official cameras unreleased,
# stable-virtual-camera issue #76): 21 views, az step 360/21, elev 10, FOV 33.8, r 2.0
SV3D_FOV, SV3D_RADIUS, SV3D_ELEV = 33.8, 2.0, 10.0

AXES = {
    "blender_obj_import": OBJ_TO_ZUP,
    "identity": np.eye(3),
    "x_minus90": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
}


def protocol_cams(name):
    if name == "sd16":
        c = np.load(SD_CAMS)
        out = []
        for i in range(16):
            w2c = np.eye(4)
            w2c[:3] = c["poses"][i]
            out.append((float(np.rad2deg(c["azimuths"][i])), float(np.rad2deg(c["elevations"][i])), w2c))
        return out
    if name == "ring32":
        return [(k * 11.25, 30.0, orbit_w2c(k * 11.25, 30.0, 1.5)) for k in range(32)]
    if name == "dev_orbit":
        azs = [0, 5, -5, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
        return [(float(a), 30.0, orbit_w2c(a, 30.0, 1.5)) for a in azs]
    if name == "sv3d21":
        return [(k * 360.0 / 21, SV3D_ELEV, orbit_w2c(k * 360.0 / 21, SV3D_ELEV, SV3D_RADIUS)) for k in range(21)]
    if name == "ext8":
        v = [(k * 90.0, 0.0) for k in range(4)] + [(k * 90.0 + 45, 60.0) for k in range(4)]
        return [(a, e, orbit_w2c(a, e, 1.5)) for a, e in v]
    raise ValueError(name)


def render_object(rend, mesh, cams, out_dir, K, size):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"K": K.tolist(), "width": size, "height": size, "convention": "opencv_w2c_zup_world",
            "frames": []}
    for i, (az, el, w2c) in enumerate(cams):
        v = rend.render(mesh, K, w2c, size, size)
        save_view(osp.join(out_dir, f"{i:03d}"), v)
        meta["frames"].append({"file": f"{i:03d}.png", "depth": f"{i:03d}_depth.npy",
                               "azimuth": az, "elevation": el, "w2c": np.asarray(w2c).tolist()})
    meta["mesh"] = mesh.scale_info
    with open(osp.join(out_dir, "cameras.json"), "w") as f:
        json.dump(meta, f)


def iou_vs_syncdreamer(rend, mesh, obj, K):
    ious = []
    for i, (_, _, w2c) in enumerate(protocol_cams("sd16")):
        a = rend.render(mesh, K, w2c, 256, 256)["alpha"].cpu().numpy() > 0.5
        b = np.array(Image.open(f"{SD_RENDERS}/{obj}/{i:03d}.png"))[..., 3] > 127
        ious.append((a & b).sum() / max((a | b).sum(), 1))
    return float(np.mean(ious)), float(np.min(ious))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="*", default=None, help="default: all GSO30")
    p.add_argument("--protocols", nargs="+", default=["sd16", "ring32", "dev_orbit", "ext8"])
    p.add_argument("--out", default=f"{DATA}/gso_renders")
    p.add_argument("--axis_check", action="store_true")
    p.add_argument("--axis", default="aligned", help="aligned = per-object R from align_gso30.py")
    p.add_argument("--gso300", action="store_true", help="render the SEVA GSO-300 split from raw GSO meshes")
    args = p.parse_args()
    objs = args.objects or sorted(os.listdir(SD_RENDERS))
    rend = Renderer()
    K = syncdreamer_K(256)

    if args.axis_check:
        res = {}
        for obj in objs:
            obj_path = f"{SD_MESHES}/{obj}/meshes/model.obj"
            res[obj] = {n: iou_vs_syncdreamer(rend, TexturedMesh(obj_path, axis=A), obj, K)
                        for n, A in AXES.items()}
            print(obj, {n: f"{m:.3f}/{mn:.3f}" for n, (m, mn) in res[obj].items()}, flush=True)
        for n in AXES:
            print(f"AXIS {n}: mean IoU {np.mean([res[o][n][0] for o in objs]):.4f} "
                  f"min {np.min([res[o][n][1] for o in objs]):.4f}")
        os.makedirs(args.out, exist_ok=True)
        with open(osp.join(args.out, "axis_check.json"), "w") as f:
            json.dump(res, f, indent=1)
        return

    if args.gso300:
        Kv = intrinsics_from_fov(SV3D_FOV, 256)
        for obj in (args.objects or sorted(os.listdir(SEVA_GSO))):
            mesh = TexturedMesh(f"{GSO_RAW}/{obj}/meshes/model.obj", axis=np.eye(3))
            render_object(rend, mesh, protocol_cams("sv3d21"), osp.join(args.out, "sv3d21", obj), Kv, 256)
        print("RENDER_DONE")
        return
    align = {}
    if args.axis == "aligned":
        align = json.load(open(f"{DATA}/gso30_syncdreamer/alignment.json"))
    for obj in objs:
        R = np.array(align[obj]["R"]) if args.axis == "aligned" else AXES[args.axis]
        mesh = TexturedMesh(f"{SD_MESHES}/{obj}/meshes/model.obj", axis=R)
        for proto in args.protocols:
            render_object(rend, mesh, protocol_cams(proto), osp.join(args.out, proto, obj), K, 256)
        print("rendered", obj, flush=True)
    print("RENDER_DONE")


if __name__ == "__main__":
    main()
