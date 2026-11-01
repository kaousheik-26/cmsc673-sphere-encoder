"""
Large object set for the latent search: G-Objaverse renders (MIT; 40 views/object @512, per-view
camera json) restricted to the CC-BY (+CC0) subset of Objaverse (ODC-By;
license per object from allenai/objaverse metadata).

Steps
  licenses : map G-Objaverse index -> Objaverse uid -> license; write
             objaverse_ccby_order.json = CC-BY/CC0 indices in a fixed shuffled
             order (seed 0), so the first 1K / 10K / 30K form nested sets.
  download : for objects [start, end) of that order, fetch the 40 views (png +
             json) in memory, downscale to 256 RGBA, convert cameras to OpenCV
             w2c + K, write <out>/<gidx_with_underscore>/{NN.png, cameras.json}.
             Nothing at 512 px is stored.
G-Objaverse camera json: "origin" = camera centre, "x","y","z" = camera axes in
world (x right, y down, z forward: OpenCV), x_fov/y_fov in radians.
"""
import argparse
import concurrent.futures as cf
import gzip
import io
import json
import os
import os.path as osp
import random
import time
import urllib.request

import numpy as np
from PIL import Image

META = "/fs/nexus-scratch/vinayakg/se3d/data/gobjaverse_meta"
OUT = "/fs/nexus-scratch/vinayakg/se3d/data/objaverse_ccby"
OSS = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/objaverse"
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata"
KEEP = {"by", "cc0"}


def fetch(url, tries=4, headers=None):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(1 + 2 * t)


def licenses():
    idx2uid = json.load(open(osp.join(META, "gobjaverse_280k_index_to_objaverse.json")))
    shards = sorted(set(v.split("/")[0] for v in idx2uid.values()))
    tok = open(osp.expanduser("~/.cache/huggingface/token")).read().strip()
    lic = {}

    def one(sh):
        p = osp.join(META, f"{sh}.json.gz")
        if not osp.exists(p):
            data = fetch(f"{HF}/{sh}.json.gz", headers={"Authorization": f"Bearer {tok}"})
            open(p, "wb").write(data)
        d = json.load(gzip.open(p))
        return {u: v.get("license") for u, v in d.items()}

    with cf.ThreadPoolExecutor(16) as ex:
        for m in ex.map(one, shards):
            lic.update(m)
    keep = []
    counts = {}
    for gidx, path in idx2uid.items():
        uid = osp.splitext(path.split("/")[1])[0]
        l = lic.get(uid)
        counts[l] = counts.get(l, 0) + 1
        if l in KEEP:
            keep.append(gidx)
    keep.sort()
    random.Random(0).shuffle(keep)
    json.dump({"order": keep, "license_counts": counts, "kept_licenses": sorted(KEEP)},
              open(osp.join(META, "objaverse_ccby_order.json"), "w"))
    print("license counts", counts, "kept", len(keep))


def camera(js, size=256):
    R = np.stack([js["x"], js["y"], js["z"]], 0).astype(np.float64)
    c = np.array(js["origin"], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ c
    fx = 0.5 * size / np.tan(js["x_fov"] / 2)
    fy = 0.5 * size / np.tan(js["y_fov"] / 2)
    K = np.array([[fx, 0, size / 2], [0, fy, size / 2], [0, 0, 1.0]])
    return w2c, K


def _one_view(gidx, v, od, size):
    b = f"{OSS}/{gidx}/campos_512_v4/{v:05d}/{v:05d}"
    try:
        png = fetch(b + ".png")
        js = json.loads(fetch(b + ".json"))
    except Exception:
        return None
    im = Image.open(io.BytesIO(png)).convert("RGBA").resize((size, size), Image.LANCZOS)
    im.save(osp.join(od, f"{v:02d}.png"), optimize=True)
    w2c, K = camera(js, size)
    return {"file": f"{v:02d}.png", "view": v, "w2c": w2c.tolist(), "K": K.tolist(),
            "scale": js.get("scale"), "offset": js.get("offset"), "bbox": js.get("bbox")}


VIEW_POOL = cf.ThreadPoolExecutor(96)


def download_object(gidx, out, size=256, views=40):
    od = osp.join(out, gidx.replace("/", "_"))
    if osp.exists(osp.join(od, "cameras.json")):
        return "skip"
    os.makedirs(od, exist_ok=True)
    frames = [f for f in VIEW_POOL.map(lambda v: _one_view(gidx, v, od, size), range(views)) if f is not None]
    if len(frames) < views // 2:
        return "fail"
    json.dump({"gidx": gidx, "frames": frames, "width": size, "height": size,
               "convention": "opencv_w2c", "source": "G-Objaverse campos_512_v4 (downscaled)"},
              open(osp.join(od, "cameras.json"), "w"))
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["licenses", "download"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    if args.step == "licenses":
        licenses()
        return
    order = json.load(open(osp.join(META, "objaverse_ccby_order.json")))["order"][args.start:args.end]
    os.makedirs(args.out, exist_ok=True)
    t0, stats = time.time(), {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(lambda g: download_object(g, args.out), order)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(order)} {stats} {time.time() - t0:.0f}s", flush=True)
    print("OBJAVERSE_DONE", stats, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
