"""
R3 data: G-Objaverse CC-BY renders prepared by se3d/data/prep_objaverse.py
(<root>/<obj>/{NN.png RGBA 256, cameras.json}). One item = one object:
S source views + T target views, images on white, cameras made RELATIVE to the
first source (its w2c becomes identity) and scaled so every camera centre lies in
[-2, 2]^3 (SEVA-style normalisation).
"""
import json
import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgba_white(path):
    a = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    return a[..., :3] * a[..., 3:] + (1 - a[..., 3:]), a[..., 3]


def camera_scale(w2c_ref, all_w2cs=None, mode="all_views", bound=2.0):
    """Translation scale for cameras expressed relative to the reference (source) camera.
    all_views: camera centres of ALL views of the object span [-bound, bound]^3 (depends on
               which other views exist: 12 random views, 64 random views and a 32-camera ring
               give different scales for the same camera pair).
    src_dist : the source camera sits at distance 1 from the object centre (world origin);
               uses only what a one-photo user knows, identical across datasets."""
    if mode == "src_dist":
        return 1.0 / max(np.linalg.norm(np.linalg.inv(w2c_ref)[:3, 3]), 1e-6)
    T0 = np.linalg.inv(w2c_ref)
    every = np.stack([w @ T0 for w in all_w2cs])
    centres = np.stack([-r[:3, :3].T @ r[:3, 3] for r in every])
    return bound / max(np.abs(centres).max(), 1e-6)


def normalise_cameras(w2cs, ref, all_w2cs=None, bound=2.0, mode="all_views"):
    """w2cs [V,4,4] -> relative to w2cs[ref] (identity), translations scaled by camera_scale."""
    T0 = np.linalg.inv(w2cs[ref])
    rel = np.stack([w @ T0 for w in w2cs])
    s = camera_scale(w2cs[ref], w2cs if all_w2cs is None else all_w2cs, mode, bound)
    rel[:, :3, 3] *= s
    return rel, s


class ObjaverseViews(Dataset):
    def __init__(self, root, objects=None, n_src=1, n_tgt=3, seed=0, deterministic=False,
                 src_views=None, tgt_views=None, length=None, cam_norm="all_views"):
        self.root = root
        self.cam_norm = cam_norm
        objs = objects or sorted(o for o in os.listdir(root) if osp.exists(osp.join(root, o, "cameras.json")))
        self.objs = objs
        self.n_src, self.n_tgt = n_src, n_tgt
        self.seed = seed
        self.deterministic = deterministic
        self.src_views, self.tgt_views = src_views, tgt_views  # optional view-index pools
        self.length = length

    def __len__(self):
        return self.length or len(self.objs)

    def __getitem__(self, i):
        oi = i % len(self.objs)
        meta = json.load(open(osp.join(self.root, self.objs[oi], "cameras.json")))
        fr = meta["frames"]
        rng = random.Random(self.seed * 1000003 + i) if self.deterministic else random
        if self.src_views is None:
            pick = rng.sample(range(len(fr)), self.n_src + self.n_tgt)
        else:
            src = rng.sample([v for v in self.src_views if v < len(fr)], self.n_src)
            tgt = rng.sample([v for v in self.tgt_views if v < len(fr) and v not in src], self.n_tgt)
            pick = src + tgt
        i = oi
        imgs, alphas = zip(*[load_rgba_white(osp.join(self.root, self.objs[i], fr[k]["file"])) for k in pick])
        w2c = np.stack([np.array(fr[k]["w2c"], dtype=np.float64) for k in pick])
        K = np.stack([np.array(fr[k].get("K", meta.get("K")), dtype=np.float64) for k in pick])
        all_w2c = np.stack([np.array(f["w2c"], dtype=np.float64) for f in fr])
        rel, s = normalise_cameras(w2c, 0, all_w2c, mode=self.cam_norm)
        t = lambda a: torch.tensor(np.stack(a) if isinstance(a, (list, tuple)) else a, dtype=torch.float32)
        x = t(imgs).permute(0, 3, 1, 2)  # [V,3,H,W] in [0,1]
        return {"x": x, "alpha": t(alphas), "K": t(K), "w2c": t(rel), "scale": s, "obj": self.objs[i]}
