"""Tier 3 prep: 24 lion images (20 train + 4 eval) from ImageNet, 256 crops + rembg masks."""
import os, os.path as osp, sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, "/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/sphere2-test")
from sphere.loader import center_crop_arr
from rembg import remove, new_session

SRC = "/fs/cml-datasets/ImageNet/ILSVRC2012/train/n02129165"
OUT = "/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/sphere2-test/workspace/data/tier3_lion"
for s in ["train/img", "train/mask", "eval/img", "eval/mask"]:
    os.makedirs(f"{OUT}/{s}", exist_ok=True)
sess = new_session("u2net")

def save_pair(path, dst, tag):
    pil = Image.open(path).convert("RGB")
    arr = np.array(center_crop_arr(pil, 256)).astype(np.uint8)
    Image.fromarray(arr).save(f"{dst}/img/{tag}.png")
    m = np.array(remove(Image.fromarray(arr), session=sess))[:, :, 3]
    Image.fromarray(m).save(f"{dst}/mask/{tag}.png")

files = sorted(glob.glob(f"{SRC}/*.JPEG"))[:24]
for k, p in enumerate(files):
    split = "train" if k < 20 else "eval"
    save_pair(p, f"{OUT}/{split}", f"lion_{k:02d}")
    if (k+1) % 6 == 0: print(f"  {k+1}/24")
print(f"train={len(glob.glob(f'{OUT}/train/img/*.png'))}  eval={len(glob.glob(f'{OUT}/eval/img/*.png'))}  in {OUT}")
