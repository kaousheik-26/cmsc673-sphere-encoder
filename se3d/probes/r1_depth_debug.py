"""Analyse R1 --save_views dumps: per generated view, depth error of the running
point cloud (known pixels) and of the fused depth used for new pixels, raw and
after one global scale (to separate a global scale offset from shape error).
Usage: python se3d/probes/r1_depth_debug.py <run_dir>/<obj>_views"""
import glob
import os.path as osp
import sys

import numpy as np


def absrel(a, b, m):
    m = m & (a > 0) & (b > 0)
    return float(np.mean(np.abs(a[m] - b[m]) / b[m])) if m.any() else float("nan"), int(m.sum())


def main(d):
    files = sorted(glob.glob(osp.join(d, "az*.npz")), key=lambda f: abs(int(osp.basename(f)[2:-4])))
    # one global scale for the whole run from all known pixels
    num = den = 0.0
    data = []
    for f in files:
        z = np.load(f)
        data.append((osp.basename(f)[2:-4], z))
        m = z["known_obj"] & (z["d_gt"] > 0) & (z["d_render"] > 0)
        num += float((z["d_gt"][m] * z["d_render"][m]).sum())
        den += float((z["d_render"][m] ** 2).sum())
    s = num / max(den, 1e-9)
    print(f"global scale (GT / ours) = {s:.4f}")
    print(f"{'az':>5} | known: absrel raw  scaled   n   | new: absrel raw  scaled   n")
    for az, z in data:
        gt = z["d_gt"]
        k_raw, nk = absrel(z["d_render"], gt, z["known_obj"])
        k_s, _ = absrel(z["d_render"] * s, gt, z["known_obj"])
        n_raw, nn = absrel(z["d_new"], gt, z["new"])
        n_s, _ = absrel(z["d_new"] * s, gt, z["new"])
        print(f"{az:>5} |        {k_raw:.4f}  {k_s:.4f} {nk:6d} |      {n_raw:.4f}  {n_s:.4f} {nn:6d}")


if __name__ == "__main__":
    main(sys.argv[1])
