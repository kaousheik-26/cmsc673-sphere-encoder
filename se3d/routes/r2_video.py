"""
R2 as a video (ORACLE: each latent is searched against the true view). For one object:
latent search from the input photo's latent z0 (mu pull toward z0) at cameras on two
elevation rings -> decode -> Gaussian splat on those decodes -> smooth 360 path.
Outputs: <out>.mp4 (input | true | 3D splat of the searched views), <out>.png (true /
searched decode at every 4th ring camera), <out>.json (ring + path metrics).
"""
import argparse
import json
import os.path as osp

import numpy as np
import torch

from se3d.data.render_gso import GSO_RAW
from se3d.eval.gs_video import visual_hull_points, write_panels
from se3d.eval.metrics import lpips, psnr
from se3d.geometry.cameras import orbit_w2c, syncdreamer_K
from se3d.geometry.render_mesh import Renderer, TexturedMesh
from se3d.gs.fit_gsplat import fit_gs, render_gs
from se3d.routes.r2_latent_search import search
from se3d.data.images import load_rgba_white
from se3d.se2.wrapper import SE2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/fs/nexus-scratch/vinayakg/se3d/data/overfit")
    p.add_argument("--object", required=True)
    p.add_argument("--src_view", type=int, default=0)
    p.add_argument("--rings", type=float, nargs="+", default=[10.0, 40.0])
    p.add_argument("--az_step", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--path_elev", type=float, default=25.0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--gs_iters", type=int, default=6000)
    p.add_argument("--sh_degree", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    meta = json.load(open(osp.join(args.root, args.object, "cameras.json")))
    K = syncdreamer_K(256)
    Kt = torch.tensor(K, dtype=torch.float32, device="cuda")
    se2 = SE2("large-fdlite-256")
    rgb, _ = load_rgba_white(osp.join(args.root, args.object, meta["frames"][args.src_view]["file"]))
    x_src = torch.tensor(rgb, device="cuda").permute(2, 0, 1)[None]
    rend = Renderer()
    mesh = TexturedMesh(f"{GSO_RAW}/{args.object}/meshes/model.obj", axis=np.eye(3))
    render = lambda W: torch.stack([rend.render(mesh, K, w, 256, 256)["rgb"].permute(2, 0, 1) for w in W])
    W_fit = [orbit_w2c(a, el, 1.5) for el in args.rings for a in np.arange(0, 360, args.az_step)]
    gt_fit = render(W_fit)
    with torch.no_grad():
        z0 = se2.encode(x_src * 2 - 1)
    dec = []
    for i in range(0, len(W_fit), 8):
        xt = gt_fit[i:i + 8]
        B = xt.shape[0]
        zb = z0.expand(B, *z0.shape[1:]).contiguous()
        best, _ = search(se2, xt, zb, zb, zb, None, args.steps, 1e-2, args.mu, 0.0, log_every=50)
        with torch.no_grad():
            dec.append(se2.to01(se2.decode(best["z"], None, 0.0)))
        print(f"searched {i + B}/{len(W_fit)}", flush=True)
    dec = torch.cat(dec)
    ring = {"psnr": float(np.mean([psnr(dec[i:i + 1], gt_fit[i:i + 1]).item() for i in range(len(W_fit))])),
            "lpips": float(np.mean([lpips(dec[i:i + 1], gt_fit[i:i + 1]).item() for i in range(len(W_fit))]))}
    imgs = dec.permute(0, 2, 3, 1).contiguous()
    w2c_fit = torch.tensor(np.stack(W_fit), dtype=torch.float32, device="cuda")
    xyz, col = visual_hull_points(imgs, w2c_fit, Kt)
    gs = fit_gs(imgs, w2c_fit, Kt, xyz, col, iters=args.gs_iters, sh_degree=args.sh_degree)
    W_path = [orbit_w2c(a, args.path_elev, 1.5) for a in np.linspace(0, 360, args.frames, endpoint=False)]
    gt = render(W_path)
    with torch.no_grad():
        gsr = torch.cat([render_gs(gs, torch.tensor(np.stack(W_path[i:i + 8]), dtype=torch.float32, device="cuda"), Kt, 256, 256)[0]
                         for i in range(0, len(W_path), 8)]).permute(0, 3, 1, 2).clamp(0, 1)
    path = {"psnr": float(np.mean([psnr(gsr[i:i + 1], gt[i:i + 1]).item() for i in range(len(W_path))])),
            "lpips": float(np.mean([lpips(gsr[i:i + 1], gt[i:i + 1]).item() for i in range(len(W_path))]))}
    json.dump({"object": args.object, "mu": args.mu, "steps": args.steps, "ring_searched": ring, "path_gs": path},
              open(args.out + ".json", "w"), indent=1)
    write_panels(args.out + ".mp4", [("input photo", x_src), ("true view", gt), ("latent search (uses true view) -> 3D splat", gsr)])
    from torchvision.utils import save_image
    sel = list(range(0, len(W_fit) // len(args.rings), 4))
    save_image(torch.cat([gt_fit[sel], dec[sel]]), args.out + ".png", nrow=len(sel))
    print("R2_VIDEO", json.dumps({"ring": ring, "path_gs": path}))
    print("R2_VIDEO_DONE")


if __name__ == "__main__":
    main()
