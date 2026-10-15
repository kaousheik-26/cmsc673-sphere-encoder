"""
Route R1: training-free iterative warp-and-fill with the SE2 projector.

Per scene (source image x0 + target cameras, autoregressive order):
  P <- unproject(x0, depth0 = MoGe2(x0), scaled to the orbit)
  for c_k: x_warp, m = render(P -> c_k); mask hygiene; pre-fill
           for a in schedule (fixed-a repeats, RePaint-style):
               x_gen = D(rot(F(E(x)), e_scene, a), y, a)
               x = (m x_warp + lam x_gen) / (m + lam)          # SP3 HQS, A = mask
           depth_k = MoGe2(x) aligned (LSQ scale+shift) to the rendered depth on m
           P <- P u unproject(new object pixels of x)
  optional VistaDream sweep: fit GS on all views, re-project each generated view
  toward its GS render with low-a HQS steps, refit.
Evaluation (GSO dev / GSO30): per-view metrics vs GT, GS fit on the views ->
held-out GT views (interleaved inside the generated arc, extrapolated outside).
"""
import argparse
import dataclasses
import json
import math
import os
import os.path as osp
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

from se3d.eval.metrics import all_metrics
from se3d.geometry.cameras import orbit_w2c
from se3d.geometry.depth import MoGeDepth, align_scale_shift, fov_x_from_K, fuse_depth
from se3d.geometry.pointcloud import PointCloud
from se3d.geometry.warp import mask_hygiene, prefill, prefill_object, with_background
from se3d.gs.fit_gsplat import fit_gs, points_from_views, render_gs
from se3d.se2.wrapper import SE2


@dataclass
class R1Config:
    model: str = "large-fdlite-256"
    decoder_ckpt: str = ""          # fine-tuned decoder (empty = released weights)
    schedule: list = field(default_factory=lambda: [75, 45, 25])  # best on 5/5 dev objects (was 75x3,45x2,25x2)
    lam: float = 1.0
    hard_paste: bool = False
    prefill: str = "nn"
    prefill_hull: bool = True       # nn inside the object hull, white outside
    hygiene: bool = True
    clean_alpha: float = -1.0       # <0: no extra clean pass; else D(F(E(x)), y, a) at the end
    class_mode: str = "null"        # null | pred
    share_e: bool = True
    depth_source: str = "moge"      # moge | gt | gt_src | gt_new  (gt* = oracle, debugging only)
    step_deg: float = 15.0
    order: str = "alternate"        # alternate | one_sided
    final_sweeps: int = 0
    sweep_schedule: list = field(default_factory=lambda: [25, 25])
    sweep_lam: float = 1.0
    gs_iters: int = 2000
    gs_sh: int = 1                  # 0 = no view-dependent colour for the GS on generated views
    gs_lpips: float = 0.0           # LPIPS weight in that GS fit (0.2 removes most splat noise on R3 views)
    fg_thresh: float = 0.08         # non-white threshold for new object pixels
    bg_margin: int = 8              # unknown band (px) around the hull of the warped object
    new_px_hull_dilate: int = 6     # new object pixels must lie within the known hull dilated by this (px)
    pc_edge_thresh: float = 0.04    # >0: drop flying pixels (relative 3x3 depth jump) before unprojecting
    pc_erode: int = 2               # erode each added mask by this many px
    depth_fusion: str = "harmonic"  # global (LSQ scale+shift) | harmonic (scale+shift + smooth seam offset)
    save_views: bool = False        # debug: keep per-view rgb / depths / masks in the returned views
    depth_scale: str = "geo_q90_aabb"  # geo_q90_aabb (best of probes/scale_check: 3.6% mean err) | limb | gt (oracle)
    seed: int = 0
    color_match: bool = False       # affine colour map gen->warp fitted on known object pixels, each step
    latent_refine_steps: int = 0    # >0: test-time latent refinement (P2) after the HQS loop
    latent_refine_lr: float = 1e-2
    latent_refine_mu: float = 1.0   # pull toward the initial F(E(x)): mu * 2 (1 - cos)
    latent_refine_hole_w: float = 0.3
    final_paste: str = "none"       # none | hard | feather: put the known warp pixels back at the end
    feather_px: float = 3.0


def _imagenet_classifier():
    """ConvNeXt-B (IN-22k -> IN-1k) class probabilities for the class-conditional mode."""
    import timm

    m = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def predict(x01):
        x = torch.nn.functional.interpolate(x01, size=cfg["input_size"][1:], mode="bicubic", antialias=True)
        return m((x - mean) / std).softmax(-1)

    return predict


def nonwhite(x01, thr):
    """x01 [H,W,3] -> bool mask of pixels that differ from white by > thr."""
    m = ((1 - x01).max(-1).values > thr).cpu().numpy().astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return torch.tensor(m.astype(bool), device=x01.device)


def color_match(x_gen, x_ref, mask, white_frac=0.3):
    """least-squares affine colour map (3x3 + bias) taking x_gen to x_ref on the
    known object pixels, constrained (by heavily weighted anchor rows) to map white
    to white so the backdrop is not tinted."""
    A = x_gen[:, mask].T  # [n,3]
    B = x_ref[:, mask].T
    nw = max(64, int(white_frac * A.shape[0]))
    ones = torch.ones(nw, 3, device=A.device)
    A = torch.cat([A, ones])
    B = torch.cat([B, ones])
    A1 = torch.cat([A, torch.ones_like(A[:, :1])], 1)
    M = torch.linalg.lstsq(A1.double(), B.double()).solution.float()  # [4,3]
    out = torch.cat([x_gen.flatten(1).T, torch.ones_like(x_gen.flatten(1).T[:, :1])], 1) @ M
    return out.T.reshape_as(x_gen).clamp(0, 1)


def feather_paste(x, x_warp, known, px):
    """x <- w x_warp + (1-w) x with w = blurred (eroded) known mask."""
    k = known.float()[None, None]
    if px > 0:
        r = int(2 * px)
        k = -torch.nn.functional.max_pool2d(-k, 2 * r + 1, 1, r)  # erode by r
        ker = torch.exp(-torch.arange(-3 * r, 3 * r + 1, device=x.device).float() ** 2 / (2 * px * px))
        ker = ker / ker.sum()
        k = torch.nn.functional.conv2d(k, ker.view(1, 1, 1, -1), padding=(0, 3 * r))
        k = torch.nn.functional.conv2d(k, ker.view(1, 1, -1, 1), padding=(3 * r, 0))
    w = k[0]
    return w * x_warp + (1 - w) * x


def orbit_unitbox_scale(d, m, K, w2c, radius):
    """Depth scale for an object-centric orbit (camera looks at the object centre
    from `radius`; object normalised to a unit bounding box): geometric mean of
    the q90 limb rule (radius / q90 depth) and the unit-box rule (1 / largest side
    of the visible points' box). Mean |log error| 3.6% on GSO30 (q95 alone: 6.4%)."""
    import math
    from se3d.geometry.cameras import unproject_depth

    s_limb = radius / torch.quantile(d[m], 0.90).item()
    pts, _ = unproject_depth(d, K, w2c, m)
    side = (pts.max(0).values - pts.min(0).values).max().item()
    return math.sqrt(s_limb * (1.0 / max(side, 1e-6)))


def hull_mask(known_obj, dilate_px):
    k = known_obj.cpu().numpy().astype(np.uint8)
    h = np.zeros_like(k)
    pts = cv2.findNonZero(k)
    if pts is not None:
        cv2.fillConvexPoly(h, cv2.convexHull(pts), 1)
        if dilate_px > 0:
            h = cv2.dilate(h, np.ones((3, 3), np.uint8), iterations=dilate_px)
    return torch.tensor(h.astype(bool), device=known_obj.device)


class R1:
    def __init__(self, cfg: R1Config):
        self.cfg = cfg
        self.se2 = SE2(cfg.model, decoder_ckpt=cfg.decoder_ckpt or None)
        self.depth = MoGeDepth() if cfg.depth_source != "gt" else None
        self._clf = None

    # ------------------------------------------------------------ helpers
    def class_label(self, x01):
        if self.cfg.class_mode == "null":
            return None
        if self._clf is None:
            self._clf = _imagenet_classifier()
        return self.se2.imagenet_to_model_class(int(self._clf(x01[None]).argmax()))

    def project(self, x_warp, m, x_init, y, e, schedule, lam, obj_mask=None):
        """HQS loop. x_warp/x_init [3,H,W] in [0,1]; m [H,W] float (1 = known).
        obj_mask [H,W] bool: known OBJECT pixels used for colour matching."""
        x = x_init
        mm = m[None]
        for a in schedule:
            with torch.no_grad():
                z = self.se2.encode(x[None] * 2 - 1)
                z = self.se2.rotate(z, e, a) if a > 0 else z
                x_gen = self.se2.to01(self.se2.decode(z, y, a))[0]
                if self.cfg.color_match and obj_mask is not None and obj_mask.sum() > 200:
                    x_gen = color_match(x_gen, x_warp, obj_mask)
            if self.cfg.hard_paste:
                x = mm * x_warp + (1 - mm) * x_gen
            else:
                x = (mm * x_warp + lam * x_gen) / (mm + lam)
        return x

    def latent_refine(self, x, x_warp, known, y):
        """Optimise z (init F(E(x))) so D(z, y, 0) matches T = known * x_warp + (1-known) * x,
        known pixels weighted 1 and hole pixels latent_refine_hole_w (L1), LPIPS on T."""
        from se3d.eval.metrics import _lpips_net

        cfg, se2 = self.cfg, self.se2
        lp = _lpips_net("vgg", "cuda")
        m = known.float()[None, None]
        T = (m * x_warp[None] + (1 - m) * x[None]).detach()
        w = m + (1 - m) * cfg.latent_refine_hole_w
        with torch.no_grad():
            z0 = se2.encode(x[None] * 2 - 1)
        u = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=cfg.latent_refine_lr)
        for _ in range(cfg.latent_refine_steps):
            z = se2.spherify(u)
            xh = se2.decode(z, y, 0.0) * 0.5 + 0.5
            cosz = (z * z0).flatten(1).sum(1) / (z.flatten(1).norm(dim=1) * z0.flatten(1).norm(dim=1))
            loss = lp(xh * 2 - 1, T * 2 - 1).mean() + (w * (xh - T).abs()).mean() + cfg.latent_refine_mu * 2 * (1 - cosz).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return se2.to01(se2.decode(se2.spherify(u), y, 0.0))[0]

    def target_order(self, azs):
        if self.cfg.order == "alternate":
            return sorted(azs, key=lambda a: (abs(a), -np.sign(a)))
        return sorted([a for a in azs if a > 0]) + sorted([a for a in azs if a < 0], reverse=True)

    # ------------------------------------------------------------ main loop
    def run_scene(self, x0, alpha0, K, src_az, src_el, radius, target_azs, gt_depth0=None, H=256, W=256,
                  gt_depth_fn=None):
        """Orbit cameras (object-centric GSO). gt_depth_fn(az) -> GT depth (oracle only)."""
        dev = x0.device
        w2c0 = torch.tensor(orbit_w2c(src_az, src_el, radius), dtype=torch.float32, device=dev)
        targets = [(az, torch.tensor(orbit_w2c(az, src_el, radius), dtype=torch.float32, device=dev))
                   for az in self.target_order(target_azs)]

        def scale_fn(d0, m0):
            if self.cfg.depth_scale == "gt" and gt_depth0 is not None:
                return align_scale_shift(d0, gt_depth0, m0)[0]
            if self.cfg.depth_scale == "limb":  # silhouette depth ~ distance to the orbit centre
                return radius / torch.quantile(d0[m0], 0.95).item()
            return orbit_unitbox_scale(d0, m0, K, w2c0, radius)

        return self.run_cams(x0, alpha0, K, w2c0, targets, scale_fn, src_key=src_az, gt_depth0=gt_depth0,
                             gt_depth_fn=gt_depth_fn, H=H, W=W)

    def run_cams(self, x0, alpha0, K, w2c0, targets, scale_fn, src_key=0.0, gt_depth0=None, gt_depth_fn=None,
                 H=256, W=256):
        """Generic cameras. targets: ordered list of (key, w2c [4,4]); scale_fn(d0, m0) -> s
        maps the source monocular depth to the camera units; gt_depth_fn(key) (oracle)."""
        cfg, se2 = self.cfg, self.se2
        dev = x0.device
        e = se2.sample_noise(1, torch.Generator(device="cuda").manual_seed(cfg.seed))
        y = self.class_label(x0.permute(2, 0, 1))
        m0 = alpha0 > 0.5
        if cfg.depth_source in ("gt", "gt_src"):
            d0 = gt_depth0
        else:
            d0, mv = self.depth(x0.permute(2, 0, 1), fov_x_from_K(K, W))
            m0 = m0 & mv
            s = scale_fn(d0, m0)
            d0 = d0 * s
            self.last_scale = s
        P = PointCloud(dev)
        P.add_view(x0, d0, K, w2c0, m0, view_id=0, edge_thresh=cfg.pc_edge_thresh, erode=cfg.pc_erode)

        views = [{"az": src_key, "rgb": x0, "mask": m0, "w2c": w2c0, "K": K, "generated": False}]
        inter = []
        t0 = time.time()
        for k, tgt in enumerate(targets, start=1):
            az, w2c = tgt[0], tgt[1]
            Kt = tgt[2] if len(tgt) > 2 else K   # per-target intrinsics (real photos)
            rgb_w, known_obj, d_w = P.render(Kt, w2c, H, W)
            rgb_b, known = with_background(rgb_w, known_obj, margin=cfg.bg_margin)
            if cfg.hygiene:
                rgb_b, known, _ = mask_hygiene(rgb_b, known)
            x_pre = prefill_object(rgb_b, known, known_obj, cfg.prefill) if cfg.prefill_hull else prefill(rgb_b, known, cfg.prefill)
            x_warp = rgb_b.permute(2, 0, 1)
            x = self.project(x_warp, known.float(), x_pre.permute(2, 0, 1), y,
                             e if cfg.share_e else se2.sample_noise(1), cfg.schedule, cfg.lam, obj_mask=known_obj)
            if cfg.clean_alpha >= 0:
                x = self.project(x_warp, torch.zeros_like(known.float()), x, y, e, [cfg.clean_alpha], 1.0, obj_mask=known_obj)
            if cfg.latent_refine_steps > 0:
                x = self.latent_refine(x, x_warp, known, y)
            if cfg.final_paste == "hard":
                x = torch.where(known[None], x_warp, x)
            elif cfg.final_paste == "feather":
                x = feather_paste(x, x_warp, known, cfg.feather_px)
            x_hwc = x.permute(1, 2, 0).contiguous()
            region = hull_mask(known_obj, cfg.new_px_hull_dilate)
            fg = (nonwhite(x_hwc, cfg.fg_thresh) & region) | known_obj
            new = fg & ~known_obj
            if new.any():
                if cfg.depth_source in ("gt", "gt_new"):
                    dk = gt_depth_fn(az)
                    mv = dk > 0
                else:
                    dk, mv = self.depth(x, fov_x_from_K(Kt, W))
                    if cfg.depth_fusion == "harmonic":
                        dk = fuse_depth(dk, d_w, known_obj & mv, new | known_obj)
                    else:
                        s_, t_ = align_scale_shift(dk, d_w, known_obj & mv)
                        dk = s_ * dk + t_
                P.add_view(x_hwc, dk, Kt, w2c, new & mv & (dk > 0), view_id=k, edge_thresh=cfg.pc_edge_thresh,
                           erode=0)  # do not erode the thin band of new pixels
            v_rec = {"az": az, "rgb": x_hwc, "mask": fg, "w2c": w2c, "K": Kt, "generated": True}
            if cfg.save_views:
                v_rec.update({"d_render": d_w, "known_obj": known_obj, "new": new,
                              "d_new": dk if new.any() else torch.zeros_like(d_w)})
            views.append(v_rec)
            inter.append({"az": az, "warp": x_warp, "pre": x_pre.permute(2, 0, 1), "out": x,
                          "known": known.float()})
        gen_time = time.time() - t0

        for sweep in range(cfg.final_sweeps):
            gs = self.fit(views, K, P)
            for v in views:
                if not v["generated"]:
                    continue
                with torch.no_grad():
                    r, a, _, _ = render_gs(gs, v["w2c"], v["K"], H, W)
                xr = r[0].clamp(0, 1).permute(2, 0, 1)
                m = torch.ones(H, W, device=dev)
                x = self.project(xr, m, xr, y, e, cfg.sweep_schedule, cfg.sweep_lam)
                v["rgb"] = x.permute(1, 2, 0).contiguous()
        return views, inter, {"gen_seconds": gen_time, "n_points": P.n}, P

    def fit(self, views, K, P=None, iters=None):
        imgs = torch.stack([v["rgb"] for v in views])
        w2cs = torch.stack([v["w2c"] for v in views])
        K = torch.stack([torch.as_tensor(v["K"], dtype=torch.float32, device=imgs.device) for v in views])
        if P is not None and P.n > 0:
            xyz, rgb = P.xyz, P.rgb
        else:
            raise ValueError("need a point cloud for init")
        return fit_gs(imgs, w2cs, K, xyz, rgb, iters=iters or self.cfg.gs_iters, sh_degree=self.cfg.gs_sh,
                      lpips_weight=self.cfg.gs_lpips)


# ---------------------------------------------------------------- GSO driver
def oracle_depth_fn(obj, K):
    """GT depth at any orbit camera (elev 30, r 1.5), rendered from the aligned mesh."""
    from se3d.data.render_gso import DATA as DROOT, SD_MESHES
    from se3d.geometry.render_mesh import Renderer, TexturedMesh

    align = json.load(open(f"{DROOT}/gso30_syncdreamer/alignment.json"))
    mesh = TexturedMesh(f"{SD_MESHES}/{obj}/meshes/model.obj", axis=np.array(align[obj]["R"]))
    rend = Renderer()
    Kn = K.cpu().numpy() if torch.is_tensor(K) else K
    return lambda az: rend.render(mesh, Kn, orbit_w2c(az, 30.0, 1.5), 256, 256)["depth"]


def run_gso(args, cfg):
    from se3d.gs.gt_ceiling import DATA, eval_views, load_views

    r1 = R1(cfg)
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    targets = [a for a in np.arange(-args.max_az, args.max_az + 1e-6, cfg.step_deg) if abs(a) > 1e-6]
    per = {}
    for obj in args.objects:
        dev_v = load_views(osp.join(DATA, "dev_orbit", obj))
        azs = [fr["azimuth"] for fr in dev_v["frames"]]
        x0, a0, d0 = dev_v["rgb"][0], dev_v["alpha"][0], dev_v["depth"][0]
        K = dev_v["K"]
        gt_fn = None
        if cfg.depth_source in ("gt", "gt_new"):
            gt_fn = oracle_depth_fn(obj, K)
        t0 = time.time()
        views, inter, info, P = r1.run_scene(x0, a0, K, 0.0, 30.0, 1.5, list(targets), gt_depth0=d0, gt_depth_fn=gt_fn)
        info["scene_seconds"] = time.time() - t0
        # per-view metrics vs GT (where the dev_orbit has the azimuth)
        gen, gts, masks = [], [], []
        for v in views[1:]:
            if float(v["az"]) in azs:
                j = azs.index(float(v["az"]))
                gen.append(v["rgb"].permute(2, 0, 1))
                gts.append(dev_v["rgb"][j].permute(2, 0, 1))
                masks.append(dev_v["alpha"][j])
        view_m = all_metrics(torch.stack(gen), torch.stack(gts), torch.stack(masks)) if gen else {}
        # GS on generated views vs GS on GT views at the same cameras, held-out GT
        ring = load_views(osp.join(DATA, "ring32", obj))
        r_az = [((fr["azimuth"] + 180) % 360) - 180 for fr in ring["frames"]]
        inside = [i for i, a in enumerate(r_az) if abs(a) < args.max_az - 1e-6 and min(abs(a - t) for t in [0.0] + list(targets)) > 1.0]
        outside = [i for i, a in enumerate(r_az) if abs(a) > args.max_az + 1e-6]
        sub = lambda idx: {k: (v[idx] if torch.is_tensor(v) and v.ndim > 2 else v) for k, v in ring.items() if k != "frames"}
        gs = r1.fit(views, K, P)
        gs_m = {"interleaved": eval_views(gs, sub(inside))[0], "extrapolated": eval_views(gs, sub(outside))[0]}
        # ceiling: GT views at the same cameras, same init recipe (GT depth points)
        gt_views = load_views(osp.join(DATA, "dev_orbit", obj), [azs.index(float(v["az"])) for v in views if float(v["az"]) in azs])
        xyz, rgb = points_from_views(gt_views["rgb"], gt_views["depth"], gt_views["alpha"], gt_views["w2c"], K, stride=2)
        gs_gt = fit_gs(gt_views["rgb"], gt_views["w2c"], K, xyz, rgb, iters=cfg.gs_iters, sh_degree=1)
        gs_m["gt_interleaved"] = eval_views(gs_gt, sub(inside))[0]
        gs_m["gt_extrapolated"] = eval_views(gs_gt, sub(outside))[0]
        per[obj] = {"views": view_m, "gs": gs_m, "info": info, "n_interleaved": len(inside), "n_extrapolated": len(outside)}
        if args.video:  # smooth path over the covered range: input | true | 3D splat of our views | 3D splat of true views
            from se3d.eval.gs_video import write_panels
            from se3d.data.render_gso import DATA as DROOT, SD_MESHES
            from se3d.geometry.render_mesh import Renderer, TexturedMesh

            align = json.load(open(f"{DROOT}/gso30_syncdreamer/alignment.json"))
            mesh = TexturedMesh(f"{SD_MESHES}/{obj}/meshes/model.obj", axis=np.array(align[obj]["R"]))
            rend = Renderer()
            Kn = K.cpu().numpy() if torch.is_tensor(K) else np.asarray(K)
            Wp = [orbit_w2c(a, 30.0, 1.5) for a in np.linspace(-args.max_az, args.max_az, 121)]
            gtp = torch.stack([rend.render(mesh, Kn, w, 256, 256)["rgb"].permute(2, 0, 1) for w in Wp])

            def path(g):
                with torch.no_grad():
                    return torch.cat([render_gs(g, torch.tensor(np.stack(Wp[i:i + 8]), dtype=torch.float32, device="cuda"), K, 256, 256)[0]
                                      for i in range(0, len(Wp), 8)]).permute(0, 3, 1, 2).clamp(0, 1)

            ours, ceil = path(gs), path(gs_gt)
            per[obj]["video_path"] = {"ours_lpips": float(np.mean([all_metrics(ours[i:i + 1], gtp[i:i + 1])["lpips_vgg"] for i in range(len(Wp))])),
                                      "ours_psnr": float(np.mean([all_metrics(ours[i:i + 1], gtp[i:i + 1])["psnr"] for i in range(len(Wp))]))}
            write_panels(osp.join(out_root, f"{obj}_video.mp4"),
                         [("input photo", x0.permute(2, 0, 1)[None]), ("true view", gtp), ("ours: 3D splat of our views", ours),
                          ("3D splat of TRUE views (same cameras)", ceil)])
        # grids: rows = targets in order; cols = warp, prefill, output, GT
        rows = []
        for it in inter:
            gt = dev_v["rgb"][azs.index(float(it["az"]))].permute(2, 0, 1) if float(it["az"]) in azs else torch.ones_like(it["out"])
            rows += [it["warp"], it["pre"], it["out"], gt]
        save_image(torch.stack(rows), osp.join(out_root, f"{obj}_steps.png"), nrow=4)
        if cfg.save_views:
            os.makedirs(osp.join(out_root, f"{obj}_views"), exist_ok=True)
            gt_fn = oracle_depth_fn(obj, K)
            for v in views[1:]:
                np.savez_compressed(osp.join(out_root, f"{obj}_views", f"az{int(v['az']):+04d}.npz"),
                                    rgb=v["rgb"].cpu().numpy(), d_render=v["d_render"].cpu().numpy(),
                                    d_new=v["d_new"].cpu().numpy(), known_obj=v["known_obj"].cpu().numpy(),
                                    new=v["new"].cpu().numpy(), d_gt=gt_fn(v["az"]).cpu().numpy())
        ring_imgs = [v["rgb"].permute(2, 0, 1) for v in sorted(views, key=lambda v: v["az"])]
        save_image(torch.stack(ring_imgs), osp.join(out_root, f"{obj}_ring.png"), nrow=len(ring_imgs))
        print(obj, json.dumps({"view_lpips": round(view_m.get("lpips_vgg", -1), 3), "view_psnr": round(view_m.get("psnr", -1), 2),
                               "gs_int": round(gs_m["interleaved"]["psnr"], 2), "gs_ext": round(gs_m["extrapolated"]["psnr"], 2),
                               "gt_int": round(gs_m["gt_interleaved"]["psnr"], 2), "gt_ext": round(gs_m["gt_extrapolated"]["psnr"], 2),
                               "sec": round(info["scene_seconds"], 1)}), flush=True)
    keys = lambda d: [k for k in d if isinstance(d[k], float)]
    summ = {"views": {k: float(np.mean([per[o]["views"][k] for o in per])) for k in keys(per[args.objects[0]]["views"])}}
    for s in ["interleaved", "extrapolated", "gt_interleaved", "gt_extrapolated"]:
        summ["gs_" + s] = {k: float(np.mean([per[o]["gs"][s][k] for o in per])) for k in per[args.objects[0]]["gs"][s]}
    summ["scene_seconds"] = float(np.mean([per[o]["info"]["scene_seconds"] for o in per]))
    json.dump({"config": dataclasses.asdict(cfg), "summary": summ, "per_object": per}, open(osp.join(out_root, "metrics.json"), "w"), indent=1)
    print("SUMMARY", json.dumps({k: {m: round(v, 3) for m, v in d.items()} if isinstance(d, dict) else round(d, 2) for k, d in summ.items()}))
    print("R1_DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="+", default=["bell", "chicken", "hat", "shoe1", "teapot", "toaster", "turtle"])
    p.add_argument("--max_az", type=float, default=90.0)
    p.add_argument("--video", action="store_true", help="also write a smooth-path mp4 (true | GS of our views | GS of true views)")
    p.add_argument("--out", required=True)
    for f in dataclasses.fields(R1Config):
        t = f.type if f.type in (int, float, str) else None
        if f.type in ("int", "float", "str", int, float, str):
            p.add_argument(f"--{f.name}", type={"int": int, "float": float, "str": str}.get(f.type, f.type), default=None)
        elif f.type in ("bool", bool):
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:  # lists
            p.add_argument(f"--{f.name}", type=float, nargs="+", default=None)
    args = p.parse_args()
    cfg = R1Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(R1Config) if getattr(args, f.name) is not None})
    print("CONFIG", dataclasses.asdict(cfg), flush=True)
    run_gso(args, cfg)


if __name__ == "__main__":
    main()
