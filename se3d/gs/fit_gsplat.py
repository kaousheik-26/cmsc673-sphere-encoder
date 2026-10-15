"""
Fast object-level 3D Gaussian Splatting fit with gsplat (Phase-0 / R1 / eval harness).

    gs = fit_gs(images, w2cs, K, init_xyz, init_rgb, iters=2000, ...)
    out = render_gs(gs, w2c, K, H, W)   # {"rgb": [H,W,3], "alpha": [H,W], "depth": [H,W]}

images: [N, H, W, 3] in [0,1] composited on a white background; w2cs: [N,4,4]
OpenCV world-to-camera; K: [3,3]. Optional: masks [N,H,W] (only used for
reporting), depths [N,H,W] (0 = unknown) for a depth loss, lpips_weight,
pose_opt (per-view SE(3) refinement, for generated views).
"""
import math
import time

import torch
import torch.nn.functional as F

from se3d.eval.metrics import ssim as ssim_fn


def _knn_scale(xyz, k=4):
    # mean distance to k-1 nearest neighbours, chunked
    d_all = []
    for i in range(0, xyz.shape[0], 4096):
        d = torch.cdist(xyz[i:i + 4096], xyz)
        d_all.append(d.topk(k, largest=False).values[:, 1:].mean(-1))
    return torch.cat(d_all).clamp_min(1e-4)


def _rgb_to_sh0(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def _axis_angle_to_R(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


class GSModel:
    def __init__(self, params, sh_degree):
        self.params = params
        self.sh_degree = sh_degree

    @property
    def n(self):
        return self.params["means"].shape[0]


def render_gs(gs, w2c, K, height, width, background=1.0, sh_degree=None):
    from gsplat import rasterization

    p = gs.params
    w2c = torch.as_tensor(w2c, dtype=torch.float32, device=p["means"].device)
    K = torch.as_tensor(K, dtype=torch.float32, device=p["means"].device)
    if w2c.ndim == 2:
        w2c, K = w2c[None], K[None] if K.ndim == 2 else K
    if K.shape[0] != w2c.shape[0]:
        K = K.expand(w2c.shape[0], 3, 3)
    colors = torch.cat([p["sh0"], p["shN"]], 1)
    sh = gs.sh_degree if sh_degree is None else sh_degree
    bg = torch.full((w2c.shape[0], 3), float(background), device=w2c.device)
    out, alpha, info = rasterization(
        p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
        torch.sigmoid(p["opacities"]), colors, w2c, K, width, height,
        sh_degree=sh, backgrounds=bg, render_mode="RGB+ED", packed=False,
    )
    return out[..., :3], alpha[..., 0], out[..., 3], info


def fit_gs(images, w2cs, K, init_xyz, init_rgb, iters=2000, depths=None, depth_weight=0.0,
           lpips_weight=0.0, ssim_weight=0.2, sh_degree=1, pose_opt=False, max_init=100_000,
           seed=0, log_every=0, scene_scale=1.0):
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(seed)
    dev = images.device
    N, H, W, _ = images.shape
    w2cs = torch.as_tensor(w2cs, dtype=torch.float32, device=dev)
    K = torch.as_tensor(K, dtype=torch.float32, device=dev)

    if init_xyz.shape[0] > max_init:
        idx = torch.randperm(init_xyz.shape[0], device=dev)[:max_init]
        init_xyz, init_rgb = init_xyz[idx], init_rgb[idx]
    n = init_xyz.shape[0]
    dist = _knn_scale(init_xyz)
    nsh = (sh_degree + 1) ** 2 - 1
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(init_xyz.clone()),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.cat([torch.ones(n, 1, device=dev), torch.zeros(n, 3, device=dev)], 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(_rgb_to_sh0(init_rgb)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, nsh, 3, device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / iters))

    strategy = DefaultStrategy(
        refine_start_iter=min(200, iters // 10), refine_stop_iter=int(iters * 0.75),
        refine_every=100, reset_every=10**9, verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    pose_delta = None
    if pose_opt:
        pose_delta = torch.nn.Parameter(torch.zeros(N, 6, device=dev))
        pose_optim = torch.optim.Adam([pose_delta], lr=1e-4)

    lp = None
    if lpips_weight > 0:
        from se3d.eval.metrics import _lpips_net
        lp = _lpips_net("vgg", str(dev))

    gs = GSModel(params, sh_degree)
    t0 = time.time()
    order = torch.randperm(N, device=dev)
    for step in range(iters):
        i = order[step % N].item()
        if step % N == N - 1:
            order = torch.randperm(N, device=dev)
        vm = w2cs[i:i + 1]
        Ki = K[i] if K.ndim == 3 else K
        if pose_delta is not None:
            d = pose_delta[i:i + 1]
            dT = torch.eye(4, device=dev)[None].repeat(1, 1, 1)
            dT[:, :3, :3] = _axis_angle_to_R(d[:, :3])
            dT[:, :3, 3] = d[:, 3:]
            vm = dT @ vm
        sh_now = min(sh_degree, step // 500)
        rgb, alpha, dep, info = render_gs(gs, vm, Ki, H, W, sh_degree=sh_now)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        gt = images[i:i + 1]
        l1 = (rgb - gt).abs().mean()
        loss = l1
        if ssim_weight > 0:
            loss = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim_fn(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).mean())
        if lp is not None:
            loss = loss + lpips_weight * lp(rgb.permute(0, 3, 1, 2) * 2 - 1, gt.permute(0, 3, 1, 2) * 2 - 1).mean()
        if depths is not None and depth_weight > 0:
            dgt = depths[i:i + 1]
            m = dgt > 0
            if m.sum() > 100:
                # scale-invariant (Pearson) depth loss, robust to generated-view depth scale
                a, b = dep[m], dgt[m]
                a = (a - a.mean()) / (a.std() + 1e-6)
                b = (b - b.mean()) / (b.std() + 1e-6)
                loss = loss + depth_weight * (1 - (a * b).mean())
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_delta is not None:
            pose_optim.step()
            pose_optim.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if log_every and (step % log_every == 0 or step == iters - 1):
            print(f"  gs step {step} loss {loss.item():.4f} n={gs.n}", flush=True)
    torch.cuda.synchronize()
    gs.fit_seconds = time.time() - t0
    gs.pose_delta = pose_delta.detach() if pose_delta is not None else None
    return gs


def points_from_views(images, depths, masks, w2cs, K, stride=1):
    """unproject RGB-D views to a colored point cloud (world frame)."""
    from se3d.geometry.cameras import unproject_depth

    xyz, rgb = [], []
    for i in range(images.shape[0]):
        d = depths[i]
        m = (d > 0) if masks is None else ((masks[i] > 0.5) & (d > 0))
        if stride > 1:
            sub = torch.zeros_like(m)
            sub[::stride, ::stride] = True
            m = m & sub
        pw, valid = unproject_depth(d, K, w2cs[i], m)
        xyz.append(pw)
        rgb.append(images[i][valid])
    return torch.cat(xyz), torch.cat(rgb)
