import math
from dataclasses import dataclass, replace

import torch

from .geometry import bdot, exp_map, f, project_tokens, rotate, tangent
from .masks import fill_missing


@dataclass
class SteerConfig:
    steps: int = 400
    mode: str = "rgd"
    lr: float = 0.5
    lr_decay: float = 0.995
    n_restarts: int = 4
    restart_angle: float = 60.0
    dec_alpha: float = 0.0
    lam_cyc: float = 0.1
    lam_seam: float = 0.0
    beta_lpips: float = 0.0
    fill: str = "mirror"
    split: str = "lr"
    seam_band: int = 4
    manifold: str = "sphere"


def seam_tv(x, split="lr", band=4):
    if split == "lr":
        c = x.shape[-1] // 2
        s = x[..., c - band : c + band]
        return (s[..., 1:] - s[..., :-1]).abs().mean((1, 2, 3))
    c = x.shape[-2] // 2
    s = x[..., c - band : c + band, :]
    return (s[..., 1:, :] - s[..., :-1, :]).abs().mean((1, 3))


def _onto(v, manifold):
    return f(project_tokens(v)) if manifold == "tokens" else f(v)


def objective(bb, v, y, mA, cfg: SteerConfig, c=None, lpips_fn=None):
    x = bb.decode(v, cfg.dec_alpha, c)
    nA = mA.sum((1, 2, 3)) * x.shape[1]
    phi = 0.5 * ((mA * (x - y)) ** 2).sum((1, 2, 3)) / nA
    if cfg.beta_lpips > 0 and lpips_fn is not None:
        phi = phi + cfg.beta_lpips * lpips_fn(x, y)
    if cfg.lam_cyc > 0:
        v_re = bb.encode(x, c)
        phi = phi + cfg.lam_cyc * (1 - bdot(v, v_re) / (v.flatten(1).norm(dim=1) * v_re.flatten(1).norm(dim=1)))
    if cfg.lam_seam > 0:
        phi = phi + cfg.lam_seam * seam_tv(x, cfg.split, cfg.seam_band)
    return phi, x


def steer_optimize(bb, y, mA, cfg: SteerConfig = SteerConfig(), c=None, lpips_fn=None, log=None, seed=0, v_init=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        v0 = bb.encode(fill_missing(y, mA, cfg.fill, cfg.split), c) if v_init is None else v_init
        v0 = _onto(v0, cfg.manifold)
    xs, vs, ls, curves = [], [], [], []
    data_cfg = replace(cfg, lam_cyc=0.0, lam_seam=0.0, beta_lpips=0.0)
    for r in range(cfg.n_restarts):
        with torch.no_grad():
            v = v0.clone() if r == 0 else _onto(rotate(v0, bb.random_latent(v0.shape[0], v0.device, g), cfg.restart_angle), cfg.manifold)
        v, curve = _run(bb, v, y, mA, cfg, c, lpips_fn, log, r)
        with torch.no_grad():
            loss, x = objective(bb, v, y, mA, data_cfg, c)
        xs.append(x.clamp(-1, 1))
        vs.append(v)
        ls.append(loss)
        curves.append(curve)
    return torch.stack(xs), torch.stack(vs), torch.stack(ls), curves


def _run(bb, v, y, mA, cfg, c, lpips_fn, log, r):
    curve = []
    if cfg.mode == "rgd":
        lr = math.radians(cfg.lr)
        for t in range(cfg.steps):
            v = v.detach().requires_grad_(True)
            phi, _ = objective(bb, v, y, mA, cfg, c, lpips_fn)
            (grad,) = torch.autograd.grad(phi.sum(), v)
            _count_bwd(bb, cfg, v.shape[0])
            with torch.no_grad():
                rg = tangent(v, grad)
                n = rg.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
                v = _onto(exp_map(v, -lr * math.sqrt(bb.L) * rg / n), cfg.manifold)
            lr *= cfg.lr_decay
            curve.append(phi.detach().cpu())
            if log and t % 50 == 0:
                log({"restart": r, "step": t, "phi": phi.mean().item()})
        return v.detach(), torch.stack(curve)
    if cfg.mode == "adam":
        u = v.clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=cfg.lr)
        for t in range(cfg.steps):
            phi, _ = objective(bb, _onto(u, cfg.manifold), y, mA, cfg, c, lpips_fn)
            opt.zero_grad(set_to_none=True)
            phi.sum().backward()
            _count_bwd(bb, cfg, v.shape[0])
            opt.step()
            with torch.no_grad():
                u.copy_(_onto(u, cfg.manifold))
            curve.append(phi.detach().cpu())
            if log and t % 50 == 0:
                log({"restart": r, "step": t, "phi": phi.mean().item()})
        return _onto(u.detach(), cfg.manifold), torch.stack(curve)
    raise ValueError(cfg.mode)


def _count_bwd(bb, cfg, B):
    bb.counters["dec_bwd"] += B
    if cfg.lam_cyc > 0:
        bb.counters["enc_bwd"] += B
