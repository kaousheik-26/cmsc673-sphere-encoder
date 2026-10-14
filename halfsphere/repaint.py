import math
from dataclasses import dataclass

import torch

from .geometry import angle_deg, effective_angle_deg, rotate
from .masks import fill_missing, prox_mask
from .metrics import seam_score


@dataclass
class RepaintConfig:
    K: int = 12
    alpha_start: float = 80.0
    alpha_end: float = 20.0
    schedule: str = "linear"
    lam_start: float = 0.0
    lam_end: float = 0.0
    fill: str = "mirror"
    split: str = "lr"
    share_noise: bool = True
    final_clean: bool = True
    dec_alpha_offset: float = 0.0


def schedule(a, b, K, kind="linear"):
    if K == 1 or kind == "constant":
        return [a] * K
    t = [k / (K - 1) for k in range(K)]
    if kind == "cosine":
        t = [0.5 - 0.5 * math.cos(math.pi * s) for s in t]
    elif kind != "linear":
        raise ValueError(kind)
    return [a + (b - a) * s for s in t]


@torch.no_grad()
def repaint_complete(bb, y, mA, cfg: RepaintConfig = RepaintConfig(), c=None, generator=None, return_traj=False, x_init=None):
    x = fill_missing(y, mA, cfg.fill, cfg.split) if x_init is None else x_init
    alphas = schedule(cfg.alpha_start, cfg.alpha_end, cfg.K, cfg.schedule)
    lams = schedule(cfg.lam_start, cfg.lam_end, cfg.K, "linear")
    B = x.shape[0]
    e_hat = bb.random_latent(B, x.device, generator)
    traj = {"x": [x], "x_pr": [], "steps": []}
    v_prev = None
    for k in range(cfg.K):
        if not cfg.share_noise and k > 0:
            e_hat = bb.noise_like(e_hat, generator)
        v = bb.encode(x, c)
        z = bb.rotate(v, e_hat, alphas[k])
        x_pr = bb.decode(z, alphas[k] + cfg.dec_alpha_offset, c)
        x = prox_mask(x_pr, y, mA, lams[k])
        if return_traj:
            traj["x"].append(x)
            traj["x_pr"].append(x_pr)
            traj["steps"].append(
                {
                    "k": k,
                    "alpha": alphas[k],
                    "lat_move_deg": (angle_deg(v, v_prev) if v_prev is not None else torch.zeros(B, device=x.device)),
                    "resid_A": ((mA * (x_pr - y)) ** 2).sum((1, 2, 3)).div(mA.sum((1, 2, 3)) * 3).sqrt(),
                    "seam": seam_score(x, cfg.split),
                }
            )
        v_prev = v
    if cfg.final_clean:
        x_pr = bb.project(x, 0.0, e_hat, c)
        x = prox_mask(x_pr, y, mA, 0.0)
        if return_traj:
            traj["x"].append(x)
            traj["x_pr"].append(x_pr)
    out = x.clamp(-1, 1)
    if return_traj:
        traj["e_hat"] = e_hat
        traj["x_prepin"] = traj["x_pr"][-1] if traj["x_pr"] else x
        return out, traj
    return out


@torch.no_grad()
def repaint_samples(bb, y, mA, n_samples=8, cfg: RepaintConfig = RepaintConfig(), c=None, seed=0):
    out = []
    for s in range(n_samples):
        g = torch.Generator(device="cpu").manual_seed(seed + s)
        out.append(repaint_complete(bb, y, mA, cfg, c, generator=g))
    return torch.stack(out)


@torch.no_grad()
def repaint_token_complete(bb, y, mA, idxA, idxB, K=12, alphaA=(10.0, 0.0), alphaB=(90.0, 30.0),
                      fill="mirror", split="lr", c=None, generator=None):
    x = fill_missing(y, mA, fill, split)
    v_obs = bb.encode(x, c)
    v = v_obs.clone()
    e_hat = bb.noise_like(v, generator)
    aA, aB = schedule(*alphaA, K), schedule(*alphaB, K)
    for k in range(K):
        z = v.clone()
        z[:, idxA] = rotate(v_obs[:, idxA], e_hat[:, idxA], aA[k])
        z[:, idxB] = rotate(v[:, idxB], e_hat[:, idxB], aB[k])
        fracA = (v_obs[:, idxA].flatten(1).pow(2).sum(1) / bb.L).mean().item()
        x = prox_mask(bb.decode(z, effective_angle_deg(fracA, aA[k], aB[k]), c), y, mA, 0.0)
        v = bb.encode(x, c)
    return x.clamp(-1, 1)
