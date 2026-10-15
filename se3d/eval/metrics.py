"""
Image metrics. Images are float tensors in [0, 1], [B, 3, H, W] (or [3, H, W]).
Masked variants evaluate only where mask > 0.5 (object pixels).
"""
import functools

import torch
import torch.nn.functional as F


def _b(x):
    return x[None] if x.ndim == 3 else x


def _m(mask):
    """mask [H,W] | [B,H,W] | [B,1,H,W] -> [B,1,H,W] float"""
    m = mask.float()
    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[:, None]
    return m


def psnr(pred, gt, mask=None):
    pred, gt = _b(pred).float(), _b(gt).float()
    se = (pred - gt).square()
    if mask is None:
        mse = se.mean((1, 2, 3))
    else:
        m = (_m(mask) > 0.5).float().expand_as(se)
        mse = (se * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp_min(1)
    return -10 * torch.log10(mse.clamp_min(1e-10))


def _gauss(win=11, sigma=1.5, device="cpu"):
    x = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-x.square() / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None]).expand(3, 1, win, win).contiguous()


def ssim(pred, gt, mask=None):
    pred, gt = _b(pred).float(), _b(gt).float()
    w = _gauss(device=pred.device)
    mu1, mu2 = F.conv2d(pred, w, padding=5, groups=3), F.conv2d(gt, w, padding=5, groups=3)
    s11 = F.conv2d(pred * pred, w, padding=5, groups=3) - mu1.square()
    s22 = F.conv2d(gt * gt, w, padding=5, groups=3) - mu2.square()
    s12 = F.conv2d(pred * gt, w, padding=5, groups=3) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1.square() + mu2.square() + c1) * (s11 + s22 + c2))
    if mask is None:
        return m.mean((1, 2, 3))
    mk = (_m(mask) > 0.5).float().expand_as(m)
    return (m * mk).sum((1, 2, 3)) / mk.sum((1, 2, 3)).clamp_min(1)


@functools.lru_cache(maxsize=4)
def _lpips_net(net, device):
    import lpips

    return lpips.LPIPS(net=net, verbose=False).to(device).eval().requires_grad_(False)


def lpips(pred, gt, net="vgg", mask=None):
    """LPIPS (inputs in [0,1]). With a mask, both images are composited on white
    outside the mask first (alpha-masked LPIPS)."""
    pred, gt = _b(pred).float(), _b(gt).float()
    if mask is not None:
        m = _m(mask).clamp(0, 1)
        pred = pred * m + (1 - m)
        gt = gt * m + (1 - m)
    fn = _lpips_net(net, str(pred.device))
    return fn(pred * 2 - 1, gt * 2 - 1).flatten()


def all_metrics(pred, gt, mask=None):
    with torch.no_grad():
        out = {
            "psnr": psnr(pred, gt).mean().item(),
            "ssim": ssim(pred, gt).mean().item(),
            "lpips_vgg": lpips(pred, gt, "vgg").mean().item(),
        }
        if mask is not None:
            out.update({
                "psnr_masked": psnr(pred, gt, mask).mean().item(),
                "ssim_masked": ssim(pred, gt, mask).mean().item(),
                "lpips_vgg_masked": lpips(pred, gt, "vgg", mask).mean().item(),
            })
    return out
