"""Small-sample KID (unbiased MMD^2, cubic polynomial kernel) on Inception-v3
pool features (torch-fidelity's FID Inception). Images: [B,3,H,W] in [0,1]."""
import functools

import torch


@functools.lru_cache(maxsize=1)
def _inception(device):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    return FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()


@torch.no_grad()
def inception_features(x01, batch=50):
    net = _inception(str(x01.device))
    feats = []
    for i in range(0, x01.shape[0], batch):
        u8 = (x01[i:i + batch].clamp(0, 1) * 255).round().to(torch.uint8)
        feats.append(net(u8)[0].double())
    return torch.cat(feats)


def kid(f_gen, f_ref):
    """unbiased MMD^2 with k(x,y) = (x.y/d + 1)^3; f_* are [n,d] double."""
    d = f_gen.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3
    n, m = f_gen.shape[0], f_ref.shape[0]
    kxx, kyy, kxy = k(f_gen, f_gen), k(f_ref, f_ref), k(f_gen, f_ref)
    sxx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    syy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    return float(sxx + syy - 2 * kxy.mean())
