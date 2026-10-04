import json
import logging
import os
import os.path as osp
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from sphere.metric import compute_fid, create_metric_feature_extractor

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# encoder registry
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderSpec:
    label: str  # short name used in file names and table headers
    kind: str  # "inception" or "timm"
    model_name: str  # timm model name (ignored for inception)
    target_size: int  # resolution fed to the encoder (inception resizes itself)


# the 256px set reproduces the FD-Loss paper (validation FD on ImageNet:
# inception 1.68, convnext 56.87, dinov2 14.19, mae 0.04, siglip 0.60, clip 5.60)
ENCODER_SPECS_256 = [
    EncoderSpec("inception", "inception", "inception-v3-compat", 299),
    EncoderSpec("convnext", "timm", "convnextv2_base.fcmae_ft_in22k_in1k", 224),
    EncoderSpec("dinov2", "timm", "vit_large_patch14_dinov2.lvd142m", 256),
    EncoderSpec("mae", "timm", "vit_large_patch16_224.mae", 224),
    EncoderSpec("siglip", "timm", "vit_so400m_patch16_siglip_256.v2_webli", 224),
    EncoderSpec("clip", "timm", "vit_large_patch14_clip_224.openai", 256),
]

# the 512px set swaps in the largest native-resolution checkpoint of each family.
# mae has no high-res checkpoint, so it keeps its native 224 and downsamples.
ENCODER_SPECS_512 = [
    EncoderSpec("inception", "inception", "inception-v3-compat", 299),
    EncoderSpec("convnext", "timm", "convnextv2_base.fcmae_ft_in22k_in1k_384", 384),
    EncoderSpec("dinov2", "timm", "vit_large_patch14_dinov2.lvd142m", 518),
    EncoderSpec("mae", "timm", "vit_large_patch16_224.mae", 224),
    EncoderSpec("siglip", "timm", "vit_so400m_patch16_siglip_512.v2_webli", 512),
    EncoderSpec("clip", "timm", "vit_large_patch14_clip_336.openai", 336),
]

ENCODER_LABELS = [s.label for s in ENCODER_SPECS_256]


def get_encoder_specs(image_size):
    """pick the encoder set for a generation resolution. anything below 512
    uses the paper's 256 set; 512 and above use the high-res set."""
    return ENCODER_SPECS_512 if image_size >= 512 else ENCODER_SPECS_256


# -----------------------------------------------------------------------------
# encoders: all take [-1, 1] float images, return (feat, extra)
#   feat  : (B, D) feature used for FD (pool for inception, cls / attn-pool for
#           timm ViTs, spatial mean for CNNs)
#   extra : (B, D) mean patch token for timm ViTs, else None
# -----------------------------------------------------------------------------
class InceptionEncoder(torch.nn.Module):
    """torch-fidelity InceptionV3, the standard FID extractor, so the inception
    FD here is the FID."""

    def __init__(self, weight_path, device):
        super().__init__()
        self.model, self.feat_dim = create_metric_feature_extractor(
            inception_weight_path=weight_path, activation_dim=2048, device=device
        )

    def forward(self, x):
        # [-1, 1] -> uint8 [0, 255]; the extractor resizes to 299 internally
        x = torch.clamp(x.float() * 0.5 + 0.5, 0.0, 1.0)
        x = torch.round(x * 255.0).to(torch.uint8)
        pool, logits = self.model(x)
        # unbiased logits of this batch, kept for the inception score (ISC);
        # read by extract_multi_stats when collect_isc_logits=True
        self.last_logits = logits.float()
        return pool.float(), None


class TimmEncoder(torch.nn.Module):
    """frozen timm backbone: [-1, 1] -> [0, 1] -> resize -> model normalize."""

    def __init__(self, model_name, target_size, device):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        kwargs = dict(pretrained=True, num_classes=0)
        try:
            # ViTs: interpolate position embeddings for non-native input sizes
            self.model = timm.create_model(
                model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs
            )
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)

        self.target_size = target_size
        self.feat_dim = self.model.num_features
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.attn_pool = getattr(self.model, "attn_pool", None)

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        self.register_buffer(
            "mean", torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        )
        logger.info(
            f"[TimmEncoder] {model_name}: feat_dim={self.feat_dim}, "
            f"native={data_cfg['input_size'][-1]}, target={target_size}, "
            f"prefix_tokens={self.num_prefix_tokens}, "
            f"attn_pool={self.attn_pool is not None}"
        )

    def forward(self, x):
        x = torch.clamp(x.float() * 0.5 + 0.5, 0.0, 1.0)
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        x = (x - self.mean) / self.std
        feats = self.model.forward_features(x)

        if feats.ndim == 4:
            # CNN: (B, C, H, W) -> spatial mean
            return feats.mean(dim=(2, 3)).float(), None

        # ViT: (B, N, C)
        patch_tokens = feats[:, self.num_prefix_tokens :]
        avg_token = patch_tokens.mean(dim=1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.attn_pool is not None:
            cls_token = self.attn_pool(feats)
        else:
            cls_token = avg_token
        return cls_token.float(), avg_token.float()


def build_encoders(specs, device, inception_weight_path=None):
    """instantiate every encoder in `specs` on `device`.
    returns a list of (spec, module) in the same order."""
    encoders = []
    for spec in specs:
        if spec.kind == "inception":
            assert inception_weight_path is not None
            enc = InceptionEncoder(inception_weight_path, device)
        elif spec.kind == "timm":
            enc = TimmEncoder(spec.model_name, spec.target_size, device)
        else:
            raise ValueError(f"unknown encoder kind: {spec.kind}")
        encoders.append((spec, enc))
    return encoders


# -----------------------------------------------------------------------------
# statistics
# -----------------------------------------------------------------------------
class GaussianStats:
    """float64 sufficient statistics (n, sum, sum of outer products) of one
    feature stream, reducible across ranks."""

    def __init__(self, dim, device):
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.s1 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, feats):
        f = feats.double()
        self.n += f.shape[0]
        self.s1 += f.sum(dim=0)
        self.s2.addmm_(f.t(), f)

    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.s1, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.s2, op=dist.ReduceOp.SUM)

    def finalize(self):
        """returns (mu, sigma) as float64 numpy; sigma is unbiased (ddof=1),
        matching np.cov and sphere.metric.extract_metric_features."""
        n = self.n
        assert n.item() >= 2, "need at least 2 samples to estimate covariance"
        mu = self.s1 / n
        sigma = (self.s2 - torch.outer(self.s1, self.s1) / n) / (n - 1.0)
        return mu.cpu().numpy(), sigma.cpu().numpy()


def autocast_dtype(device):
    """bfloat16 where the hardware runs it natively (Ampere and newer, xpu),
    float32 elsewhere. on Turing (e.g. RTX 2080 Ti) cuDNN has no bf16 kernels
    for depthwise convolutions, so ConvNeXt fails with 'unable to find an
    engine'; the timm ViTs are cheap enough in fp32 there."""
    device_type = device if isinstance(device, str) else device.type
    if device_type == "cuda":
        idx = (
            device.index
            if hasattr(device, "index") and device.index is not None
            else None
        )
        major, _ = torch.cuda.get_device_capability(idx)
        return torch.bfloat16 if major >= 8 else torch.float32
    if device_type == "xpu":
        return torch.bfloat16
    return torch.float32


ISC_LOGITS_KEY = "_inception_logits"


@torch.inference_mode()
def extract_multi_stats(
    loader, encoders, device, desc="extracting", collect_isc_logits=False
):
    device_type = device if isinstance(device, str) else device.type
    ac_dtype = autocast_dtype(device)
    logger.info(f"[rank {_rank()}] timm encoders run under autocast dtype {ac_dtype}")
    stats = {}
    for spec, enc in encoders:
        stats[spec.label] = {"cls": GaussianStats(enc.feat_dim, device), "avg": None}

    if collect_isc_logits:
        assert any(
            spec.kind == "inception" for spec, _ in encoders
        ), "collect_isc_logits needs an inception encoder"
    logits_local = []

    n_local = 0
    pbar = tqdm(loader, desc=desc, disable=not _is_rank0())
    for batch in pbar:
        imgs = batch[0].to(device, non_blocking=True).float()
        assert torch.all(imgs >= -1) and torch.all(
            imgs <= 1
        ), "input values are out of range [-1, 1]"
        n_local += imgs.shape[0]

        for spec, enc in encoders:
            if spec.kind == "inception":
                feat, extra = enc(imgs)  # float32 path, no autocast
                if collect_isc_logits:
                    logits_local.append(enc.last_logits.cpu().numpy())
            else:
                with torch.autocast(
                    device_type, dtype=ac_dtype, enabled=ac_dtype != torch.float32
                ):
                    feat, extra = enc(imgs)
            stats[spec.label]["cls"].update(feat)
            if extra is not None:
                if stats[spec.label]["avg"] is None:
                    stats[spec.label]["avg"] = GaussianStats(enc.feat_dim, device)
                stats[spec.label]["avg"].update(extra)

    out = {}
    for label, d in stats.items():
        d["cls"].all_reduce()
        mu, sigma = d["cls"].finalize()
        entry = {"mu": mu, "sigma": sigma, "n": int(d["cls"].n.item())}
        # Empty shards never observe patch tokens, but must participate in the
        # same reductions as ranks that did see them.
        has_avg = torch.tensor(int(d["avg"] is not None), device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_avg, op=dist.ReduceOp.MAX)
        if has_avg.item():
            if d["avg"] is None:
                d["avg"] = GaussianStats(mu.shape[0], device)
            d["avg"].all_reduce()
            avg_mu, avg_sigma = d["avg"].finalize()
            entry["avg_mu"], entry["avg_sigma"] = avg_mu, avg_sigma
        out[label] = entry
    logger.info(
        f"[rank {_rank()}] local images: {n_local}, "
        f"global images: {out[encoders[0][0].label]['n']}"
    )
    if collect_isc_logits:
        out[ISC_LOGITS_KEY] = _all_gather_rows(logits_local, device)
        logger.info(f"[rank {_rank()}] global ISC logits: {out[ISC_LOGITS_KEY].shape}")
    return out


def _all_gather_rows(chunks, device):
    local = (
        np.concatenate(chunks, axis=0).astype(np.float32)
        if chunks
        else np.empty((0, 0), dtype=np.float32)
    )
    if not (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    ):
        return local
    shape = torch.tensor(local.shape, dtype=torch.long, device=device)
    shapes = [torch.empty_like(shape) for _ in range(dist.get_world_size())]
    dist.all_gather(shapes, shape)
    shapes = [tuple(s.tolist()) for s in shapes]
    max_rows = max(rows for rows, _ in shapes)
    cols = max(c for _, c in shapes)
    buf = torch.zeros(max_rows, cols, dtype=torch.float32, device=device)
    if local.shape[0]:
        buf[: local.shape[0]] = torch.from_numpy(local).to(device)
    gathered = [torch.empty_like(buf) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, buf)
    return (
        torch.cat([g[:rows] for g, (rows, _) in zip(gathered, shapes)], dim=0)
        .cpu()
        .numpy()
    )


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _is_rank0():
    return _rank() == 0


# -----------------------------------------------------------------------------
# file naming / io
# -----------------------------------------------------------------------------
def stats_file_path(stats_dir, source, dataset_name, image_size, spec):
    return osp.join(
        stats_dir,
        f"fdr6_stats_{source}_{dataset_name}_{image_size}px"
        f"_{spec.label}_t{spec.target_size}.npz",
    )


def valfd_file_path(stats_dir, source, dataset_name, image_size):
    return osp.join(
        stats_dir, f"fdr6_valfd_{source}_{dataset_name}_{image_size}px.json"
    )


def save_stats(stats, specs, stats_dir, source, dataset_name, image_size):
    os.makedirs(stats_dir, exist_ok=True)
    paths = []
    for spec in specs:
        entry = stats[spec.label]
        path = stats_file_path(stats_dir, source, dataset_name, image_size, spec)
        payload = {"mu": entry["mu"], "sigma": entry["sigma"]}
        if "avg_mu" in entry:
            payload["avg_mu"] = entry["avg_mu"]
            payload["avg_sigma"] = entry["avg_sigma"]
        np.savez(path, **payload)
        paths.append(path)
        logger.info(f"saved {path} (n={entry['n']}, dim={entry['mu'].shape[0]})")
    return paths


def load_ref_stats(specs, stats_dir, source, dataset_name, image_size):
    paths = {
        spec.label: stats_file_path(stats_dir, source, dataset_name, image_size, spec)
        for spec in specs
    }
    missing = [path for path in paths.values() if not osp.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"FDr-6 stats for source {source!r} are missing:\n" + "\n".join(missing)
        )
    ref = {}
    for label, path in paths.items():
        d = np.load(path)
        ref[label] = {"mu": d["mu"], "sigma": d["sigma"]}
    return ref


def load_valfd(stats_dir, source, dataset_name, image_size):
    path = valfd_file_path(stats_dir, source, dataset_name, image_size)
    if not osp.exists(path):
        return None
    with open(path, "r") as f:
        d = json.load(f)
    return d["valfd"]


def save_valfd(valfd, stats_dir, source, dataset_name, image_size, meta=None):
    os.makedirs(stats_dir, exist_ok=True)
    path = valfd_file_path(stats_dir, source, dataset_name, image_size)
    payload = {"valfd": {}}
    if osp.exists(path):
        with open(path, "r") as f:
            payload = json.load(f)
    payload["valfd"].update(valfd)
    if meta:
        for k, v in meta.items():
            if isinstance(v, dict) and isinstance(payload.get(k), dict):
                payload[k].update(v)
            else:
                payload[k] = v
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info(f"saved {path}")
    return path


# -----------------------------------------------------------------------------
# metric
# -----------------------------------------------------------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2):
    return compute_fid(
        torch.from_numpy(np.asarray(mu1)),
        torch.from_numpy(np.asarray(mu2)),
        torch.from_numpy(np.asarray(sigma1)),
        torch.from_numpy(np.asarray(sigma2)),
    )


def compute_fdr6(gen_stats, ref_stats, valfd=None, labels=ENCODER_LABELS):
    """returns a dict with
    fd_{label}  : raw FD(gen, ref) per encoder
    fdr_{label} : FD(gen, ref) / valfd[label]   ('-' when valfd is missing)
    fdr6        : mean of fdr_{label} over `labels` ('-' when valfd is missing)
    """
    out = {}
    fdrs = []
    for label in labels:
        fd = frechet_distance(
            gen_stats[label]["mu"],
            gen_stats[label]["sigma"],
            ref_stats[label]["mu"],
            ref_stats[label]["sigma"],
        )
        out[f"fd_{label}"] = fd
        if valfd is not None and label in valfd and valfd[label] > 0:
            out[f"fdr_{label}"] = fd / valfd[label]
            fdrs.append(out[f"fdr_{label}"])
        else:
            out[f"fdr_{label}"] = "-"
    out["fdr6"] = float(np.mean(fdrs)) if len(fdrs) == len(labels) else "-"
    return out


def fdr6_table_columns(labels=ENCODER_LABELS):
    return ["fdr6"] + [f"fdr_{l}" for l in labels] + [f"fd_{l}" for l in labels]
