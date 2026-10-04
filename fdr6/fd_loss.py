"""
FD-Loss implementation from
    Representation Frechet Loss for Visual Generation, Yang et al., arXiv:2604.28190
"""

import logging
import os.path as osp
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from fdr6.fdr6_utils import (
    EncoderSpec,
    TimmEncoder,
    get_encoder_specs,
    load_valfd,
    stats_file_path,
)
from sphere.metric import create_metric_feature_extractor
from sphere.utils import nn_concat_all_gather

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# judges
# -----------------------------------------------------------------------------
class DiffInceptionEncoder(nn.Module):
    """
    torch-fidelity InceptionV3 (the FID extractor, same weights as
    fdr6_utils.InceptionEncoder) with a float input path that keeps the
    gradient: the reference forward asserts uint8 and quantizes, which is a
    dead end for autograd. every other op is the reference's, in order, so
    the features match the uint8 path up to the missing rounding.
    """

    def __init__(self, weight_path, device):
        super().__init__()
        self.model, self.feat_dim = create_metric_feature_extractor(
            inception_weight_path=weight_path, activation_dim=2048, device=device
        )

    _BLOCKS = [
        "Conv2d_1a_3x3",
        "Conv2d_2a_3x3",
        "Conv2d_2b_3x3",
        "MaxPool_1",
        "Conv2d_3b_1x1",
        "Conv2d_4a_3x3",
        "MaxPool_2",
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_5d",
        "Mixed_6a",
        "Mixed_6b",
        "Mixed_6c",
        "Mixed_6d",
        "Mixed_6e",
        "Mixed_7a",
        "Mixed_7b",
        "Mixed_7c",
        "AvgPool",
    ]

    def forward(self, x):
        from torch_fidelity.interpolate_compat_tensorflow import (
            interpolate_bilinear_2d_like_tensorflow1x,
        )

        m = self.model
        # [-1, 1] -> [0, 255] float (no rounding), then the reference's own
        # resize + normalization
        x = torch.clamp(x.float() * 0.5 + 0.5, 0.0, 1.0) * 255.0
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x, size=(m.INPUT_IMAGE_SIZE, m.INPUT_IMAGE_SIZE), align_corners=False
        )
        x = (x - 128) / 128
        for name in self._BLOCKS:
            x = getattr(m, name)(x)
        return torch.flatten(x, 1).float(), None


@dataclass
class Judge:
    spec: EncoderSpec
    model: nn.Module
    weight: float
    mu_ref: Tensor  # [D] float64
    sigma_ref: Tensor  # [D, D] float64
    sigma_ref_sqrt: Tensor | None  # [D, D] float64 (eigvalsh path)
    valfd: float | None  # FD(heldout, ref) for FDr logging

    @property
    def label(self):
        return self.spec.label


def build_judge(
    spec: EncoderSpec,
    device,
    stats_dir: str,
    stats_source: str,
    dataset_name: str,
    image_size: int,
    pool: str = "cls",
    weight: float = 1.0,
    use_eigvalsh: bool = True,
    inception_weight_path: str | None = None,
    valfd: dict | None = None,
):
    if spec.kind == "inception":
        assert inception_weight_path is not None
        model = DiffInceptionEncoder(inception_weight_path, device)
    elif spec.kind == "timm":
        model = TimmEncoder(spec.model_name, spec.target_size, device)
    else:
        raise ValueError(f"unknown encoder kind: {spec.kind}")

    path = stats_file_path(stats_dir, stats_source, dataset_name, image_size, spec)
    assert osp.exists(path), (
        f"FDr-6 reference stats not found at: {path}. run "
        f"fdr6/extract_fdr6_stats.py --dataset_name {dataset_name} "
        f"--image_size {image_size} first"
    )
    ref = np.load(path)
    if pool == "avg":
        assert "avg_mu" in ref, (
            f"pool='avg' but {path} has no 'avg_mu' (only timm ViT judges "
            f"store the mean patch token); keys: {list(ref.keys())}"
        )
        mu_key, sigma_key = "avg_mu", "avg_sigma"
    else:
        mu_key, sigma_key = "mu", "sigma"
    mu_ref = torch.tensor(ref[mu_key], device=device, dtype=torch.float64)
    sigma_ref = torch.tensor(ref[sigma_key], device=device, dtype=torch.float64)
    assert (
        mu_ref.shape[0] == model.feat_dim
    ), f"{spec.label}: stats dim {mu_ref.shape[0]} != judge dim {model.feat_dim}"
    sigma_ref_sqrt = sigma_sqrt(sigma_ref) if use_eigvalsh else None

    v = None
    if valfd is not None and spec.label in valfd and valfd[spec.label] > 0:
        v = float(valfd[spec.label])

    logger.info(
        f"[FD judge] {spec.label} ({spec.model_name} @ {spec.target_size}px): "
        f"dim={model.feat_dim}, pool={pool}, weight={weight}, stats={path}, "
        f"eig={'eigvalsh' if use_eigvalsh else 'eigvals'}, valfd={v}"
    )
    return Judge(spec, model, float(weight), mu_ref, sigma_ref, sigma_ref_sqrt, v)


# -----------------------------------------------------------------------------
# frechet distance
# -----------------------------------------------------------------------------
def sigma_sqrt(sigma: Tensor) -> Tensor:
    """sigma^{1/2} of a symmetric PSD matrix by eigendecomposition (one-time)."""
    evals, evecs = torch.linalg.eigh(sigma)
    evals = torch.clamp(evals, min=0)
    return evecs @ torch.diag(evals.sqrt()) @ evecs.T


def frechet_distance(
    mu: Tensor,
    sigma: Tensor,
    mu_ref: Tensor,
    sigma_ref: Tensor,
    sigma_ref_sqrt: Tensor | None = None,
    eig_floor: float = 1e-12,
) -> Tensor:
    """
    differentiable FD between N(mu, sigma) (with grad) and the fixed
    N(mu_ref, sigma_ref). with `sigma_ref_sqrt` the trace term uses the
    symmetric product S_ref^1/2 S S_ref^1/2 and eigvalsh (exact, and much
    faster than eigvals of the non-symmetric S @ S_ref).
    """
    dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype)
    sigma_ref = sigma_ref.to(dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    if sigma_ref_sqrt is not None:
        s = sigma_ref_sqrt.to(dtype)
        M = s @ sigma @ s
        M = 0.5 * (M + M.T)
        evals = torch.linalg.eigvalsh(M)
    else:
        evals = torch.linalg.eigvals(sigma @ sigma_ref).real
    tr_covmean = torch.sqrt(torch.clamp(evals, min=eig_floor)).sum()

    trace_term = torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum()
    return mean_term + trace_term - 2.0 * tr_covmean


def weighted_sums(x: Tensor, w: Tensor | None):
    """
    x : [M, D], w : [M] or None
    returns (n, s1 = sum w x [D], s2 = sum w x x^T [D, D]) in float64
    """
    x = x.double()
    if w is None:
        n = torch.tensor(float(x.shape[0]), device=x.device, dtype=torch.float64)
        return n, x.sum(dim=0), x.T @ x
    w = w.double().reshape(-1, 1)
    return w.sum(), (w * x).sum(dim=0), x.T @ (w * x)


# -----------------------------------------------------------------------------
# generated-side history
# -----------------------------------------------------------------------------
class JudgeEMA(nn.Module):
    """
    detached history of one judge's generated features: mu_ema [D] and
    m2_ema = E[xx^T] [D, D]. stats blends the weighted batch moments in with
    weight (1 - beta) and takes sigma = m2 - mu mu^T (biased, as the reference
    does).

    fill(x) streams rows in while the history is being seeded (a running mean
    of `size` samples); once `ready` it behaves as update. all buffers are
    registered, so the history survives a checkpoint round trip.

    size : number of samples the ema is seeded from, 0 = no seeding
    dim  : feature dimension
    beta : ema decay
    """

    def __init__(self, size: int, dim: int, ema_beta: float = 0.999):
        super().__init__()
        self.size = int(size)
        self.dim = int(dim)
        self.beta = float(ema_beta)

        self.register_buffer("mu_ema", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("m2_ema", torch.zeros(dim, dim, dtype=torch.float64))
        # samples streamed during the fill (running mean until `size`)
        self.register_buffer("n_fill", torch.zeros((), dtype=torch.float64))
        # EMA steps committed (0 -> the next blend takes the batch as is)
        self.register_buffer("n_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))

    # -- state -----------------------------------------------------------------
    @property
    def ready(self) -> bool:
        if self.size == 0:
            return True
        return bool(self.filled.item())

    @property
    def _has_state(self) -> bool:
        """the EMA buffers hold a real estimate (seeded, or stepped at least once)"""
        return bool(self.filled.item()) or int(self.n_steps.item()) > 0

    @property
    def n_history(self) -> int:
        return int(self.n_fill.item()) if not self.ready else self.size

    # -- fill (no grad) --------------------------------------------------------
    @torch.no_grad()
    def fill(self, x: Tensor, w: Tensor | None = None):
        """
        x : [M, D] rows that belong to the generated regime (already masked)
        w : [M] weights of those rows
        """
        if self.ready:
            return self.update(x, w)
        if x.shape[0] == 0:
            return
        n, s1, s2 = weighted_sums(x, w)
        self.mu_ema.add_(s1)
        self.m2_ema.add_(s2)
        self.n_fill.add_(n)
        if float(self.n_fill.item()) >= self.size:
            self.mu_ema.div_(self.n_fill)
            self.m2_ema.div_(self.n_fill)
            self.filled.fill_(1)
            logger.info(
                f"[JudgeEMA] EMA seeded from {float(self.n_fill.item()):.0f} "
                f"samples (beta={self.beta})"
            )

    # -- statistics with grad --------------------------------------------------
    def stats(self, x: Tensor, w: Tensor | None):
        """
        x   : [M, D] gathered batch, with grad
        w   : [M] per-row weights or None
        out : (mu [D], sigma [D, D], n) in float64; n is the effective sample
              count behind the moments
        """
        n_b, s1_b, s2_b = weighted_sums(x, w)
        mu_b = s1_b / n_b
        m2_b = s2_b / n_b
        if not self._has_state:
            mu, m2 = mu_b, m2_b  # nothing to blend with yet
        else:
            b = self.beta
            mu = b * self.mu_ema.clone() + (1.0 - b) * mu_b
            m2 = b * self.m2_ema.clone() + (1.0 - b) * m2_b
        sigma = m2 - torch.outer(mu, mu)
        n_eff = n_b + (self.size if self.ready else float(self.n_fill.item()))
        return mu, sigma, n_eff

    # -- commit (no grad) ------------------------------------------------------
    @torch.no_grad()
    def update(self, x: Tensor, w: Tensor | None = None):
        """
        x : [M, D] detached batch
        w : [M] per-row weights or None
        """
        if x.shape[0] == 0:
            return
        n, s1, s2 = weighted_sums(x, w)
        if n.item() <= 0:
            return
        mu_b, m2_b = s1 / n, s2 / n
        if not self._has_state:
            self.mu_ema.copy_(mu_b)
            self.m2_ema.copy_(m2_b)
        else:
            b = self.beta
            self.mu_ema.mul_(b).add_(mu_b, alpha=1.0 - b)
            self.m2_ema.mul_(b).add_(m2_b, alpha=1.0 - b)
        self.n_steps.add_(1)

    def extra_repr(self):
        return f"size={self.size}, dim={self.dim}, beta={self.beta}"


# -----------------------------------------------------------------------------
# loss
# -----------------------------------------------------------------------------
class MultiJudgeFDLoss(nn.Module):
    """
    x_gen : [B, 3, H, W] decoded images in [-1, 1], with grad
    w     : [B] per-sample weights (anchor weight of the generation regime)
            or None for all ones

    forward returns  sum_j weight_j * fd_j / (fd_j.detach() + norm_eps)
    and fills log_dict with the raw fd_{judge}, fdr_{judge} (fd / valfd when
    the normalizer json exists), fdr_mean, fd_n_eff and fd_history.

    the judges are frozen pretrained models and are NOT part of the
    state_dict (they are rebuilt from their names); only the per-judge
    history (JudgeEMA buffers) is saved and resumed.
    """

    def __init__(
        self,
        judges: list[str],
        image_size: int,
        dataset_name: str,
        stats_dir: str,
        stats_source: str,
        device,
        inception_weight_path: str | None = None,
        judge_weights: list[float] | None = None,
        pool: str = "cls",
        seed_size: int = 50000,
        ema_beta: float = 0.999,
        norm_eps: float = 0.01,
        use_eigvalsh: bool = True,
        gather: bool = True,
        min_eff_samples: float = 2.0,
    ):
        super().__init__()
        assert pool in ["cls", "avg"]
        if judge_weights is None:
            judge_weights = [1.0] * len(judges)
        assert len(judge_weights) == len(judges)

        self.pool = pool
        self.seed_size = int(seed_size)
        self.ema_beta = float(ema_beta)
        self.norm_eps = float(norm_eps)
        self.gather = gather
        self.min_eff_samples = float(min_eff_samples)
        self.device = device

        specs = {s.label: s for s in get_encoder_specs(image_size)}
        unknown = [j for j in judges if j not in specs]
        assert not unknown, f"unknown judges {unknown}; available: {list(specs)}"

        valfd = load_valfd(stats_dir, stats_source, dataset_name, image_size)
        if valfd is None:
            logger.warning("no FDr-6 valfd normalizer found; the loss logs raw FD only")

        # plain list on purpose: keeps the frozen judges out of state_dict()
        self.judges: list[Judge] = [
            build_judge(
                specs[label],
                device=device,
                stats_dir=stats_dir,
                stats_source=stats_source,
                dataset_name=dataset_name,
                image_size=image_size,
                pool=pool,
                weight=wt,
                use_eigvalsh=use_eigvalsh,
                inception_weight_path=inception_weight_path,
                valfd=valfd,
            )
            for label, wt in zip(judges, judge_weights)
        ]
        self.history = nn.ModuleDict(
            {
                j.label: JudgeEMA(self.seed_size, j.model.feat_dim, ema_beta=ema_beta)
                for j in self.judges
            }
        )
        self.log_dict = {}
        self._warned_not_ready = False

    # -- state -----------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return all(h.ready for h in self.history.values())

    @property
    def labels(self) -> list[str]:
        return [j.label for j in self.judges]

    # -- features --------------------------------------------------------------
    def _featurize(self, x: Tensor, autocast_ctx=None) -> dict[str, Tensor]:
        """run every judge on x ([-1, 1]); returns {label: [B, D] float32}"""
        device_type = x.device.type
        out = {}
        for j in self.judges:
            if j.spec.kind == "inception":
                ctx = torch.autocast(device_type=device_type, enabled=False)
            else:
                ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
            with ctx:
                feat, extra = j.model(x)
            if self.pool == "avg" and extra is not None:
                feat = extra
            out[j.label] = feat.float()
        return out

    def _gather(self, feats: dict[str, Tensor], w: Tensor | None):
        if not self.gather:
            return feats, w
        feats = {k: nn_concat_all_gather(v) for k, v in feats.items()}
        if w is not None:
            w = nn_concat_all_gather(w.detach().float().contiguous())
        return feats, w

    @staticmethod
    def _mask(w: Tensor | None, n: int, device) -> Tensor:
        if w is None:
            return torch.ones(n, dtype=torch.bool, device=device)
        return w >= 0.5

    # -- warm-up ---------------------------------------------------------------
    @torch.no_grad()
    def fill(self, x_gen: Tensor, w: Tensor | None = None, autocast_ctx=None):
        """
        seed / advance the generated-side history without a loss (the
        reference seeds it from sampled images). call on every step the loss
        is gated off.
        """
        feats = self._featurize(x_gen.detach(), autocast_ctx)
        feats, w = self._gather(feats, w)
        n = next(iter(feats.values())).shape[0]
        mask = self._mask(w, n, x_gen.device)
        for j in self.judges:
            f = feats[j.label]
            self.history[j.label].fill(f[mask], None if w is None else w[mask])
        self.log_dict = {
            "fd_n_eff": float(w.sum().item()) if w is not None else float(n),
            "fd_history": self.history[self.labels[0]].n_history,
        }

    # -- loss ------------------------------------------------------------------
    def forward(self, x_gen: Tensor, w: Tensor | None = None, autocast_ctx=None):
        zero = torch.zeros((), device=x_gen.device, dtype=torch.float32)

        if not self.ready and not self._warned_not_ready:
            self._warned_not_ready = True
            logger.warning(
                "[MultiJudgeFDLoss] loss opened before the history was filled "
                f"({self.history[self.labels[0]].n_history}/{self.seed_size}); "
                "the statistics start from fewer samples than the reference"
            )

        feats = self._featurize(x_gen, autocast_ctx)
        feats, w = self._gather(feats, w)
        n = next(iter(feats.values())).shape[0]
        n_eff = float(w.sum().item()) if w is not None else float(n)

        log = {"fd_n_eff": n_eff}
        if n_eff < 1e-3:
            # no generated row in the batch: nothing to push, keep history as is
            log["fd_history"] = self.history[self.labels[0]].n_history
            self.log_dict = log
            return zero

        total = zero
        fdrs = []
        for j in self.judges:
            f = feats[j.label]
            mu, sigma, n_stat = self.history[j.label].stats(f, w)
            if float(n_stat) < self.min_eff_samples:
                logger.warning(
                    f"[MultiJudgeFDLoss] {j.label}: {float(n_stat):.1f} effective "
                    f"samples < {self.min_eff_samples}, skipping this step"
                )
                continue
            fd = frechet_distance(mu, sigma, j.mu_ref, j.sigma_ref, j.sigma_ref_sqrt)
            fd_raw = fd.detach()
            if not torch.isfinite(fd_raw):
                logger.warning(
                    f"[MultiJudgeFDLoss] {j.label}: non-finite FD, skipping this step"
                )
                continue
            total = total + j.weight * (fd / (fd_raw + self.norm_eps)).float()
            log[f"fd_{j.label}"] = fd_raw.float()
            if j.valfd is not None:
                fdr = fd_raw.float() / j.valfd
                log[f"fdr_{j.label}"] = fdr
                fdrs.append(fdr)
        if fdrs:
            log["fdr_mean"] = torch.stack(fdrs).mean()

        # commit the detached batch to the history after the graph is built
        # (the stats above only ever read clones of the buffers, so the order
        # does not matter)
        with torch.no_grad():
            for j in self.judges:
                self.history[j.label].update(feats[j.label].detach(), w)
        log["fd_history"] = self.history[self.labels[0]].n_history
        self.log_dict = log
        return total

    def __repr__(self):
        js = ", ".join(f"{j.label}:{j.weight}" for j in self.judges)
        return (
            f"{self.__class__.__name__}(judges=[{js}], pool={self.pool}, "
            f"seed_size={self.seed_size}, "
            f"ema_beta={self.ema_beta}, norm_eps={self.norm_eps}, gather={self.gather})"
        )
