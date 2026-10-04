import logging
import os

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Sampler
from cli_utils import get_device_type

logger = logging.getLogger(__name__)


class DistributedEvalSampler(Sampler):
    """Shard evaluation indices exactly once, without padding or dropping.

    Ranks may have different numbers of batches, or no samples. Consumers must
    perform distributed collectives after iteration, not once per batch.
    """

    def __init__(self, dataset, num_replicas, rank):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("expected num_replicas >= 1 and 0 <= rank < num_replicas")
        self.indices = range(rank, len(dataset), num_replicas)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def ensure_inception_weights(inception_weight_path):
    """download the torch-fidelity InceptionV3 checkpoint to `inception_weight_path`
    if it is missing. under DDP only rank 0 downloads; the other ranks wait at a
    barrier so they never see a half-written file."""
    import os.path as osp

    from torch.hub import download_url_to_file
    from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3

    is_dist = dist.is_available() and dist.is_initialized()
    rank0 = (not is_dist) or dist.get_rank() == 0
    if rank0 and not osp.isfile(inception_weight_path):
        os.makedirs(osp.dirname(osp.abspath(inception_weight_path)), exist_ok=True)
        logger.info(f"downloading {URL_INCEPTION_V3} -> {inception_weight_path}")
        tmp_path = inception_weight_path + ".part"
        download_url_to_file(URL_INCEPTION_V3, tmp_path, progress=True)
        os.replace(tmp_path, inception_weight_path)
    if is_dist:
        dist.barrier()
    assert osp.isfile(inception_weight_path), f"missing {inception_weight_path}"


def create_metric_feature_extractor(
    inception_weight_path=None, activation_dim=2048, device=get_device_type()
):
    # load InceptionV3
    # https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    assert inception_weight_path is not None
    ensure_inception_weights(inception_weight_path)

    model = FeatureExtractorInceptionV3(
        # name from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/registry.py#L172
        name="inception-v3-compat",
        features_list=[str(activation_dim), "logits_unbiased"],  # [for fid, for isc]
        feature_extractor_weights_path=inception_weight_path,
        feature_extractor_internal_dtype="float32",
    )
    model.to(device).eval().requires_grad_(False)
    return model, activation_dim


def extract_metric_features(loader, model, activation_dim, device=get_device_type()):
    # this function requires inputs strictly normalized to [-1, 1]
    # (e.g. Normalize([0.5], [0.5])).

    # init
    features = []
    logits_unbiased = []

    # float64 sufficient statistics for mean/covariance, accumulated across all
    # samples this rank sees; reduced across ranks after the loop.
    #   n  = sample count
    #   s1 = sum_i x_i               (D,)
    #   s2 = sum_i x_i x_i^T         (D, D)
    n = torch.zeros((), dtype=torch.float64, device=device)
    s1 = torch.zeros(activation_dim, dtype=torch.float64, device=device)
    s2 = torch.zeros(activation_dim, activation_dim, dtype=torch.float64, device=device)

    # loop
    pbar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, batch_data in pbar:
        imgs, clss = batch_data[:2]  # FIXME: ignore the rest
        imgs = imgs.to(device, non_blocking=True)
        clss = clss.to(device, non_blocking=True)

        # check: inputs must be within [-1, 1] before any rescaling, otherwise
        # the clamp below would silently hide a wrongly-normalized loader
        imgs = imgs.float()
        assert torch.all(imgs >= -1) and torch.all(
            imgs <= 1
        ), "input values are out of range [-1, 1]"

        # shift pixel values from [-1, 1] to [0, 1]
        imgs = imgs * 0.5 + 0.5
        imgs = torch.clamp(imgs, min=0.0, max=1.0)

        # bring back to [0, 255] uint8
        imgs = torch.round(imgs * 255.0).to(torch.uint8)

        # forward
        x = model(imgs)
        assert len(x) == 2

        # accumulate float64 sufficient statistics
        feat64 = x[0].double()  # (B, D)
        n += feat64.shape[0]
        s1 += feat64.sum(dim=0)
        s2 += feat64.t() @ feat64

        # keep local features/logits for optional downstream use (e.g. ISC)
        features.append(x[0].data.cpu().numpy())
        logits_unbiased.append(x[1].data.cpu().numpy())

    # reduce sufficient statistics across ranks so mu/sigma cover the full dataset
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        dist.all_reduce(s1, op=dist.ReduceOp.SUM)
        dist.all_reduce(s2, op=dist.ReduceOp.SUM)

    assert n.item() >= 2, "need at least 2 samples to estimate covariance"

    # mu = s1 / n
    mu = s1 / n

    # unbiased covariance (matches np.cov default, ddof=1):
    #   cov = (sum_i x_i x_i^T - (sum_i x_i)(sum_i x_i)^T / n) / (n - 1)
    sigma = (s2 - torch.outer(s1, s1) / n) / (n - 1.0)

    # to float64 numpy
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    assert mu.shape == (activation_dim,)
    assert sigma.shape == (activation_dim, activation_dim)

    # stack local shards
    features = (
        np.concatenate(features, axis=0)
        if features
        else np.empty((0, activation_dim), dtype=np.float32)
    )
    logits_local = (
        np.concatenate(logits_unbiased, axis=0)
        if logits_unbiased
        else np.empty((0, 0), dtype=np.float32)
    )

    # all-gather per-sample logits across ranks so ISC can be computed globally.
    # Exchange shapes first, including from empty ranks. Pad only the transport
    # tensors; remove padding before ISC so every real image counts exactly once.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        shape = torch.tensor(logits_local.shape, dtype=torch.long, device=device)
        shapes = [torch.empty_like(shape) for _ in range(dist.get_world_size())]
        dist.all_gather(shapes, shape)
        shapes = [tuple(s.tolist()) for s in shapes]
        max_rows = max(rows for rows, _ in shapes)
        num_logits = max(cols for _, cols in shapes)
        logits_t = torch.zeros(max_rows, num_logits, dtype=torch.float32, device=device)
        if logits_local.shape[0]:
            logits_t[: logits_local.shape[0]] = torch.from_numpy(logits_local).to(
                device
            )
        gathered = [torch.empty_like(logits_t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, logits_t)
        logits_unbiased = (
            torch.cat(
                [shard[:rows] for shard, (rows, _) in zip(gathered, shapes)], dim=0
            )
            .cpu()
            .numpy()
        )
    else:
        logits_unbiased = logits_local

    logger.info(
        f"metric features (local shard) shape: {features.shape}; "
        f"global logits shape: {logits_unbiased.shape}; "
        f"global sample count: {int(n.item())}"
    )

    return mu, sigma, features, logits_unbiased


def compute_fid(mu1, mu2, sigma1, sigma2):
    assert torch.is_tensor(mu1) and torch.is_tensor(mu2)
    assert torch.is_tensor(sigma1) and torch.is_tensor(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    # FID is numerically sensitive (eigvals of the non-symmetric product
    # Σ1@Σ2); force double precision regardless of caller dtype.
    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()

    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
    fid = a + b - 2 * c
    return fid.item()


def compute_isc(features, splits=10):
    assert torch.is_tensor(features) and features.ndim == 2

    # compute in double precision for numerical stability
    features = features.double()

    # random shuffle rows
    idx = torch.randperm(features.shape[0])
    features = features[idx]

    # calc prob and logits
    prob = features.softmax(dim=1)
    log_prob = features.log_softmax(dim=1)

    # chunk into groups
    prob = prob.chunk(splits, dim=0)
    log_prob = log_prob.chunk(splits, dim=0)

    # calculate score per split
    mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
    kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
    kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
    kl = torch.stack(kl_)

    return kl.mean().item(), kl.std().item()
