"""
Representation frechet distance (fd) loss in the model's own encoder space:
match the mean and covariance of the generated features to the real ones.

  - judge     : the model's own encoder
  - reference : statistics of the clean training images, kept as an ema and
                warmed up while the loss is gated off
  - generated : weighted per sample (anchor weights), not masked

inputs are latents [B, N, D] or [B, D]; `pool` reduces the token axis:

    mean  : token mean per image          -> [B, D]
    token : every token is its own sample -> [B * N, D]
    cls   : the backbone's cls token, passed in as [B, D]
"""

import logging
import torch
import torch.nn as nn

from sphere.utils import nn_concat_all_gather

logger = logging.getLogger(__name__)


"""
helpers
"""


def precompute_sigma_sqrt(sigma: torch.Tensor):
    """
    compute sigma^{1/2} via eigen decomposition.
    """
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    eigvals = torch.clamp(eigvals, min=1e-10)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def pool_tokens(z: torch.Tensor, w: torch.Tensor | None, pool: str):
    """
    reduce the token axis of the latents; [B, D] inputs pass through as is

    z    : [B, N, D] or [B, D]
    w    : [B] per-image weights or None
    pool : mean (token mean, M = B) | token (every token a sample, M = B * N)
           | cls (expects [B, D])
    out  : (x [M, D], w [M] or None)
    """
    if z.ndim == 2:
        return z, w
    assert (
        pool != "cls"
    ), f"pool='cls' expects [B, D] CLS features, got {tuple(z.shape)}"
    assert z.ndim == 3, f"expected [B, N, D] or [B, D], got {tuple(z.shape)}"
    B, N, D = z.shape
    if pool == "mean":
        return z.mean(dim=1), w
    if pool == "token":
        x = z.reshape(B * N, D)
        w = w.repeat_interleave(N) if w is not None else None
        return x, w
    raise ValueError(f"unknown pool: {pool!r}")


"""
loss
"""


class FDLoss(nn.Module):
    """
    dim             : feature dimension
    ema_decay       : decay of the ema statistics
    use_clean_ema   : ema for the reference side, else the current batch
    use_noisy_ema   : ema for the generated side, else the current batch
    pool            : mean | token | cls, see pool_tokens
    gather          : all-gather the features across ranks
    cov_eps         : added to the diagonal of the covariances
    normalize_fid   : divide the loss by its detached value + norm_eps
    norm_eps        : see normalize_fid
    min_eff_samples : return zero when the generated side has fewer effective
                      samples (batch + ema history)

    forward:
    z_clean : [B, N, D] or [B, D], encoder(real image), detached inside
    z_noisy : [B, N, D] or [B, D], encoder(fake image), carries grad
    w_noisy : [B] per-sample weight for the generated side, or None
    out     : scalar loss
    """

    def __init__(
        self,
        dim: int = 768,
        ema_decay: float = 0.999,
        use_clean_ema: bool = True,
        use_noisy_ema: bool = False,
        pool: str = "mean",
        gather: bool = True,
        cov_eps: float = 1e-6,
        normalize_fid: bool = True,
        norm_eps: float = 0.01,
        min_eff_samples: float = 2.0,
    ):
        super().__init__()
        assert pool in ["mean", "token", "cls"]
        self.dim = dim
        self.m = ema_decay
        self.pool = pool
        self.gather = gather
        self.cov_eps = cov_eps
        self.normalize_fid = normalize_fid
        self.norm_eps = norm_eps
        self.min_eff_samples = min_eff_samples

        self.use_clean_ema = use_clean_ema
        self.use_noisy_ema = use_noisy_ema

        for mode in ["clean", "noisy"]:
            self.register_buffer(
                f"mu_{mode}_ema", torch.zeros(dim, dtype=torch.float64)
            )
            self.register_buffer(
                f"m2_{mode}_ema", torch.zeros(dim, dim, dtype=torch.float64)
            )
            self.register_buffer(f"_n_{mode}_ema", torch.zeros((), dtype=torch.long))

        self.log_dict = {}

    """
    moments
    """

    @staticmethod
    def _weighted_mu_m2(x: torch.Tensor, w: torch.Tensor | None):
        """
        x   : [M, D]
        w   : [M] or None
        out : mu [D], m2 = E[x x^T] [D, D], both float64
        """
        x = x.double()
        if w is None:
            mu = x.mean(dim=0)
            m2 = (x.T @ x) / x.shape[0]
            return mu, m2
        w = w.double().reshape(-1, 1)
        s = w.sum()
        mu = (w * x).sum(dim=0) / s
        m2 = (x.T @ (w * x)) / s
        return mu, m2

    def _compute_sigma(self, mu: torch.Tensor, m2: torch.Tensor):
        sigma = m2 - torch.outer(mu, mu)
        sigma = 0.5 * (sigma + sigma.T)
        eye = torch.eye(self.dim, device=sigma.device, dtype=sigma.dtype)
        return sigma + self.cov_eps * eye

    def _step_ema(
        self,
        mu_b: torch.Tensor,
        m2_b: torch.Tensor,
        mode: str,
        debias: bool = False,
        n_batch: int = 0,
    ):
        n_ema = getattr(self, f"_n_{mode}_ema")
        n = float(n_ema.item())
        if n <= 0:
            beta = 0.0
        elif debias:
            beta = min(n / max(n + float(n_batch), 1.0), self.m)
        else:
            beta = self.m
        mu = beta * getattr(self, f"mu_{mode}_ema") + (1.0 - beta) * mu_b
        m2 = beta * getattr(self, f"m2_{mode}_ema") + (1.0 - beta) * m2_b
        return mu, m2

    @torch.no_grad()
    def _commit_ema(self, mu: torch.Tensor, m2: torch.Tensor, mode: str, n: int):
        getattr(self, f"mu_{mode}_ema").copy_(mu.detach())
        getattr(self, f"m2_{mode}_ema").copy_(m2.detach())
        getattr(self, f"_n_{mode}_ema").add_(int(n))

    def _accumulate(self, x: torch.Tensor, w: torch.Tensor | None, mode: str):
        use_ema = self.use_clean_ema if mode == "clean" else self.use_noisy_ema
        if use_ema:
            mu_b, m2_b = self._weighted_mu_m2(x, w)
            mu, m2 = self._step_ema(mu_b, m2_b, mode, debias=True, n_batch=x.shape[0])
            self._commit_ema(mu, m2, mode, x.shape[0])

    def _stats(self, x: torch.Tensor, w: torch.Tensor | None, mode: str, use_ema: bool):
        mu_b, m2_b = self._weighted_mu_m2(x, w)
        if use_ema:
            return self._step_ema(mu_b, m2_b, mode)
        return mu_b, m2_b

    """
    frechet distance
    """

    def _trace_term(self, sigma: torch.Tensor, sigma_ref: torch.Tensor):
        sigma_ref_sqrt = precompute_sigma_sqrt(sigma_ref.detach())
        M = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
        M = 0.5 * (M + M.T)
        evals = torch.linalg.eigvalsh(M)
        evals = torch.clamp(evals, min=1e-10)
        tr_covmean = torch.sqrt(evals).sum()
        return (
            torch.diagonal(sigma).sum()
            + torch.diagonal(sigma_ref).sum()
            - 2.0 * tr_covmean
        )

    def _fd(self, mu_ref, sigma_ref, mu, sigma):
        diff = mu - mu_ref
        return diff.dot(diff) + self._trace_term(sigma, sigma_ref)

    """
    api
    """

    def _prep(self, z: torch.Tensor, w: torch.Tensor | None, with_grad: bool):
        x, w = pool_tokens(z.float(), w, self.pool)
        if not with_grad:
            x = x.detach()
        if self.gather:
            x = nn_concat_all_gather(x)
            if w is not None:
                w = nn_concat_all_gather(w.detach().float().contiguous())
        return x, w

    @torch.no_grad()
    def update_reference(self, z_clean: torch.Tensor):
        x, _ = self._prep(z_clean, None, with_grad=False)
        self._accumulate(x, None, "clean")

    @torch.no_grad()
    def update_generated(
        self, z_noisy: torch.Tensor, w_noisy: torch.Tensor | None = None
    ):
        x, w = self._prep(z_noisy, w_noisy, with_grad=False)
        self._accumulate(x, w, "noisy")

    def forward(
        self,
        z_clean: torch.Tensor,
        z_noisy: torch.Tensor,
        w_noisy: torch.Tensor | None = None,
    ):
        y, _ = self._prep(z_clean, None, with_grad=False)
        x, w = self._prep(z_noisy, w_noisy, with_grad=True)

        zero = torch.zeros((), device=z_noisy.device, dtype=torch.float32)

        # ---- reference (clean) ----
        mu_c, m2_c = self._stats(y, None, "clean", use_ema=self.use_clean_ema)
        mu_c, m2_c = mu_c.detach(), m2_c.detach()

        # ---- generated (noisy): need enough effective samples. `w` is the
        # gathered global weight vector, so this decision is identical on
        # every rank.
        n_eff = float(w.sum().item()) if w is not None else float(x.shape[0])
        n_hist = int(self._n_noisy_ema.item()) if self.use_noisy_ema else 0
        if n_eff < 1e-3 or (n_eff + n_hist) < self.min_eff_samples:
            self._finish(y, x, w, mu_c, m2_c, None, None)
            self.log_dict = {"fd_raw": 0.0, "fd_n_eff": n_eff}
            return zero

        mu_n, m2_n = self._stats(x, w, "noisy", use_ema=self.use_noisy_ema)

        sigma_c = self._compute_sigma(mu_c, m2_c)
        sigma_n = self._compute_sigma(mu_n, m2_n)

        fd = self._fd(mu_ref=mu_c, sigma_ref=sigma_c, mu=mu_n, sigma=sigma_n)
        fd_raw = fd.detach()
        if self.normalize_fid:
            fd = fd / (fd_raw + self.norm_eps)

        self._finish(y, x, w, mu_c, m2_c, mu_n, m2_n)
        self.log_dict = {"fd_raw": fd_raw.float(), "fd_n_eff": n_eff}
        return fd.float()

    @torch.no_grad()
    def _finish(self, y, x, w, mu_c, m2_c, mu_n, m2_n):
        if self.use_clean_ema:
            self._commit_ema(mu_c, m2_c, "clean", y.shape[0])
        if mu_n is not None and self.use_noisy_ema:
            self._commit_ema(mu_n, m2_n, "noisy", x.shape[0])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, pool={self.pool}, ema_decay={self.m}, "
            f"clean_ema={self.use_clean_ema}, noisy_ema={self.use_noisy_ema}, "
            f"gather={self.gather}, normalize={self.normalize_fid}, "
            f"norm_eps={self.norm_eps})"
        )
