import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext
from functools import partial
from torch import Tensor
from sphere.lpips import PerceptualLoss
from sphere.utils import vector_compute_magnitude, nn_concat_all_gather
from sphere.layers import vector_rms_norm, grid_rms_norm
from sphere.fd_loss import FDLoss

logger = logging.getLogger(__name__)


def l1_loss(x, y, reduction="mean", keepdim=False):
    return F.smooth_l1_loss(x, y, reduction=reduction)


def l2_loss(x, y, reduction="mean", keepdim=False):
    return F.mse_loss(x, y, reduction=reduction)


"""
classes
"""


class G1Loss(nn.Module):
    def __init__(
        self,
        latent_spherify_mode: str = "global",
        # low-angle (reconstruction phase) weights
        perceptual_loss: str = "lpips-convnext_s-1.0-0.1",
        perceptual_loss_chns_range: tuple[int, int] = None,
        perceptual_ckpt_path: str = "",
        perceptual_weight: float = 1.0,
        perceptual_image_size: int | None = None,
        perceptual_convnext_aug: str = None,
        perceptual_convnext_loss_type: str = "mse",
        perceptual_convnext_loss_reduction: str = "mean",
        distance_loss_type: str = "l2",
        distance_weight: float = 1.0,
        distance_apply_regime: str = "lo",
        # high-angle (generation phase) weights
        latent_consistency_weight: float = 0.0,
        latent_consistency_apply_regime: str = "hi",
        weight_anchor_loss: bool = True,
        weight_anchor_cutoff_params: str = "70.0-5.0-soft",
        # high-angle: score-based distribution matching
        use_score: bool = False,
        score_config: dict = None,
        score_device: torch.device = None,
        score_num_classes: int = 0,
        # high-angle: fd loss in the self.encoder space
        fd_weight: float = 0.0,
        fd_start_epoch: int = 0,
        fd_dim: int = 768,
        fd_pool: str = "mean",
        fd_apply_regime: str = "hi",
        fd_ema_decay: float = 0.999,
        fd_use_clean_ema: bool = True,
        fd_use_noisy_ema: bool = False,
        fd_norm_eps: float = 0.01,
        # high-angle: fd loss in the frozen judge space
        fd_judge_loss: nn.Module = None,
        fd_judge_weight: float = 0.0,
        fd_judge_start_epoch: int = None,
        # gradient probe: every this many steps, log the norm of each term's
        # gradient on the decoded image (0 disables)
        grad_probe_interval: int = 0,
        # basics
        ptdtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.grad_probe_interval = int(grad_probe_interval)

        spherify_fn = {"global": vector_rms_norm, "local": grid_rms_norm}[
            latent_spherify_mode
        ]
        self.f = partial(spherify_fn, zero_mean=False)

        # low-angle (< cutoff) weights
        assert distance_loss_type in ["l1", "l2", "l2+l1", "l1+l2"]
        self.distance_loss_type = distance_loss_type

        assert distance_apply_regime in ["hi", "lo", "all"]
        self.dist_apply_regime = distance_apply_regime

        self.perceptual_loss = PerceptualLoss(
            model_name=perceptual_loss,
            ckpt_path=perceptual_ckpt_path,
            perceptual_loss_chns_range=(
                (0, 5)
                if perceptual_loss_chns_range is None
                else perceptual_loss_chns_range
            ),
            image_size=perceptual_image_size,
            ptdtype=ptdtype,
            convnext_aug=perceptual_convnext_aug,
            convnext_loss_type=perceptual_convnext_loss_type,
            convnext_loss_reduction=perceptual_convnext_loss_reduction,
        )
        self.perceptual_loss.eval().requires_grad_(False)

        self.pix_dist_weight = distance_weight
        self.pix_perc_weight = perceptual_weight

        # high-angle (> cutoff) weights
        assert latent_consistency_apply_regime in ["hi", "lo", "all"]
        self.lat_con_weight = latent_consistency_weight
        self.lat_con_apply_regime = latent_consistency_apply_regime

        # score distribution matching weights and config
        self.use_score = use_score
        self.score_match = None
        if self.use_score:
            from sphere.score import build_score_matching

            assert weight_anchor_loss, "score matching needs anchor weighting"
            thres_str, band_str, mode = weight_anchor_cutoff_params.split("-")
            self.score_match = build_score_matching(
                score_config,
                anchor_params=(thres_str, band_str, mode),
                device=score_device,
                num_classes=score_num_classes,
            )

        assert fd_apply_regime in ["hi", "lo", "all"]
        self.fd_weight = fd_weight
        self.fd_start_epoch = fd_start_epoch
        self.fd_apply_regime = fd_apply_regime
        self.fd_loss = None
        self.fd_judge_loss = None
        self.fd_judge_weight = float(fd_judge_weight)
        self.fd_judge_start_epoch = (
            fd_start_epoch
            if fd_judge_start_epoch is None
            else int(fd_judge_start_epoch)
        )
        if self.fd_judge_weight > 0:
            assert fd_judge_loss is not None, "fd_judge_weight > 0 needs fd_judge_loss"
            self.fd_judge_loss = fd_judge_loss
            logger.info(
                f"FD loss (judges, weight {self.fd_judge_weight}) from epoch "
                f"{self.fd_judge_start_epoch}: {self.fd_judge_loss}"
            )
        if self.fd_weight > 0:
            self.fd_loss = FDLoss(
                dim=fd_dim,
                ema_decay=fd_ema_decay,
                use_clean_ema=fd_use_clean_ema,
                use_noisy_ema=fd_use_noisy_ema,
                pool=fd_pool,
                norm_eps=fd_norm_eps,
            )
            logger.info(f"FD loss from epoch {fd_start_epoch}: {self.fd_loss}")

        # weights for low-/high-angle losses
        self.weight_anchor_loss = weight_anchor_loss
        if self.weight_anchor_loss:
            thres_str, band_str, mode = weight_anchor_cutoff_params.split("-")
            self.anchor_weight_thres_deg = float(thres_str)
            self.anchor_weight_band_deg = float(band_str)
            self.anchor_weight_mode = mode
        else:
            self.anchor_weight_thres_deg = None
            self.anchor_weight_band_deg = None
            self.anchor_weight_mode = None

        self.log_dict = {}
        self._data_range_checked = False

    def compute_distance_loss(self, x: Tensor, y: Tensor, reduction: str = "mean"):
        if self.distance_loss_type == "l1":
            return l1_loss(x, y, reduction=reduction)
        elif self.distance_loss_type == "l2":
            return l2_loss(x, y, reduction=reduction)
        elif self.distance_loss_type in ["l2+l1", "l1+l2"]:
            a = l1_loss(x, y, reduction=reduction)
            b = l2_loss(x, y, reduction=reduction)
            return 0.5 * a + 0.5 * b
        else:
            raise NotImplementedError(
                f"unknown distance loss type: {self.distance_loss_type}"
            )

    @staticmethod
    def _probe_grad_norms(x: Tensor, terms: dict[str, Tensor]):
        """
        debug probe: gradient norm of each loss term w.r.t. the decoded image,
        useful for setting the loss weights by gradient ratio. the graph is
        retained for the real backward and no parameter grad is written

        x     : decoded image, with grad
        terms : {name: scalar loss term}
        out   : {gnorm_<name>: float}, plus gnorm_<name>_over_perc, the ratio
                to the perceptual term, for the main terms
        """
        out = {}
        if not (torch.is_tensor(x) and x.requires_grad):
            return out
        for name, term in terms.items():
            if not (torch.is_tensor(term) and term.requires_grad):
                out[f"gnorm_{name}"] = 0.0
                continue
            (g,) = torch.autograd.grad(term, x, retain_graph=True, allow_unused=True)
            out[f"gnorm_{name}"] = 0.0 if g is None else g.float().norm().item()
        ref = out.get("gnorm_perc", 0.0)
        if ref > 0:
            for name in ["dist", "fd", "fd_judge", "score", "score_fd", "score_latent"]:
                if f"gnorm_{name}" in out:
                    out[f"gnorm_{name}_over_perc"] = out[f"gnorm_{name}"] / ref
        return out

    def forward(
        self,
        input: dict[str, Tensor],
        target: Tensor,
        epoch: int = 0,
        step: int = 0,
        hidden_states: dict[str, Tensor] = None,
        latent_list: dict[str, Tensor] = None,
        eps: float = 1e-6,
        autocast_ctx=None,
        class_labels: Tensor = None,
    ):
        # ------------------------------------------------------------------
        # 0. setup: images to [0, 1], anchor weights
        # ------------------------------------------------------------------
        device = target.device

        x_NOISY = input["x_NOISY"].float() * 0.5 + 0.5
        x_tgt = target.float() * 0.5 + 0.5

        alpha_NOISY = input["alpha_NOISY"].float()  # [B]

        B = x_NOISY.shape[0]

        if self.weight_anchor_loss:
            w_lo_angle = get_anchor_weight(
                alpha_NOISY,
                thres_deg=self.anchor_weight_thres_deg,
                band_deg=self.anchor_weight_band_deg,
                mode=self.anchor_weight_mode,
            ).reshape(-1, 1, 1, 1)
        else:
            w_lo_angle = torch.ones((B, 1, 1, 1), device=device)

        w_hi_angle = (
            1.0 - w_lo_angle if self.weight_anchor_loss else torch.ones_like(w_lo_angle)
        )
        w_ones = torch.ones_like(w_lo_angle)
        w_dist = {"lo": w_lo_angle, "hi": w_hi_angle, "all": w_ones}[
            self.dist_apply_regime
        ]

        if not self._data_range_checked:
            assert torch.all((x_tgt >= 0) & (x_tgt <= 1))
            self._data_range_checked = True

        # ------------------------------------------------------------------
        # 1. pixel reconstruction (low angle): distance + perceptual
        # ------------------------------------------------------------------
        d = self.compute_distance_loss(x_NOISY, x_tgt, reduction="none").mean(
            dim=[1, 2, 3], keepdim=True
        )
        d = ddp_weighted_mean(d, w_dist, eps)
        dist_psnr = -10 * torch.log10(d.detach() + eps)
        dist_loss = self.pix_dist_weight * d

        p = self.perceptual_loss(x_NOISY, x_tgt)
        perc_parts = dict(self.perceptual_loss.log_dict)  # lpips / convnext
        p = ddp_weighted_mean(p, w_dist.view_as(p), eps)
        perc_loss = self.pix_perc_weight * p

        # ------------------------------------------------------------------
        # 2. latent consistency: encode(decode(z_NOISY)) -> z_clean
        # ------------------------------------------------------------------
        lat_con_loss = torch.zeros((), device=device)

        if self.lat_con_weight > 0:
            v_inp = latent_list["v_NOISY"].float()  # [B, D]
            v_tgt = latent_list["z_clean"].float()  # [B, D]

            _a = 1.0 - F.cosine_similarity(
                v_inp.flatten(start_dim=1),
                v_tgt.flatten(start_dim=1).detach(),
                dim=-1,
            )

            if self.lat_con_apply_regime == "lo":
                _a = ddp_weighted_mean(_a, w_lo_angle.view_as(_a), eps)

            elif self.lat_con_apply_regime == "hi":
                _a = ddp_weighted_mean(_a, w_hi_angle.view_as(_a), eps)

            elif self.lat_con_apply_regime == "all":
                _a = ddp_weighted_mean(_a, w_ones.view_as(_a), eps)

            lat_con_loss = self.lat_con_weight * _a

        # ------------------------------------------------------------------
        # 3. representation frechet distance (high angle)
        # ------------------------------------------------------------------
        # fd_judge_loss : decode(z_NOISY) through the frozen fdr-6 judges against
        #                 the fixed reference statistics
        # fd_loss       : moments of encoder(decode(z_NOISY)) matched to
        #                 encoder(x), the reference tracked online
        # the two are independent and summed; the generated side is weighted
        # per sample by the anchor weight
        fd_loss = torch.zeros((), device=device)
        fd_judge_loss = torch.zeros((), device=device)
        w_fd = {"lo": w_lo_angle, "hi": w_hi_angle, "all": w_ones}[
            self.fd_apply_regime
        ].reshape(-1)

        if self.fd_judge_loss is not None:
            x_gen = input["x_NOISY"].float()  # [B, 3, H, W] in [-1, 1]
            if epoch >= self.fd_judge_start_epoch and self.fd_judge_loss.ready:
                _fd = self.fd_judge_loss(x_gen, w_fd, autocast_ctx=autocast_ctx)
                fd_judge_loss = self.fd_judge_weight * _fd
            else:
                self.fd_judge_loss.fill(x_gen.detach(), w_fd, autocast_ctx=autocast_ctx)

        if self.fd_loss is not None:
            assert "v_NOISY" in latent_list, "FD loss needs the re-encoded latent"
            if self.fd_loss.pool == "cls":
                assert "cls_NOISY" in latent_list, (
                    "fd_pool='cls' needs the encoder's CLS token: use a "
                    "pretrained DINO encoder (siglip2 / sphere encoders have none)"
                )
                z_clean = latent_list["cls_clean"].float()  # [B, D]
                v_NOISY = latent_list["cls_NOISY"].float()  # [B, D]
            else:
                z_clean = latent_list["z_clean"].float()  # [B, N, D]
                v_NOISY = latent_list["v_NOISY"].float()  # [B, N, D]
            if epoch >= self.fd_start_epoch:
                _fd = self.fd_loss(z_clean.detach(), v_NOISY, w_noisy=w_fd)
                fd_loss = self.fd_weight * _fd
            else:
                # warm up both sides while the loss is gated off, so it opens
                # on settled statistics rather than on one batch
                self.fd_loss.update_reference(z_clean.detach())
                self.fd_loss.update_generated(v_NOISY.detach(), w_fd)

        # ------------------------------------------------------------------
        # 4. score distribution matching (high angle)
        # ------------------------------------------------------------------
        score_loss = torch.zeros((), device=device)
        score_logits_loss = torch.zeros((), device=device)
        score_latent_loss = torch.zeros((), device=device)
        score_fd_loss = torch.zeros((), device=device)

        if self.score_match is not None and epoch >= self.score_match.start_epoch:
            self.score_match._epoch = epoch

            with autocast_ctx or nullcontext():
                real, real_w, fake, fake_w, score_conditions = (
                    self.score_match.prepare_feats(input, target, class_labels)
                )
                if self.score_match.gen_warmup_scale(epoch) > 0:
                    score_loss = self.score_match.generator_loss(
                        fake, fake_w, score_conditions["fake"]
                    )

                # for using score feature extractor to compute semantic alignment loss
                if (
                    self.score_match.logits_weight > 0
                    and epoch >= self.score_match.logits_start_epoch
                ):
                    score_logits_loss = self.score_match.logits_loss(fake_w)

                # for using score feature extractor to compute latent consistency loss
                if (
                    self.score_match.latent_weight > 0
                    and epoch >= self.score_match.latent_start_epoch
                ):
                    score_latent_loss = self.score_match.latent_loss(fake_w)

                # for using score feature extractor to compute fd loss
                if self.score_match.fd_weight > 0:
                    score_fd_loss = self.score_match.frechet_loss(fake_w)

            # stash detached features for the score-net DSM update in train.py
            self.score_match._cache = (
                [r.detach() for r in real],
                real_w.detach(),
                [f.detach() for f in fake],
                fake_w.detach(),
                score_conditions,
            )

        # ------------------------------------------------------------------
        # 5. gradient probe (log steps only)
        # ------------------------------------------------------------------
        gnorm = {}
        if self.grad_probe_interval > 0 and (
            step == 0 or (step + 1) % self.grad_probe_interval == 0
        ):
            gnorm = self._probe_grad_norms(
                input["x_NOISY"],
                {
                    "dist": dist_loss,
                    "perc": perc_loss,
                    "lat_con": lat_con_loss,
                    "fd": fd_loss,
                    "fd_judge": fd_judge_loss,
                    "score": score_loss,
                    "score_logits": score_logits_loss,
                    "score_latent": score_latent_loss,
                    "score_fd": score_fd_loss,
                },
            )

        # ------------------------------------------------------------------
        # 6. total + logging
        # ------------------------------------------------------------------
        total_loss = (
            dist_loss
            + perc_loss
            + lat_con_loss
            + fd_loss
            + fd_judge_loss
            + score_loss
            + score_logits_loss
            + score_latent_loss
            + score_fd_loss
        )

        self.log_dict.update(
            {
                "total_loss": total_loss.clone().detach(),
                "dist_loss": dist_loss.clone().detach(),
                "perc_loss": perc_loss.clone().detach(),
                "dist_psnr": dist_psnr.detach(),
            }
        )

        # raw (unweighted) perceptual components, for monitoring only
        for k, v in perc_parts.items():
            self.log_dict[f"perc_{k}"] = v

        if self.lat_con_weight > 0:
            self.log_dict["lat_con_loss"] = lat_con_loss.clone().detach()

        # judges first, encoder second: the encoder-space keys (fd_raw,
        # fd_n_eff) win on collision; the judge term keeps its per-judge
        # fd_{judge} / fdr_{judge} keys and gets its own fd_judge_* copies
        if self.fd_judge_loss is not None:
            if epoch >= self.fd_judge_start_epoch:
                self.log_dict["fd_judge_loss"] = fd_judge_loss.clone().detach()
            for k, v in self.fd_judge_loss.log_dict.items():
                self.log_dict[k] = v
                if k in ("fd_n_eff", "fd_history"):
                    self.log_dict[f"fd_judge_{k[3:]}"] = v
        if self.fd_loss is not None:
            if epoch >= self.fd_start_epoch:
                self.log_dict["fd_loss"] = fd_loss.clone().detach()
            for k, v in self.fd_loss.log_dict.items():
                self.log_dict[k] = v

        if self.score_match is not None:
            self.log_dict.update(self.score_match.log_dict)
            self.log_dict["score_loss"] = score_loss.clone().detach()
            if self.score_match.logits_weight > 0:
                self.log_dict["score_logits_loss"] = score_logits_loss.clone().detach()
            if self.score_match.latent_weight > 0:
                self.log_dict["score_latent_loss"] = score_latent_loss.clone().detach()
            if self.score_match.fd_weight > 0:
                self.log_dict["score_fd_loss"] = score_fd_loss.clone().detach()

        self.log_dict.update(gnorm)
        return total_loss


"""
functions
"""


def get_anchor_weight(alpha_rad, thres_deg=80.0, band_deg=5.0, mode="soft"):
    """
    per-sample weight of the low-angle (reconstruction) regime: ~1 for small
    angles, ~0 for large angles (generation regime)

    alpha_rad : noise angles in radians
    thres_deg : cutoff angle in degrees
    band_deg  : width of the transition band around the cutoff, soft mode only
    mode      : soft (sigmoid) | hard (step at thres_deg)
    out       : weights in [0, 1], same shape as alpha_rad
    """
    alpha_deg = torch.rad2deg(alpha_rad)
    if mode == "soft":
        return torch.sigmoid(-(alpha_deg - thres_deg) / (band_deg * 0.25))
    elif mode == "hard":
        return (alpha_deg <= thres_deg).to(alpha_rad.dtype)
    else:
        raise ValueError(f"unknown mode: {mode}")


def ddp_weighted_mean(values: Tensor, weights: Tensor, eps: float = 1e-6):
    """
    weighted mean over the global batch, sum(w * v) / sum(w) across all ranks,
    rather than the per-rank weighted mean: a rank whose local weights sum to
    ~0 would otherwise give a biased estimate of the global loss.

    the returned scalar has the right value and the right gradient under ddp:

        value : the global mean, the same on every rank
        grad  : that of world_size * (local sum(w * v)) / (global sum(w)),
                which ddp's gradient averaging turns into the gradient of the
                global mean

    without ddp it reduces to the plain weighted mean

    values  : per-sample values, any shape
    weights : per-sample weights, same shape as values
    eps     : added to the denominator
    out     : scalar
    """
    num = (values * weights).sum()
    den = weights.sum()
    if dist.is_initialized() and dist.get_world_size() > 1:
        # one all-reduce for both sums; detached -> no autograd interaction with
        # DDP's own gradient bucketing (safe under static_graph).
        stats = torch.stack([num.detach(), den.detach()]).clone()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        num_g, den_g = stats[0], stats[1]
        m = num_g / (den_g + eps)  # true global mean (value, detached)
        s = dist.get_world_size() * num / (den_g + eps)  # gradient carrier
        return m + (s - s.detach())
    return num / (den + eps)
