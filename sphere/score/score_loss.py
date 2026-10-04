"""
Score-based distribution matching for the high-angle (generation) regime.

The idea:

    * `score_real` : score of the REAL distribution. Trained on clean (0 degree)
                     features (anchored to data through the reconstruction loss).
    * `score_fake` : score of the GENERATOR's current distribution. Trained
                     online on the high-angle features.
    * `score_nets` : one 4-layer transformer EDM denoiser per feature layer,
                     trained on real and fake features.

Generator gradient on a generated feature x_t:
    grad = w(sigma) * (D_fake(x_t) - D_real(x_t))
"""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sphere.fd_loss import FDLoss
from sphere.utils import concat_all_gather

from .score_net import ScoreNet

logger = logging.getLogger(__name__)


"""
functions
"""


def _ddp_weighted_mean(weighted_sum, weight_sum, eps=1e-8):
    if dist.is_initialized() and dist.get_world_size() > 1:
        stats = torch.stack([weighted_sum.detach(), weight_sum.detach()]).clone()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        num_g, den_g = stats[0], stats[1]
        m = num_g / (den_g + eps)  # true global mean (value)
        s = dist.get_world_size() * weighted_sum / (den_g + eps)  # grad carrier
        return m + (s - s.detach())
    return weighted_sum / (weight_sum + eps)


def _ddp_weighted_mean_all(pairs, extra=None, eps=1e-8):
    """DDP-safe weighted mean of a GROUP of (weighted_sum, weight_sum) pairs,
    taking ONE all_reduce for the whole group instead of one per pair. The
    returned means carry gradient into their `weighted_sum`.

    Returns (means, extra_mean); extra_mean is None when `extra` is.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return [n / (d + eps) for n, d in pairs], extra

    # .float(): the pairs come from different layers (and both critics), so
    # this stack has to survive a dtype mismatch between them, not just within
    # one pair.
    stats = [t.detach().float() for pair in pairs for t in pair]
    n_pair = len(stats)
    if extra is not None:
        stats = torch.cat([torch.stack(stats), extra.detach().float().reshape(-1)])
    else:
        stats = torch.stack(stats)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    ws = dist.get_world_size()

    out = []
    for i, (num, _) in enumerate(pairs):
        num_g, den_g = stats[2 * i], stats[2 * i + 1]
        m = num_g / (den_g + eps)  # true global mean (value)
        s = ws * num / (den_g + eps)  # grad carrier
        out.append(m + (s - s.detach()))
    extra_mean = None if extra is None else (stats[n_pair:] / ws).reshape(extra.shape)
    return out, extra_mean


def _sync_scale_mirrors(module, incompatible_keys):
    module._seen = int(module._scale_cnt.item())
    module._frozen = 0 < module.scale_calib_steps <= module._seen
    module._frozen_logs = None
    module._tb_cnt = int(module._transport_seen.item())


def _drop_legacy_keys(module, state_dict, prefix, *args):
    state_dict.pop(prefix + "_dsm_step", None)
    for k in [k for k in state_dict if k.startswith(prefix + "lat_con_mean")]:
        state_dict.pop(k)


def _gather(t):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    return concat_all_gather(t)


"""
classes
"""


class ScoreMatchingLoss(nn.Module):

    def __init__(
        self,
        # args for score nets
        space: str = "convnext",
        width: int = 512,
        width_margin: int = 0,
        use_qk_norm: bool = True,
        depth: int = 4,
        real_depth: int = None,
        fake_depth: int = None,
        num_heads: int = 8,
        drop_residual_path_prob: float = 0.0,
        # args for score loss
        sigma_data: float = 0.5,
        sdpa_mode: str = "manual",
        sigma_cond: bool = True,
        class_cond: bool = False,
        num_classes: int = 0,
        adaln_single: bool = False,
        sigma_p_mean: float = -1.2,
        sigma_p_std: float = 1.2,
        sigma_min: float = 2e-3,
        sigma_max: float = 8.0,
        gen_sigma_p_mean: float = None,
        gen_sigma_p_std: float = None,
        gen_sigma_max: float = None,
        weight: float = 1.0,
        start_epoch: int = 0,
        real_start_epoch: int = None,
        fake_start_epoch: int = None,
        gen_start_epoch: int = None,
        gen_warmup_start_epoch: int = None,
        gen_warmup_shape: str = "linear",
        anchor_thres_deg: float = 70.0,
        anchor_band_deg: float = 5.0,
        anchor_mode: str = "soft",
        dino_ckpt_path: str = None,
        dino_recipe: str = "S_8",
        dino_layers: list = None,
        dino_img_size: int = 224,
        dino_device: torch.device = None,
        convnext_model: str = "timm/convnextv2_nano.fcmae_ft_in1k",
        convnext_stages: list = None,
        convnext_img_size: int = 256,
        standardize: bool = True,
        standardize_per_channel: bool = False,
        standardize_channel_floor: float = 0.1,
        apply_regime: str = "hi",
        scale_ema_decay: float = 0.99,
        scale_calib_steps: int = None,
        normalizer_reduction: str = "sample",
        normalizer_floor: float = 1e-2,
        grad_clip: float = 0.0,
        transport_balance: bool = False,
        transport_balance_target: float = None,
        transport_balance_decay: float = 0.99,
        transport_balance_calib_steps: int = 200,
        logits_weight: float = 0.0,
        logits_loss_type: str = "mse",
        logits_start_epoch: int = None,
        latent_weight: float = 0.0,
        latent_loss_type: str = "cosine",
        latent_level: str = "pooled",
        latent_start_epoch: int = None,
        fd_weight: float = 0.0,
        fd_start_epoch: int = None,
        fd_pool: str = "mean",
        fd_ema_decay: float = 0.999,
        fd_use_clean_ema: bool = True,
        fd_use_noisy_ema: bool = False,
        fd_norm_eps: float = 0.01,
    ):
        super().__init__()
        assert space in ["dino", "convnext"]
        assert normalizer_reduction in ["sample", "batch"]
        assert apply_regime in ["all", "lo", "hi"]
        assert logits_loss_type in ["mse", "cosine"]
        assert latent_loss_type in ["mse", "cosine"]
        assert latent_level in ["pooled", "tokens"]
        assert fd_pool in ["mean", "token"]
        self.space = space
        self.weight = weight

        self.logits_weight = float(logits_weight)
        self.logits_loss_type = logits_loss_type
        assert self.logits_weight <= 0 or space == "convnext", (
            f"logits_weight={self.logits_weight} needs a classifier head, which "
            f"only the convnext space has (got space={space!r})"
        )

        self.latent_weight = float(latent_weight)
        self.latent_loss_type = latent_loss_type
        self.latent_level = latent_level

        # representation FD on the featurizer's final-stage tokens: the moments
        # of the generated features are matched to the clean ones (see
        # sphere/fd_loss.py). Reads the same forward as latent_loss.
        self.fd_weight = float(fd_weight)
        self.fd_pool = fd_pool

        self.class_cond = bool(class_cond)
        self.num_classes = int(num_classes)
        assert not self.class_cond or self.num_classes > 0, (
            f"class-conditioned score loss needs num_classes > 0, got "
            f"{self.num_classes}"
        )

        self.start_epoch = start_epoch
        self.gen_start_epoch = (
            start_epoch if gen_start_epoch is None else gen_start_epoch
        )
        assert self.gen_start_epoch >= self.start_epoch, (
            f"gen_start_epoch ({self.gen_start_epoch}) "
            f"must be >= "
            f"start_epoch ({self.start_epoch})"
        )

        # warmup for transport loss
        if gen_warmup_start_epoch is None:
            gen_warmup_start_epoch = self.gen_start_epoch - 10  # default
        self.gen_warmup_start_epoch = max(int(gen_warmup_start_epoch), self.start_epoch)
        assert self.gen_warmup_start_epoch <= self.gen_start_epoch, (
            f"gen_warmup_start_epoch ({self.gen_warmup_start_epoch}) "
            f"must be <= "
            f"gen_start_epoch ({self.gen_start_epoch})"
        )
        assert gen_warmup_shape in ["linear", "cosine"]
        self.gen_warmup_shape = gen_warmup_shape

        self.real_start_epoch = (
            start_epoch if real_start_epoch is None else int(real_start_epoch)
        )
        self.fake_start_epoch = (
            start_epoch if fake_start_epoch is None else int(fake_start_epoch)
        )
        self.logits_start_epoch = (
            self.start_epoch if logits_start_epoch is None else int(logits_start_epoch)
        )
        assert self.logits_start_epoch >= self.start_epoch, (
            f"logits_start_epoch ({self.logits_start_epoch}) must be >= "
            f"start_epoch ({self.start_epoch}); nothing is featurized before it"
        )
        self.latent_start_epoch = (
            self.start_epoch if latent_start_epoch is None else int(latent_start_epoch)
        )
        self.fd_start_epoch = (
            self.start_epoch if fd_start_epoch is None else int(fd_start_epoch)
        )
        assert self.fd_start_epoch >= self.start_epoch, (
            f"fd_start_epoch ({self.fd_start_epoch}) must be >= "
            f"start_epoch ({self.start_epoch}): the clean statistics are "
            "warmed up from the featurizer forward, which only runs from there"
        )
        assert self.latent_start_epoch >= self.start_epoch, (
            f"latent_start_epoch ({self.latent_start_epoch}) must be >= "
            f"start_epoch ({self.start_epoch}); nothing is featurized before it"
        )

        _starts = [
            ("real_start_epoch", self.real_start_epoch),
            ("fake_start_epoch", self.fake_start_epoch),
        ]
        for _n, _v in _starts:
            assert _v >= self.start_epoch, (
                f"{_n} ({_v}) must be >= start_epoch ({self.start_epoch}); "
                "nothing is cached for the DSM step before start_epoch"
            )
            assert _v <= self.gen_warmup_start_epoch, (
                f"{_n} ({_v}) must be <= gen_warmup_start_epoch "
                f"({self.gen_warmup_start_epoch}): the transport reads "
                f"both denoisers as soon as the generator surrogate fades in"
            )

        self.standardize = standardize
        self.standardize_per_channel = standardize_per_channel
        self.standardize_channel_floor = standardize_channel_floor
        self.scale_ema_decay = scale_ema_decay

        self.scale_calib_steps = (
            1000 if scale_calib_steps is None else int(scale_calib_steps)
        )

        # sigma distribution for training and transport (gen_*)
        self.sigma_data = sigma_data
        self.sigma_p_mean = sigma_p_mean
        self.sigma_p_std = sigma_p_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.gen_sigma_p_mean = (
            sigma_p_mean if gen_sigma_p_mean is None else gen_sigma_p_mean
        )
        self.gen_sigma_p_std = (
            sigma_p_std if gen_sigma_p_std is None else gen_sigma_p_std
        )
        self.gen_sigma_max = sigma_max if gen_sigma_max is None else gen_sigma_max

        # angle cutoff for lo and hi regimes
        self.anchor_thres_deg = anchor_thres_deg
        self.anchor_band_deg = anchor_band_deg
        self.anchor_mode = anchor_mode
        self.apply_regime = apply_regime

        self.normalizer_reduction = normalizer_reduction
        self.normalizer_floor = float(normalizer_floor)

        self._epoch = 0  # current epoch for training
        self.grad_clip = grad_clip  # transport side
        # pixel-space featurizer (dino or convnext)
        self.featurizer = None
        self.dino = None
        _return_final = self.latent_weight > 0 or self.fd_weight > 0
        if space == "dino":
            from .dino_feat import DinoV1NoDrop

            assert dino_ckpt_path is not None, "score_space='dino' needs dino_ckpt_path"

            _score_req = [2, 5, 8, 11] if dino_layers is None else list(dino_layers)

            self.dino = DinoV1NoDrop(
                dino_ckpt_path=dino_ckpt_path,
                recipe=dino_recipe,
                layers=sorted({int(v) for v in _score_req}),
                img_size=dino_img_size,
                device=dino_device if dino_device is not None else "cpu",
                return_final=_return_final,
            )
            self.featurizer = self.dino
            _n_extract = len(self.dino.layers)
            _all_dims = [self.dino.embed_dim] * _n_extract
            _all_tokens = [self.dino.num_tokens] * _n_extract
            _score_ids = [i for i in self.dino.layers if i in set(_score_req)]
            assert _score_ids, (
                f"dino_layers={_score_req} resolved to no usable layer for "
                f"recipe={dino_recipe!r}"
            )

        else:  # convnext
            from .convnext_feat import build_convnext_featurizer

            _score_req = [1, 3] if convnext_stages is None else list(convnext_stages)

            self.featurizer = build_convnext_featurizer(
                model_name=convnext_model,
                stages=sorted({int(v) for v in _score_req}),
                img_size=convnext_img_size,
                device=dino_device if dino_device is not None else "cpu",
                return_logits=self.logits_weight > 0,
                return_final=_return_final,
            )
            _all_dims = list(self.featurizer.embed_dims)
            _all_tokens = list(self.featurizer.num_tokens_list)
            _score_ids = [i for i in self.featurizer.layers if i in set(_score_req)]
            assert _score_ids, f"convnext_stages={_score_req} resolved to nothing"

        # The featurizer hands back one entry per extracted layer, ascending,
        # and the scored layers are the only ones extracted -- so everything
        # sized by num_layers (score nets, standardization buffers,
        # normalizers) indexes that list directly.
        _layers = list(self.featurizer.layers)
        self._score_pos = [_layers.index(i) for i in _score_ids]

        self.num_layers = len(self._score_pos)
        self.feat_dims = [_all_dims[p] for p in self._score_pos]
        self.num_tokens_list = [_all_tokens[p] for p in self._score_pos]

        self.score_layer_ids = _score_ids
        assert len(self.score_layer_ids) == self.num_layers
        self.latent_dim = getattr(self.featurizer, "final_dim", None)

        self.fd_loss = None
        if self.fd_weight > 0:
            assert self.latent_dim is not None, (
                f"fd_weight={self.fd_weight} needs the featurizer's final "
                f"feature (return_final), which space={space!r} did not expose"
            )
            self.fd_loss = FDLoss(
                dim=self.latent_dim,
                ema_decay=fd_ema_decay,
                use_clean_ema=fd_use_clean_ema,
                use_noisy_ema=fd_use_noisy_ema,
                pool=fd_pool,
                norm_eps=fd_norm_eps,
            )

        # per-layer running stats for standardization
        self.register_buffer("feat_std", torch.ones(self.num_layers))
        self._mean_uniform = len(set(self.feat_dims)) == 1
        if self._mean_uniform:
            self.register_buffer(
                "feat_mean", torch.zeros(self.num_layers, self.feat_dims[0])
            )
            self.register_buffer(
                "feat_m2", torch.ones(self.num_layers, self.feat_dims[0])
            )
        else:
            for i, d in enumerate(self.feat_dims):
                self.register_buffer(f"feat_mean_{i}", torch.zeros(d))
                self.register_buffer(f"feat_m2_{i}", torch.ones(d))

        # per-channel std buffers, same layout scheme as the means
        if standardize_per_channel:
            if self._mean_uniform:
                self.register_buffer(
                    "feat_ch_std", torch.ones(self.num_layers, self.feat_dims[0])
                )
            else:
                for i, d in enumerate(self.feat_dims):
                    self.register_buffer(f"feat_ch_std_{i}", torch.ones(d))

        self.register_buffer("_scale_cnt", torch.zeros((), dtype=torch.long))
        self._cnt = 0
        self._frozen = False
        self._frozen_logs = None  # metrics cached at freeze time (see _scale_logs)
        self.register_load_state_dict_post_hook(_sync_scale_mirrors)
        self.register_load_state_dict_pre_hook(_drop_legacy_keys)

        self.sigma_cond = sigma_cond
        self.adaln_single = bool(adaln_single)
        self.drop_residual_path_prob = drop_residual_path_prob

        self.real_depth = int(depth if real_depth is None else real_depth)
        self.fake_depth = int(depth if fake_depth is None else fake_depth)

        def _make_net(ld, nt, net_depth):
            return ScoreNet(
                latent_dim=ld,
                num_tokens=nt,
                width=width,
                width_margin=width_margin,
                use_qk_norm=use_qk_norm,
                depth=net_depth,
                num_heads=num_heads,
                drop_residual_path_prob=drop_residual_path_prob,
                sigma_data=sigma_data,
                sigma_cond=sigma_cond,
                class_cond=self.class_cond,
                num_classes=self.num_classes,
                sdpa_mode=sdpa_mode,
                adaln_single=adaln_single,
            )

        # per-layer score nets, one for each feature layer
        self.real_score_nets = nn.ModuleList(
            _make_net(self.feat_dims[i], self.num_tokens_list[i], self.real_depth)
            for i in range(self.num_layers)
        )
        self.fake_score_nets = nn.ModuleList(
            _make_net(self.feat_dims[i], self.num_tokens_list[i], self.fake_depth)
            for i in range(self.num_layers)
        )

        # per-layer widths, such that score model width >= feature dim
        self.score_widths = [n.width for n in self.real_score_nets]
        if any(w != width for w in self.score_widths):
            logger.info(
                f"score net width raised above the requested {width} "
                f"(+{width_margin} margin) to fit the feature dims: "
                f"{dict(zip(self.score_layer_ids, self.score_widths))}"
            )

        # transport balance: scale the per-layer gradients
        self.transport_balance = bool(transport_balance)
        self.transport_balance_decay = float(transport_balance_decay)
        self.transport_balance_calib_steps = max(1, int(transport_balance_calib_steps))
        self._tb_target_cfg = transport_balance_target
        self._transport_dof = [
            float(self.num_tokens_list[i] * self.feat_dims[i]) ** 0.5
            for i in range(self.num_layers)
        ]  # = sqrt(N_j * D_j) per scored layer

        self.register_buffer("transport_rms_ema", torch.ones(self.num_layers))
        self.register_buffer("transport_target", torch.zeros(()))
        self.register_buffer("_transport_seen", torch.zeros((), dtype=torch.long))
        self._tb_cnt = 0  # host mirror, re-derived by _sync_scale_mirrors

        self.pixel_augs = None
        self.pixel_augs_symmetric = False

        # (real, real_w, fake, fake_w, conditions) detached, for the DSM step
        self._cache = None
        self._logits = None
        self._latent = None
        self._fd = None
        self.log_dict = {}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"space={self.space}, "
            f"depth=(real={self.real_depth}, fake={self.fake_depth}), "
            f"weight={self.weight}, "
            f"start_epoch={self.start_epoch}, "
            f"real_start_epoch={self.real_start_epoch}, "
            f"fake_start_epoch={self.fake_start_epoch}, "
            f"gen_start_epoch={self.gen_start_epoch}, "
            f"gen_warmup=({self.gen_warmup_start_epoch}->{self.gen_start_epoch}, "
            f"shape={self.gen_warmup_shape}), "
            f"apply_regime={self.apply_regime}, "
            f"anchor_thres_deg={self.anchor_thres_deg}, "
            f"scored={self.score_layer_ids}, "
            f"dino_layers={self.dino.layers if self.dino is not None else None}, "
            f"backbone={getattr(self.featurizer, 'model_name', None)}"
            f":{self.featurizer.layers if self.space != 'dino' else None}, "
            f"feat_dims={self.feat_dims}, "
            f"num_tokens={self.num_tokens_list}, "
            f"score_widths={self.score_widths}, "
            f"standardize={self.standardize}"
            f"(per_channel={self.standardize_per_channel}, "
            f"floor={self.standardize_channel_floor}, "
            f"calib_steps={self.scale_calib_steps}, "
            f"ema_decay={self.scale_ema_decay}), "
            f"sigma_data={self.sigma_data}, "
            f"sigma_cond={self.sigma_cond}, "
            f"class_cond={self.class_cond}(num_classes={self.num_classes}), "
            f"adaln_single={self.adaln_single}, "
            f"gen_sigma=(p_mean={self.gen_sigma_p_mean}, p_std={self.gen_sigma_p_std}, max={self.gen_sigma_max}), "
            f"normalizer_reduction={self.normalizer_reduction}, "
            f"normalizer_floor={self.normalizer_floor}, "
            f"grad_clip={self.grad_clip}, "
            f"transport_balance={self.transport_balance}"
            f"(target={self._tb_target_cfg}, "
            f"decay={self.transport_balance_decay}, "
            f"calib_steps={self.transport_balance_calib_steps}), "
            f"logits=(weight={self.logits_weight}, "
            f"type={self.logits_loss_type}, "
            f"start_epoch={self.logits_start_epoch}), "
            f"latent=(weight={self.latent_weight}, "
            f"type={self.latent_loss_type}, "
            f"level={self.latent_level}, "
            f"dim={self.latent_dim}, "
            f"start_epoch={self.latent_start_epoch}), "
            f"fd=(weight={self.fd_weight}, "
            f"start_epoch={self.fd_start_epoch}, "
            f"module={self.fd_loss}))"
        )

    """
    helpers
    """

    @torch.no_grad()
    def _anchor_weight(self, alpha_rad):
        """
        copy the get_anchor_weight function from sphere.loss
        """
        alpha_deg = torch.rad2deg(alpha_rad.float())
        t, b = self.anchor_thres_deg, self.anchor_band_deg
        if self.anchor_mode == "soft":
            w_lo = torch.sigmoid(-(alpha_deg - t) / (b * 0.25))
        elif self.anchor_mode == "hard":
            w_lo = (alpha_deg <= t).float()
        else:
            raise ValueError(f"unknown anchor mode: {self.anchor_mode}")
        return w_lo.reshape(-1), (1.0 - w_lo).reshape(-1)

    def _mean(self, i):
        if self._mean_uniform:
            return self.feat_mean[i]
        return getattr(self, f"feat_mean_{i}")

    def _m2(self, i):
        if self._mean_uniform:
            return self.feat_m2[i]
        return getattr(self, f"feat_m2_{i}")

    def _ch_std(self, i):
        if self._mean_uniform:
            return self.feat_ch_std[i]
        return getattr(self, f"feat_ch_std_{i}")

    @torch.no_grad()
    def _scale_logs(self):
        return {"score_feat_std": self.feat_std.mean()}

    @torch.no_grad()
    def _update_scale(self, real_list):
        """
        update the standardization affine stats from the REAL features
        """
        if self._frozen:
            if self._frozen_logs is None:  # first call after a resume
                self._frozen_logs = self._scale_logs()
            self.log_dict.update(self._frozen_logs)
            return

        # also catches a layer-count mismatch (the lists compare unequal)
        assert [r.shape[-1] for r in real_list] == self.feat_dims, (
            f"feature dims {[r.shape[-1] for r in real_list]} != {self.feat_dims}; "
            "the packed reduction below slices by self.feat_dims"
        )

        # one all-reduce for both moments; the per-channel variance below is then
        # formed from the GLOBAL mean, not a per-rank one.
        packed = torch.cat(
            [r.float().mean(dim=(0, 1)) for r in real_list]
            + [r.float().pow(2).mean(dim=(0, 1)) for r in real_list]
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed /= dist.get_world_size()

        chunks, offset = [], 0
        for d in self.feat_dims + self.feat_dims:
            chunks.append(packed[offset : offset + d])
            offset += d
        b_mu, b_m2 = chunks[: self.num_layers], chunks[self.num_layers :]

        # a single non-finite batch statistic would otherwise be folded into
        # feat_mean/feat_m2 PERMANENTLY -- these are buffers, not weights, so
        # the trainer's skip-the-step guard does not protect them and every
        # later forward inherits the nan with no way back. Drop the batch.
        if not torch.isfinite(packed).all():
            logger.warning(
                "non-finite feature moments; skipping this scale update "
                "(buffers left untouched)"
            )
            self.log_dict.update(self._scale_logs())
            return

        # start with w = 1 for the first update
        w = 1.0 / (self._cnt + 1)
        calibrating = self._cnt < self.scale_calib_steps
        if not calibrating:
            w = max(w, 1.0 - self.scale_ema_decay)

        for i in range(self.num_layers):
            self._mean(i).add_(b_mu[i] - self._mean(i), alpha=w)
            self._m2(i).add_(b_m2[i] - self._m2(i), alpha=w)
        self._scale_cnt += 1
        self._cnt += 1

        for i in range(self.num_layers):
            var = (self._m2(i) - self._mean(i).pow(2)).clamp_min(1e-12)  # [D]
            std = var.mean().sqrt().clamp_min(1e-6)  # scalar, layer i
            self.feat_std[i] = std
            if self.standardize_per_channel:
                ch = var.sqrt()
                ref = ch.median().clamp_min(1e-6)
                self._ch_std(i).copy_(
                    ch.clamp_min(self.standardize_channel_floor * ref)
                )

        logs = self._scale_logs()
        self.log_dict.update(logs)

        if calibrating and self._cnt >= self.scale_calib_steps:
            self._frozen = True
            self._frozen_logs = logs
            logger.info(
                f"score feature scale frozen after {self._cnt} steps: "
                f"std={[round(v, 4) for v in self.feat_std.tolist()]}"
            )

    @torch.no_grad()
    def _regime_weight(self, regime, w_lo, w_hi, n_fake, device):
        """Per-sample weight over the fake batch for one angle regime. `n_fake`
        is passed in rather than read off fake.shape[0], so the weight length is
        tied to the angle vector that defines it."""
        return {
            "hi": w_hi,
            "lo": w_lo,
            "all": torch.ones(n_fake, device=device),
        }[regime]

    def prepare_feats(self, pixels, target, class_labels=None):
        """
        Run the frozen featurizer on the clean target and on the generated
        images, and return (real, real_w, fake, fake_w, conditions):
          real  : list of [B, N, D] reference (clean) features, detached
          fake  : list of [B, N, D] generated features WITH grad (decoder
                  graph)
          *_w   : [B] per-sample anchor weights (DDP-safe, no masking),
                  shared across layers. apply_regime acts through these weights
                  only -- no row is ever dropped, so the batch shape is the same
                  on every rank and every step.
        """

        alpha_NOISY = pixels["alpha_NOISY"]

        B = alpha_NOISY.shape[0]
        alpha = alpha_NOISY
        n_fake = alpha.shape[0]
        w_lo, w_hi = self._anchor_weight(alpha)

        if self.class_cond:
            assert (
                class_labels is not None
            ), "class-conditioned score loss needs the training labels"
            class_labels = class_labels.reshape(-1).to(
                device=alpha.device, dtype=torch.long
            )
            assert (
                class_labels.shape[0] == B
            ), f"label batch {class_labels.shape[0]} != image batch {B}"

        conditions = {
            "real": {
                "class_labels": class_labels if self.class_cond else None,
            },
            "fake": {
                "class_labels": class_labels if self.class_cond else None,
            },
        }
        assert target is not None, f"score_space={self.space!r} needs the clean target"
        tgt_px = target.float()
        if self.pixel_augs is not None and not self.pixel_augs_symmetric:
            _nonflip = [
                n for n, _, p in self.pixel_augs.pipeline if p > 0 and n != "flip"
            ]
            assert not _nonflip, (
                f"fake-only score aug {self.pixel_augs} leaks: {_nonflip} "
                "must see BOTH sides -- set score.augment.symmetric=True "
                "(or leave flip as the only active module)"
            )
        if self.pixel_augs is not None and self.pixel_augs_symmetric:
            tgt_px = self.pixel_augs.aug(tgt_px)
        real, tgt_aux = self._featurize(tgt_px)
        real = [real[p].float().detach() for p in self._score_pos]

        fk_px = pixels["x_NOISY"].float()
        if self.pixel_augs is not None:
            fk_px = self.pixel_augs.aug(fk_px)
        fake, fk_aux = self._featurize(fk_px)
        fake = [fake[p].float() for p in self._score_pos]

        # pair every fake row with its own clean target
        self._logits = None
        if "logits" in tgt_aux:
            self._logits = (tgt_aux["logits"].detach(), fk_aux["logits"])

        # latent consistency pair (target detached, input with grad) -- see
        # latent_loss.
        self._latent = None
        _key = "final_pooled" if self.latent_level == "pooled" else "final_tokens"
        if _key in tgt_aux:
            self._latent = (tgt_aux[_key].detach(), fk_aux[_key])

        # representation FD pair: clean final tokens (detached, B rows) vs the
        # generated ones (with grad) -- see frechet_loss. Raw features, never
        # standardized, so the loss lives in the space FDr-6 scores.
        self._fd = None
        if self.fd_loss is not None:
            assert (
                "final_tokens" in fk_aux
            ), "fd_weight > 0 needs the featurizer's final tokens (return_final)"
            self._fd = (tgt_aux["final_tokens"].detach(), fk_aux["final_tokens"])

        real_w = torch.ones(real[0].shape[0], device=real[0].device)
        assert fake[0].shape[0] == n_fake, (
            f"fake batch {fake[0].shape[0]} != {n_fake}; the anchor weights are "
            "built from that angle vector and must line up with it 1:1"
        )
        fake_w = self._regime_weight(
            self.apply_regime, w_lo, w_hi, n_fake, fake[0].device
        )
        if self.standardize:
            self._update_scale(real)
            if self.standardize_per_channel:
                s = [
                    (self._ch_std(i) / self.sigma_data).clamp_min(1e-6)
                    for i in range(len(real))
                ]
            else:
                _s = (self.feat_std / self.sigma_data).clamp_min(1e-6)
                s = [_s[i] for i in range(len(real))]
            real = [(r - self._mean(i)) / s[i] for i, r in enumerate(real)]
            fake = [(f - self._mean(i)) / s[i] for i, f in enumerate(fake)]

        return real, real_w, fake, fake_w, conditions

    def _featurize(self, px):
        out = self.featurizer(px)
        if isinstance(out, tuple):
            return out
        return out, {}

    def logits_loss(self, fake_w):
        """
        Pointwise alignment of the generated image's classifier logits with
        those of its own clean target, weighted by the same per-sample anchor
        weights as the transport (so apply_regime picks the angle band).

          mse    : mean over classes of the squared logit difference
          cosine : 1 - cos between the two logit vectors, each centered over
                   its classes first (per sample, as sphere.lpips does) --
                   the classifier's shared bias otherwise dominates the angle

        Returns the weighted term (logits_weight * loss).
        """
        assert self._logits is not None, (
            "logits_loss needs the featurizer's head output; call prepare_feats "
            "first, and build with logits_weight > 0"
        )
        tgt, inp = self._logits
        assert inp.shape == tgt.shape, f"logits {inp.shape} vs targets {tgt.shape}"
        inp = inp.float()
        tgt = tgt.float()

        inp_c = inp - inp.mean(dim=1, keepdim=True)
        tgt_c = tgt - tgt.mean(dim=1, keepdim=True)
        cos = F.cosine_similarity(inp_c, tgt_c, dim=1, eps=1e-6)  # [F]

        if self.logits_loss_type == "mse":
            per = (inp - tgt).pow(2).mean(dim=1)  # [F]
        else:  # cosine
            per = 1.0 - cos

        loss = _ddp_weighted_mean((per * fake_w).sum(), fake_w.sum())
        total = self.logits_weight * loss
        self.log_dict["score_logits_loss"] = total.detach()
        self.log_dict["score_logits_cos"] = cos.detach().mean()
        return total

    def latent_loss(self, fake_w):
        """
        Latent consistency on the featurizer's FINAL feature: the generated
        image's last-block / last-stage feature is pulled toward that of its
        target, weighted by the same per-sample anchor weights as the
        transport (so apply_regime picks the angle band).

          level  pooled : [F, D] global vector (CLS / pooled head input)
                 tokens : [F, N, D] last block's tokens; the loss is taken
                          per token and averaged over N
          type   cosine : 1 - cos over D (per token for `tokens`)
                 mse    : mean squared difference over D
          target        : the clean target image's feature

        Returns the weighted term (latent_weight * loss).
        """
        assert self._latent is not None, (
            "latent_loss needs the featurizer's final feature; call "
            "prepare_feats first, and build with latent_weight > 0"
        )
        tgt, inp = self._latent
        assert inp.shape == tgt.shape, f"latent {inp.shape} vs target {tgt.shape}"
        inp = inp.float()
        tgt = tgt.float()
        w = fake_w
        assert (
            w.shape[0] == inp.shape[0]
        ), f"latent rows {inp.shape[0]} vs anchor weights {w.shape[0]}"

        cos = F.cosine_similarity(inp, tgt, dim=-1, eps=1e-6)  # [F] or [F, N]
        if self.latent_loss_type == "mse":
            per = (inp - tgt).pow(2).mean(dim=-1)
        else:  # cosine
            per = 1.0 - cos
        if per.dim() == 2:  # tokens: average over N
            per = per.mean(dim=1)
            cos = cos.mean(dim=1)

        loss = _ddp_weighted_mean((per * w).sum(), w.sum())
        total = self.latent_weight * loss
        self.log_dict["score_latent_loss"] = total.detach()
        self.log_dict["score_latent_cos"] = cos.detach().mean()
        return total

    def frechet_loss(self, fake_w):
        """
        Representation FD between the generated and the clean final-stage
        features of the featurizer, on the forward prepare_feats already ran.
        Before fd_start_epoch only the clean statistics are advanced (warm-up)
        and a zero is returned; from there on the generated side is weighted
        per sample by the same anchor weights as the transport.

        Returns the weighted term (fd_weight * loss).
        """
        assert self._fd is not None, (
            "frechet_loss needs the featurizer's final tokens; call "
            "prepare_feats first, and build with fd_weight > 0"
        )
        tgt, inp = self._fd
        if self._epoch < self.fd_start_epoch:
            # warm up BOTH sides, the way the reference seeds its statistics
            # from sampled images before the first FD step. Warming only the clean
            # side leaves the generated EMA starting from a single batch, and
            # the FD then decays on the EMA's own schedule for thousands of
            # steps regardless of what the generator does.
            self.fd_loss.update_reference(tgt)
            self.fd_loss.update_generated(inp.detach(), fake_w)
            total = torch.zeros((), device=inp.device)
            self.log_dict["score_fd_raw"] = total
            self.log_dict["score_fd_n_eff"] = 0.0
            self.log_dict["score_fd_warmup_n"] = float(
                self.fd_loss._n_noisy_ema.item()
            )
            return total

        w = fake_w
        assert (
            w.shape[0] == inp.shape[0]
        ), f"fd rows {inp.shape[0]} vs anchor weights {w.shape[0]}"
        loss = self.fd_loss(tgt, inp.float(), w_noisy=w)
        total = self.fd_weight * loss
        self.log_dict["score_fd_loss"] = total.detach()
        for k, v in self.fd_loss.log_dict.items():
            self.log_dict[f"score_{k}"] = v
        return total

    """
    forward/backward
    """

    @torch.no_grad()
    def _sample_sigma(self, batch, device, dtype, transport=False):
        """
        https://github.com/NVlabs/edm/blob/main/training/loss.py#L73-L74
        """
        if transport:
            p_mean, p_std = self.gen_sigma_p_mean, self.gen_sigma_p_std
            s_min, s_max = self.sigma_min, self.gen_sigma_max
        else:
            p_mean, p_std = self.sigma_p_mean, self.sigma_p_std
            s_min, s_max = None, None  # no clamping for dsm training

        eps = torch.randn(batch, device=device)
        sigma = (p_mean + p_std * eps).exp()
        if s_min is not None and s_max is not None:
            sigma = sigma.clamp(min=s_min, max=s_max)
        return sigma.to(dtype).reshape(batch, 1, 1)

    def _transport_loss(self, real_net, fake_net, fake, fake_w, conditions):
        stats = {}
        B = fake.shape[0]
        sigma = self._sample_sigma(B, fake.device, fake.dtype, transport=True)
        eps = torch.randn_like(fake)

        with torch.no_grad():
            x_t = fake + sigma * eps
            pred_real = real_net(x_t, sigma, **conditions)
            pred_fake = fake_net(x_t, sigma, **conditions)

            r_p = fake.detach() - pred_real
            abs_err = r_p.abs()
            if self.normalizer_reduction == "sample":
                normalizer = abs_err.mean(dim=[1, 2], keepdim=True)
            else:  # "batch"
                normalizer = abs_err.mean()

            # The floor is a SAFETY NET against the 0/0 blowup when a sample
            # already sits on the manifold at this sigma, NOT a knob meant to
            # shape the bulk of the batch: it should sit well under the typical
            # normalizer (score_norm_clamped) and only ever catch the left tail.
            normalizer = normalizer.clamp_min(self.normalizer_floor)
            stats["norm_clamped"] = normalizer.detach().float().reshape(-1).mean()
            grad = (pred_fake - pred_real) / normalizer

            # per-sample RMS of the (PRE-clip) transport gradient -- the exact
            # unit grad_clip is expressed in (= L2 norm / sqrt(N*D)). Its bulk
            # (mean) and tail (max) are what you tune grad_clip against: set the
            # threshold just above the bulk so it caps the tail without touching
            # the majority. Logged whether or not clipping is on.
            rms = grad.pow(2).mean(dim=[1, 2], keepdim=True).clamp_min(1e-12).sqrt()
            rms_flat = rms.reshape(-1)  # [B]

            # per-sample clip on the transport gradient so an over-amplified
            # sample can't be shoved into the grid attractor in one step (the
            # "only some flowers collapse" symptom). Threshold is on the
            # per-sample RMS, i.e. the SAME units as score_grad_mag, so grad_clip
            # is interpretable independent of N/D (a raw L2-norm clip scales with
            # sqrt(N*D) ~ tens and would zero the signal for small thresholds).
            if self.grad_clip > 0:
                scale = (self.grad_clip / rms).clamp(max=1.0)  # [B, 1, 1]
                grad = grad * scale

            target = (fake - grad).detach()

        loss_per = 0.5 * ((fake - target) ** 2).mean(dim=[1, 2])  # [B]
        # left UNREDUCED, like _dsm: generator_loss reduces every layer's pair
        # (and the transport-balance statistic) in one collective.
        loss = ((loss_per * fake_w).sum(), fake_w.sum())
        stats.update(
            {
                "grad_mag": grad.abs().mean().detach(),  # post-clip |grad|
                "rms_mean": rms_flat.mean().detach(),  # pre-clip per-sample RMS, bulk
                "rms_max": rms_flat.max().detach(),  # pre-clip per-sample RMS, tail
            }
        )
        return loss, stats

    @torch.no_grad()
    def _transport_weights(self, rms):
        """Per-layer weights that equalize the pixel-space gradient magnitude.

        `rms` is the per-layer rms_mean ALREADY averaged across ranks -- that
        reduction rides in generator_loss's fused collective, because rms_mean
        is a LOCAL-batch statistic and every rank must derive identical weights
        for the buffers to stay in sync on their own (the generator path calls
        this module unwrapped, so DDP's buffer broadcast does not cover it).

        Returns a [J] tensor to multiply the per-layer transport losses by --
        scaling L_j by w_j scales the delivered gradient by exactly w_j, since
        dL_j/df_j = g_j / (N_j*D_j), so no custom backward is needed. Updates the
        running RMS estimate as a side effect. See the derivation in __init__.
        """
        rms = rms.float()  # [J]

        # same reasoning as _update_scale: transport_rms_ema is a running BUFFER
        # read back every step as `a = transport_rms_ema / (s * dof)`, so one nan
        # rms poisons the transport weights of every future step -- an
        # unrecoverable nan that looks like "the main grad went nan forever"
        # while the weights themselves are still fine. Freeze the EMA (and the
        # counters that drive its schedule) instead of updating it.
        rms_ok = bool(torch.isfinite(rms).all())
        if not rms_ok:
            logger.warning(
                "non-finite transport rms; freezing transport_rms_ema for this "
                "step (buffer left untouched)"
            )

        w = 1.0 / (self._tb_cnt + 1)
        calibrating = self._tb_cnt < self.transport_balance_calib_steps
        if not calibrating:
            w = max(w, 1.0 - self.transport_balance_decay)
        if rms_ok:
            self.transport_rms_ema.mul_(1.0 - w).add_(rms, alpha=w)
            self._transport_seen += 1
            self._tb_cnt += 1

        dof = torch.as_tensor(
            self._transport_dof, device=rms.device, dtype=rms.dtype
        )  # = sqrt(N_j * D_j) per scored layer

        if self.standardize:
            if self.standardize_per_channel:
                s = torch.stack(
                    [
                        1.0 / (1.0 / self._ch_std(i)).pow(2).mean().sqrt()
                        for i in range(self.num_layers)
                    ]
                ).to(rms.dtype)
            else:
                s = self.feat_std.to(rms.dtype)
            s = (s / self.sigma_data).clamp_min(1e-6)
        else:  # no standardization in the chain, so df/dPhi == 1
            s = torch.ones_like(rms)
        a = (self.transport_rms_ema / (s * dof)).clamp_min(1e-12)

        if self._tb_target_cfg is not None:
            self.transport_target.fill_(float(self._tb_target_cfg))
        elif calibrating and rms_ok:
            self.transport_target.copy_(a.mean())
        return (self.transport_target / a).clamp(0.05, 20.0)

    def gen_warmup_scale(self, epoch: int = None):
        """
        Scalar in [0, 1] multiplying the generator surrogate, ramping over
        [gen_warmup_start_epoch, gen_start_epoch) and pinned at 1 afterwards.
        Reads the epoch published by G1Loss.forward when not given one.

        Shapes (t = fraction through the ramp):
          "linear" : t                                -- constant slope
          "cosine" : 0.5 (1 - cos(pi t))              -- smooth at BOTH ends, so
                     no slope discontinuity at gen_start_epoch either.
        """
        e = self._epoch if epoch is None else epoch
        if e >= self.gen_start_epoch:
            return 1.0
        span = self.gen_start_epoch - self.gen_warmup_start_epoch
        if span <= 0 or e < self.gen_warmup_start_epoch:
            return 0.0
        t = (e - self.gen_warmup_start_epoch) / span
        if self.gen_warmup_shape == "linear":
            return t
        elif self.gen_warmup_shape == "cosine":
            return float(0.5 * (1.0 - math.cos(math.pi * t)))
        else:
            raise ValueError(f"unknown gen_warmup_shape: {self.gen_warmup_shape}")

    def generator_loss(self, fake, fake_w, fake_conditions=None):
        """
        DMD distribution-matching surrogate, averaged over layers. `fake` is a
        per-layer list carrying the decoder graph; the score nets are frozen
        here (no_grad). Per-layer transports are MEANed so `weight` stays
        interpretable independent of the layer count. Returns a scalar loss.
        """
        if fake_conditions is None:
            fake_conditions = {"class_labels": None}
        pairs, stats = [], []
        for j in range(self.num_layers):
            p, s = self._transport_loss(
                self.real_score_nets[j],
                self.fake_score_nets[j],
                fake[j],
                fake_w,
                fake_conditions,
            )
            pairs.append(p)
            stats.append(s)

        rms = (
            torch.stack([s["rms_mean"] for s in stats])
            if self.transport_balance
            else None
        )
        losses, rms = _ddp_weighted_mean_all(pairs, extra=rms)

        w_bal = None
        if self.transport_balance:
            w_bal = self._transport_weights(rms)
            loss = sum(w_bal[j] * losses[j] for j in range(len(losses)))
        else:
            loss = sum(losses)
        loss = loss / self.num_layers

        # fmt: off
        self.log_dict["score_gen_loss"] = loss.detach()
        self.log_dict["score_grad_mag"] = torch.stack([s["grad_mag"] for s in stats]).mean()
        self.log_dict["score_grad_rms"] = torch.stack([s["rms_mean"] for s in stats]).mean()
        self.log_dict["score_grad_rms_max"] = torch.stack([s["rms_max"] for s in stats]).max()
        self.log_dict["score_norm_clamped"] = torch.stack([s["norm_clamped"] for s in stats]).mean()
        # fmt: on

        if w_bal is not None:
            self.log_dict["score_transport_target"] = self.transport_target.clone()

        total = self.weight * loss
        w_warm = self.gen_warmup_scale()
        self.log_dict["score_gen_warmup"] = w_warm
        if w_warm != 1.0:
            total = w_warm * total
        return total

    def _dsm(self, net, x0, w, conditions):
        """
        https://github.com/NVlabs/edm/blob/main/training/loss.py#L66

        Eq.(8) in the EDM paper:
            loss = E[λ(σ)·||D(x_t, σ) - x_0||^2]
        """
        sigma = self._sample_sigma(x0.shape[0], x0.device, x0.dtype)
        eps = torch.randn_like(x0)
        x_t = x0 + sigma * eps
        pred = net(x_t, sigma, **conditions)
        lam = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)  # λ(σ)
        loss = lam * ((pred - x0) ** 2)  # [B, N, D]
        loss = loss.mean(dim=[1, 2])  # [B]
        return (loss * w).sum(), w.sum()

    def forward(self, real, real_w, fake, fake_w, conditions=None):
        n_score = self.num_layers

        if conditions is None:
            empty = {"class_labels": None}
            conditions = {"real": empty, "fake": empty}

        def _dsm_layers(nets, feats, w, cond):
            """Run the DSM over the scored layers. Returns the per-layer
            (weighted_sum, weight_sum) pairs STILL UNREDUCED -- they go into the
            one fused collective below, which is also where the summed loss gets
            logged."""
            return [self._dsm(nets[j], feats[j], w, cond) for j in range(n_score)]

        def _touch(mod):
            """This module is DDP-wrapped without find_unused_parameters, so
            every parameter must receive a gradient on every forward. A critic
            that has not started yet gets an exact-zero one: costs one reduction
            per param and no data forward."""
            return 0.0 * sum(p.float().sum() for p in mod.parameters())

        total, groups = 0.0, {}
        if self._epoch >= self.real_start_epoch:
            groups["real"] = _dsm_layers(
                self.real_score_nets, real, real_w, conditions["real"]
            )
        else:
            total = total + _touch(self.real_score_nets)

        if self._epoch >= self.fake_start_epoch:
            groups["fake"] = _dsm_layers(
                self.fake_score_nets, fake, fake_w, conditions["fake"]
            )
        else:
            total = total + _touch(self.fake_score_nets)

        if groups:
            flat = [p for tag in groups for p in groups[tag]]
            means, _ = _ddp_weighted_mean_all(flat)
            i = 0
            for tag, pairs in groups.items():
                l_tot = sum(means[i : i + len(pairs)])
                i += len(pairs)
                self.log_dict[f"score_dsm_{tag}_loss"] = (l_tot / n_score).detach()
                total = total + l_tot

        return total
