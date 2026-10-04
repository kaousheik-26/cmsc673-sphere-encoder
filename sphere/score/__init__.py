import logging

import torch
import torch.nn as nn

from .config import config
from .score_loss import ScoreMatchingLoss
from .score_net import ScoreNet

logger = logging.getLogger(__name__)

__all__ = [
    "ScoreNet",
    "ScoreMatchingLoss",
    "config",
    "build_score_matching",
]


def build_score_matching(
    config: dict,
    anchor_params: tuple,
    device: torch.device,
    num_classes: int = 0,
) -> ScoreMatchingLoss:

    score_cfg = config["score"]
    arch = score_cfg["arch"]
    sigma = score_cfg["sigma"]
    loss_cfg = config["loss"]

    # both spaces derive their feature dims and token counts from the frozen
    # featurizer inside ScoreMatchingLoss, so nothing model-side is needed here.
    space = loss_cfg.get("space", "convnext")
    dino_cfg = score_cfg.get("dino", {})
    convnext_cfg = score_cfg.get("convnext", {})
    thres_deg, band_deg, anchor_mode = anchor_params

    return ScoreMatchingLoss(
        space=space,
        width=int(arch.get("width", 512)),
        width_margin=int(arch.get("width_margin", 128)),
        depth=int(arch.get("depth", 4)),
        real_depth=arch.get("real_depth", None),
        fake_depth=arch.get("fake_depth", None),
        num_heads=int(arch.get("num_heads", 8)),
        use_qk_norm=bool(arch.get("use_qk_norm", False)),
        drop_residual_path_prob=float(arch.get("drop_residual_path_prob", 0.0)),
        sigma_data=float(arch.get("sigma_data", 0.5)),
        sdpa_mode=str(arch.get("sdpa_mode", "manual")),
        sigma_cond=bool(arch.get("sigma_cond", True)),
        class_cond=bool(arch.get("class_cond", False)),
        num_classes=int(num_classes),
        adaln_single=bool(arch.get("adaln_single", False)),
        sigma_p_mean=float(sigma.get("p_mean", -1.2)),
        sigma_p_std=float(sigma.get("p_std", 1.2)),
        sigma_min=float(sigma.get("min", 2e-3)),
        sigma_max=float(sigma.get("max", 8.0)),
        gen_sigma_p_mean=sigma.get("gen_p_mean", None),
        gen_sigma_p_std=sigma.get("gen_p_std", None),
        gen_sigma_max=sigma.get("gen_max", None),
        weight=float(loss_cfg.get("weight", 1.0)),
        start_epoch=int(loss_cfg.get("start_epoch", 0)),
        real_start_epoch=loss_cfg.get("real_start_epoch", None),
        fake_start_epoch=loss_cfg.get("fake_start_epoch", None),
        gen_start_epoch=loss_cfg.get("gen_start_epoch", None),
        gen_warmup_start_epoch=loss_cfg.get("gen_warmup_start_epoch", None),
        gen_warmup_shape=loss_cfg.get("gen_warmup_shape", "linear"),
        anchor_thres_deg=float(thres_deg),
        anchor_band_deg=float(band_deg),
        anchor_mode=anchor_mode,
        apply_regime=loss_cfg.get("apply_regime", "hi"),
        dino_ckpt_path=dino_cfg.get("ckpt_path", None),
        dino_recipe=dino_cfg.get("recipe", "S_8"),
        dino_layers=dino_cfg.get("layers", [2, 5, 8, 11]),
        dino_img_size=int(dino_cfg.get("img_size", 224)),
        dino_device=device,
        convnext_model=convnext_cfg.get("model_name", "convnextv2_tiny.fcmae"),
        convnext_stages=convnext_cfg.get("stages", None),
        convnext_img_size=int(convnext_cfg.get("img_size", 256)),
        standardize=bool(loss_cfg.get("standardize", True)),
        standardize_per_channel=bool(loss_cfg.get("standardize_per_channel", False)),
        standardize_channel_floor=float(loss_cfg.get("standardize_channel_floor", 0.1)),
        scale_ema_decay=float(loss_cfg.get("scale_ema_decay", 0.99)),
        scale_calib_steps=loss_cfg.get("scale_calib_steps", None),
        normalizer_reduction=loss_cfg.get("normalizer_reduction", "sample"),
        normalizer_floor=float(loss_cfg.get("normalizer_floor", 1e-2)),
        grad_clip=float(loss_cfg.get("grad_clip", 0.0)),
        transport_balance=bool(loss_cfg.get("transport_balance", False)),
        transport_balance_target=loss_cfg.get("transport_balance_target", None),
        transport_balance_decay=float(loss_cfg.get("transport_balance_decay", 0.99)),
        transport_balance_calib_steps=int(
            loss_cfg.get("transport_balance_calib_steps", 200)
        ),
        logits_weight=float(loss_cfg.get("logits_weight", 0.0)),
        logits_loss_type=str(loss_cfg.get("logits_loss_type", "mse")),
        logits_start_epoch=loss_cfg.get("logits_start_epoch", None),
        latent_weight=float(loss_cfg.get("latent_weight", 0.0)),
        latent_loss_type=str(loss_cfg.get("latent_loss_type", "cosine")),
        latent_level=str(loss_cfg.get("latent_level", "pooled")),
        latent_start_epoch=loss_cfg.get("latent_start_epoch", None),
        fd_weight=float(loss_cfg.get("fd_weight", 0.0)),
        fd_start_epoch=loss_cfg.get("fd_start_epoch", None),
        fd_pool=str(loss_cfg.get("fd_pool", "mean")),
        fd_ema_decay=float(loss_cfg.get("fd_ema_decay", 0.999)),
        fd_use_clean_ema=bool(loss_cfg.get("fd_use_clean_ema", True)),
        fd_use_noisy_ema=bool(loss_cfg.get("fd_use_noisy_ema", False)),
        fd_norm_eps=float(loss_cfg.get("fd_norm_eps", 0.01)),
    ).to(device)
