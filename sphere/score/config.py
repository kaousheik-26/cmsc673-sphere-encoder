config = {
    "score": {
        # score-net (EDM-preconditioned token denoiser)
        "arch": {
            "width": 384,  # floor width of the score net
            "width_margin": 0,  # extra width when the featurizer is wide
            "depth": 2,
            "real_depth": None,
            "fake_depth": None,
            "num_heads": 8,
            "drop_residual_path_prob": 0.1,
            "use_qk_norm": True,
            "sigma_data": 0.5,
            "sdpa_mode": "sdpa",
            "adaln_single": False,
            "sigma_cond": True,
            "class_cond": False,
        },
        "optimizer": {
            "lr": 1.0e-4,
            "min_lr": 1.0e-5,
            "real_min_lr": None,  # None -> use min_lr
            "fake_min_lr": None,
            "warmup_epochs": 0,
            "decay_epochs": 0,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,  # for both real and fake score nets
        },
        "sigma": {
            "p_mean": -1.2,
            "p_std": 1.2,
            "min": 2.0e-3,
            "max": 8.0,
            "gen_p_mean": None,
            "gen_p_std": None,
            "gen_max": None,
        },
        "augment": {
            "enabled": False,
            "type": "diffaug",
            "symmetric": False,  # dino or convnext space only
            # probs
            "brightness_prob": 0.0,
            "saturation_prob": 0.0,
            "contrast_prob": 0.0,
            "translation_prob": 0.0,
            "flip_prob": 0.0,
            "cutout_prob": 0.0,
            # strengths
            "brightness_strength": 0.5,
            "saturation_strength": 1.0,
            "contrast_strength": 0.5,
            "translation_ratio": 0.125,
            # hyperparams
            "translation_padding": "reflect",
            "cutout_ratio": 0.25,
        },
        # frozen pixel featurizer
        "dino": {
            "ckpt_path": "workspace/pretrained/discs/dino_vit_small_patch8_224.pth",
            "recipe": "S_8",  # S_16 | S_8 | B_16
            "layers": [5, 11],
            "img_size": 256,
        },
        "convnext": {
            "model_name": "timm/convnextv2_nano.fcmae_ft_in1k",
            "stages": [1, 2],  # 0-3; stride 4/8/16/32, dims 96/192/384/768
            "img_size": 256,  # inputs resized here before the backbone
        },
    },
    "loss": {
        # frozen pixel featurizer
        "space": "convnext",  # dino | convnext
        "weight": 1.0,
        "compile": False,
        "apply_regime": "hi",  # all | hi | lo
        "start_epoch": 0,
        "real_start_epoch": None,
        "fake_start_epoch": None,
        "gen_start_epoch": None,
        "gen_warmup_start_epoch": None,
        "gen_warmup_shape": "cosine",  # linear | cosine
        "grad_clip": 0.0,  # clip the gradient applied on the generator surrogate
        "skip_load_optim_state": False,
        # ------------------------------------------------------------------
        "standardize": False,
        "standardize_per_channel": False,
        "standardize_channel_floor": 0.1,
        "scale_calib_steps": None,
        "scale_ema_decay": 0.99,  # only used once calibration is over / disabled
        # ------------------------------------------------------------------
        "updates": 1,  # DSM optimizer steps per training iteration
        # ------------------------------------------------------------------
        "sync_interval": 1,  # how often the score nets synchronize across ranks.
        "normalizer_reduction": "batch",  # sample | batch
        "normalizer_floor": 1e-6,
        # ------------------------------------------------------------------
        "transport_balance": False,
        "transport_balance_target": None,
        "transport_balance_decay": 0.99,
        "transport_balance_calib_steps": 1000,
        # ------------------------------------------------------------------
        "logits_weight": 0.0,
        "logits_loss_type": "cosine",  # mse | cosine (per-sample centered)
        "logits_start_epoch": None,  # None -> start_epoch
        # ------------------------------------------------------------------
        "latent_weight": 0.0,
        "latent_loss_type": "cosine",  # cosine | mse
        "latent_level": "pooled",  # pooled ([B, D] global) | tokens ([B, N, D])
        "latent_start_epoch": None,  # None -> start_epoch
        # ------------------------------------------------------------------
        # representation FD (sphere/fd_loss.py) in the featurizer's
        # final-stage space, sharing prepare_feats' forward with the transport
        "fd_weight": 0.0,
        "fd_start_epoch": None,  # None -> start_epoch; clean stats warm up before
        "fd_pool": "mean",  # mean (token mean per image) | token (every token a sample)
        "fd_ema_decay": 0.999,
        "fd_use_clean_ema": True,
        "fd_use_noisy_ema": False,
        "fd_norm_eps": 0.01,
    },
}
