from sphere.model import G1


def get_model_args(args, for_train=False, **kwargs):

    model_args = dict(
        input_size=args.image_size,
        patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size,
        vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size,
        spherify_model=args.spherify_model,
        spherify_mode=args.spherify_mode,
        pixel_head_type=args.pixel_head_type,
        pixel_head_use_tanh=args.pixel_head_use_tanh,
        in_context_size=args.in_context_size,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        use_angle_condition=args.use_angle_condition,
        angle_condition_mode=args.angle_condition_mode,
        load_pretrained_enc=args.load_pretrained_encoder,
        pretrained_encoder_interpolate_tokens=args.pretrained_encoder_interpolate_tokens,
        pretrained_encoder_interpolate_pos_embed=args.pretrained_encoder_interpolate_pos_embed,
        pretrained_encoder_fix_compression_ratio=args.pretrained_encoder_fix_compression_ratio,
        sdpa_mode=args.sdpa_mode,
        freeze_encoder=args.freeze_encoder,
        enc_train_last_n_blocks=args.pretrained_encoder_train_last_n_blocks,
        enc_train_norm_ls_layers=args.pretrained_encoder_train_norm_ls_layers,
        enc_train_final_norm=args.pretrained_encoder_train_final_norm,
        use_qk_norm=args.use_qk_norm,
    )

    if for_train:
        model_args.update(
            drop_residual_path_prob=args.drop_residual_path_prob,
            # the FD loss reads the same re-encoded latent as lat-con
            use_latent_consistency=(
                args.lat_con_loss_weight > 0 or getattr(args, "fd_loss_weight", 0.0) > 0
            ),
            latent_consistency_use_ema=args.use_ema
            and args.load_pretrained_encoder is None,
            latent_consistency_freeze_encoder=getattr(
                args, "lat_con_freeze_encoder", False
            ),
        )
    return model_args


def build_model(args, for_train=False, **kwargs):
    return G1(**get_model_args(args, for_train=for_train, **kwargs))
