#!/usr/bin/env bash

DIST_MODE="local"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d)
      DIST_MODE="distributed"
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"


# ========================= E2E Training ==========================

IMAGE_SIZE=512

./run.sh \
  --dist_mode=$DIST_MODE \
  train.py \
  --dataset_name imagenet \
  --image_size $IMAGE_SIZE \
  --warmup_epochs 10 \
  --epochs 300 \
  --vis_interval 10 \
  --ckpt_save_interval 10 \
  --cleanup_ckpt True \
  --log_interval 100 \
  --concat_train_val_splits False \
  --batch_size_per_rank 16 \
  --grad_accum_steps 1 \
  --learning_rate 1.0e-4 \
  --min_lr 1.0e-5 \
  --encoder_lr_scaler 0.1 \
  --encoder_lr_scaler_min 0.1 \
  --encoder_lr_scaler_decay_epochs 0 \
  --decay_lr False \
  --optimizer_name adamw \
  --rescale_lr_with_batch_size True \
  --crop_mode center_adm \
  --compile_model False \
  --dtype bfloat16 \
  --sdpa_mode sdpa_fp32 \
  --grad_comm_dtype fp32 \
  --use_wandb False \
  --vit_enc_model_size base \
  --vit_dec_model_size base \
  --drop_residual_path_prob 0.1 \
  --pixel_head_type linear \
  --use_qk_norm True \
  --out_dir experiments \
  --perceptual_convnext_aug 0.5-0.0625-0.85-1.0 \
  --perceptual_convnext_loss_type cosine \
  --perceptual_convnext_loss_reduction mean \
  --pix_recon_dist_loss_weight 1.0 \
  --pix_recon_perc_loss_weight 1.0 \
  --use_ema True \
  --ema_model_decay 0.99985 \
  --use_ema_scheduler False \
  --spherify_model False \
  --spherify_mode global \
  --use_angle_condition True \
  --angle_condition_mode both \
  --noise_sigma_max_angle 90.0 \
  --weight_anchor_loss True \
  --weight_anchor_cutoff_params 70-5-soft \
  --compression_ratio 1.0 \
  --latent_resolution low \
  --lat_con_loss_weight 0.1 \
  --lat_con_apply_regime hi \
  --lat_con_use_diffaug False \
  --lat_con_freeze_encoder False \
  --load_pretrained_encoder dinov3-vits16plus-pretrain-lvd1689m \
  --pretrained_encoder_train_last_n_blocks 0 \
  --pretrained_encoder_train_norm_ls_layers False \
  --pretrained_encoder_train_final_norm False \
  --pretrained_encoder_interpolate_tokens False \
  --pretrained_encoder_fix_compression_ratio False \
  --pretrained_encoder_interpolate_pos_embed False \
  --freeze_encoder True \
  --use_score True \
  --score_set \
    score.arch.width=384 \
    score.arch.depth=2 \
    score.arch.num_heads=8 \
    score.arch.use_qk_norm=True \
    score.arch.sigma_cond=True \
    score.arch.class_cond=True \
    loss.space=convnext \
    loss.normalizer_reduction=sample \
    loss.standardize=True \
    loss.standardize_per_channel=True \
    loss.grad_clip=1.0 \
    loss.weight=1.0 \
    loss.transport_balance=True \
    loss.transport_balance_calib_steps=10000 \
    loss.start_epoch=100 \
    loss.gen_start_epoch=115 \
    loss.gen_warmup_start_epoch=105 \
    loss.gen_warmup_shape=cosine \
    loss.apply_regime=hi \
    loss.normalizer_floor=1e-6 \
    loss.updates=1 \
    loss.skip_load_optim_state=False \
    score.optimizer.lr=5e-4 \
    score.optimizer.fake_min_lr=5e-5 \
    score.optimizer.real_min_lr=5e-5 \
    score.optimizer.warmup_epochs=5 \
    score.optimizer.decay_epochs=200 \
    score.optimizer.grad_clip=0.15 \
    score.sigma.gen_p_mean=-1.5 \
    score.sigma.gen_p_std=1.0 \
    score.sigma.gen_max=6.0 \
    score.augment.enabled=True \
    score.augment.type=diffaug \
    score.augment.flip_prob=0.5 \
    score.augment.translation_prob=1.0 \
    score.augment.translation_ratio=0.015625 \
    score.augment.translation_padding="reflect" \
    score.augment.symmetric=True \
    score.dino.layers=[5,11] \
    score.dino.img_size=$IMAGE_SIZE \
    score.convnext.model_name=convnextv2_nano.fcmae_ft_in1k \
    score.convnext.stages=[1,2] \
    score.convnext.img_size=$IMAGE_SIZE \
  --log_interval 1 \
  --run_str 20261001_e2e

# =================================================================
