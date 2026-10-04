#!/usr/bin/env bash
#
# sample images from pretrained models
#
# usage:
# ./scripts/sample.sh
#
# images are saved to workspace/visualization/<job_dir>

# models to sample from, one folder each under workspace/experiments
# checkpoints: https://huggingface.co/tomg-group-umd/sphere2
JOB_DIRS=(
  # "sphere2-base-flowers-256px"
  # "sphere2-base-flowers-512px"
  # "sphere2-large-flowers-256px"
  "sphere2-large-flowers-512px"
)


# =========================== Sampling ============================

for JOB_DIR in "${JOB_DIRS[@]}"; do
  ./run.sh sample.py \
    --job_dir $JOB_DIR \
    --out_dir visualization \
    --use_ema_model True \
    --load_ckpt_strict False \
    --num_gen_samples 64 \
    --batch_size_per_rank 16 \
    --forward_steps 1 2 4 \
    --sampling_init_angle 84.0 \
    --cache_sampling_noise True \
    --seed_sampling False \
    --use_cfg False \
    --cfg_min 0.0 \
    --cfg_max 5.0 \
    --cfg_gap 5.0 \
    --cfg_position angle \
    --grid_nrow 8 \
    --random_sample_classes True
done

# =================================================================
