#!/bin/bash
#SBATCH --job-name=se2_latent1k
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=120G
#SBATCH -c 15
#SBATCH -o /fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text/logs/latent1k_%j.log
# Text-conditioned flow DiT in the latent space of a frozen SE2 (25 classes, up to 1000 clips each).
# Stages are skipped when their output exists, so resubmitting this script resumes.
#   A. CLAP audio embeddings for AudioCaps clips (4 GPUs)      -> data/clap_audiocaps
#   B. label-checked 25 x <=1000 dataset                        -> data1k
#   C. DiT training, 4 GPUs, $EPOCHS epochs (default 1)           -> $RUN (default runs/latent1k)
#   D. final evaluation (1 GPU)                                 -> runs/latent1k/eval
source /fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text/scripts/env.sh
set -o pipefail
RUN=${RUN:-runs/latent1k}
EPOCHS=${EPOCHS:-1}
EVAL_EVERY=${EVAL_EVERY:-10000}
END=$(( $(date +%s) + $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') ))
echo "job $SLURM_JOB_ID on $(hostname), ends $(date -d @$END)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f data/clap_audiocaps/audio_3.pt ]; then
  echo "== A: CLAP AudioCaps $(date)"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/clap_embed_all.py --shard $i --nshard 4 --src audiocaps --out data/clap_audiocaps > logs/clap_ac_$i.log 2>&1 &
  done
  wait
  grep -h SHARD_DONE logs/clap_ac_*.log
  [ -f data/clap_audiocaps/audio_3.pt ] || { echo "stage A failed"; exit 1; }
fi

if [ ! -f data1k/meta.json ]; then
  echo "== B: dataset $(date)"
  python scripts/prep_data1k.py --out data1k --workers 14 2>&1 | grep -v -i warn || { echo "stage B failed"; exit 1; }
fi

if ! grep -q TRAIN_DONE $RUN/train.out 2>/dev/null; then
  HOURS=$(python -c "print(round(($END - $(date +%s)) / 3600 - 2.5, 2))")
  echo "== C: train for up to $HOURS h $(date)"
  mkdir -p $RUN
  torchrun --standalone --nproc_per_node 4 scripts/train_latent.py --data $PWD/data1k --out $RUN --max_hours $HOURS --epochs $EPOCHS --eval_every $EVAL_EVERY \
    2>&1 | grep --line-buffered -v -i warn | tee -a $RUN/train.out
fi

echo "== D: eval $(date)"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_latent.py --ckpt $RUN/ckpt_last.pt --out $RUN/eval 2>&1 | grep -v -i warn | tee $RUN/eval.out
echo "ALL_DONE $(date)"
