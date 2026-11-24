#!/bin/bash
# usage: run_train.sh <run_name> <ngpu> [extra train.py args]   (run inside a GPU allocation)
source /fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text/scripts/env.sh
NAME=$1; NG=$2; shift 2
mkdir -p runs/$NAME
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
torchrun --standalone --nproc_per_node $NG scripts/train.py --out runs/$NAME "$@" 2>&1 | grep -v -i "warn" | tee -a runs/$NAME/train.out
