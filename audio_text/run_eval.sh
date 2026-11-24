#!/bin/bash
# usage: run_eval.sh <run_name> [extra evaluate.py args]
source /fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text/scripts/env.sh
NAME=$1; shift
python scripts/evaluate.py --ckpt runs/$NAME/ckpt_last.pt --out runs/$NAME/eval "$@" 2>&1 | grep -v -i "warn" | tee runs/$NAME/eval.out
