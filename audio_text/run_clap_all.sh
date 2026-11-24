#!/bin/bash
source /fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text/scripts/env.sh
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/clap_embed_all.py --shard $i --nshard 4 --out data/clap_all > logs/clap_all_$i.log 2>&1 & done
wait; echo ALL_DONE
