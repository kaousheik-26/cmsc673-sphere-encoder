#!/bin/bash
# 1-GPU smoke test of the latent pipeline (small sizes)
source /fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text/scripts/env.sh
set -e
S=smoke_latent; rm -rf $S; mkdir -p $S
python scripts/clap_embed_all.py --shard 0 --nshard 1 --src audiocaps --limit 3000 --out $S/clap_ac 2>&1 | grep -v -i warn | tail -n 2
python scripts/prep_data1k.py --out $S/data --n_train 15 --workers 8 --clap_ac $S/clap_ac 2>&1 | grep -v -i warn | tail -n 30
torchrun --standalone --nproc_per_node 1 scripts/train_latent.py --data $PWD/$S/data --out $S/run --bs 8 --hidden 256 --depth 2 --heads 4 \
  --max_steps 40 --save_every 20 --eval_every 20 --log_every 10 --eval_n 2 2>&1 | grep -v -i warn | tail -n 20
torchrun --standalone --nproc_per_node 1 scripts/train_latent.py --data $PWD/$S/data --out $S/run --bs 8 --hidden 256 --depth 2 --heads 4 \
  --max_steps 50 --save_every 20 --eval_every 1000 --log_every 10 --eval_n 2 2>&1 | grep -v -i warn | grep -E "resumed|TRAIN_DONE|Error"
python scripts/eval_latent.py --ckpt $S/run/ckpt_last.pt --out $S/run/eval --n 2 --steps 5 --cfg 1,3 --main 5,3 2>&1 | grep -v -i warn | tail -n 8
ls $S/run $S/run/eval/wav | head -30
echo SMOKE_OK
