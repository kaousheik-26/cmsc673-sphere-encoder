# SE2 for text-conditioned sound (AudioSet mel spectrograms)

Sounds are turned into images and passed through Sphere Encoder 2:
16 kHz AudioSet clip → mono 22.05 kHz → BigVGAN log-mel (80 bands, hop 256) → 768 frames (8.9 s) laid out as
3 stacked strips in one 256×256 grey image (exactly invertible, global loudness scale log 1e-5 … 1.5) →
SE2 → mel → BigVGAN v2 22 kHz vocoder → audio.

The scripts run from a working folder (`ROOT` in `textse2.py` / `env.sh`) that holds the uv venv, the HF cache
(released SE2 `sphere2-large-imagenet-fd-lite-256px`, BigVGAN, CLAP), data and runs; those are not in git.

## Text-conditioned fine-tuning of SE2

The released class embedding is replaced by `base + MLP(CLAP text embedding)` (base = released null class,
last layer zero-initialised, 10% caption dropout for CFG); encoder and decoder are fine-tuned with the released
losses minus the score/FD terms.

| File | Purpose |
|---|---|
| `audiomel.py` | mel ↔ SE2 image, BigVGAN-identical mel |
| `textse2.py` | text-conditioned SE2 wrapper, SE2 sampler with angle CFG |
| `prep_data.py` | first dataset: 5 classes × (20 train + 5 test) |
| `verify_labels.py`, `clap_embed_all.py`, `class_pool.py` | label check: CLAP rank of the label among all 456 AudioSet labels |
| `prep_data25.py`, `prep_data_classes.py` | label-checked 25-class dataset (10 test + up to 100 train per class) |
| `train.py`, `run_train.sh` | DDP training (torchrun) |
| `evaluate.py`, `run_eval.sh` | reconstruction, CLAP 25-way accuracy, sampler grid, nearest-train distance |
| `build_page.py`, `build_page_5c.py` | results pages |

Results (held-out clips; CLAP = share of generated clips assigned to the prompted class):

| Run | Data | Recon mel error | CLAP, train captions | CLAP, unseen captions |
|---|---|---|---|---|
| released SE2 | – | 7.6 / 8.2 dB | – | – |
| `text5c_6k` | 5 classes, 100 clips | 5.1 dB | 100% (5-way) | 88–100% |
| `text25c_20k` (step 16k) | 25 classes, 2,220 clips | 3.1 dB | 44–46% (25-way; real audio 69%) | 34–41% |

## Frozen SE2, text-to-latent flow model

SE2 encoder and decoder are frozen (the audio-fine-tuned `text25c_20k` EMA weights; the released decoder alone is
8.2 dB on mels). Only a 174M DiT is trained, with flow matching on the spherified SE2 latent (256 tokens × 384),
conditioned on CLAP text (template class captions, or the clip's human AudioCaps captions). Sampling: Euler with
CFG → latent → frozen decoder → mel → BigVGAN.

| File | Purpose |
|---|---|
| `prep_data1k.py` | 25 classes × up to 1,000 clips: AudioSet-strong + AudioCaps clips whose captions name exactly one class and whose CLAP rank for that class is ≤ 10; same 250 test clips as the 25-class set |
| `latent_dit.py` | DiT (adaLN-Zero, QK-norm), flow-matching loss, Euler/CFG sampler |
| `train_latent.py` | DDP training, resumable, `--epochs` or `--max_hours`, periodic CLAP check |
| `eval_latent.py` | frozen-SE2 ceiling, CLAP accuracy over an Euler-steps × CFG grid, nearest-train distance |
| `sbatch_latent1k.sh` | one Slurm job: CLAP over AudioCaps → dataset → training → evaluation (stages resume); `RUN`, `EPOCHS`, `EVAL_EVERY` via environment |
| `smoke_latent.sh` | 1-GPU end-to-end smoke test |

Data: 15,173 train clips (7 classes at 1,000; subway, vacuum, rapping, engine knocking, choir, chainsaw,
fire-engine siren and truck stay at 79–221). Frozen-SE2 ceiling on the test clips: 3.11 dB.
1 epoch (60 steps): CLAP 4% = chance, every prompt gives the same averaged sound. A 100-epoch run is queued.
