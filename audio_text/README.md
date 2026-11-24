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
