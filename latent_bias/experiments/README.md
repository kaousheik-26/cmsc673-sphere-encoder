# Follow-up experiments on the latent bias

Building on the probe in `../latent_bias.py` (see `../README.md`), these scripts
characterize the bias further and then test whether we can **preserve the prior**
through an A→B→A round trip despite it. `exp_common.py` holds shared helpers
(offline DINOv3 patch, model load, encode, debiased cosine, PSNR).

Run everything from the repo root (paths inside are repo-root relative).

## Measurement (characterization)

| script | question | headline result |
| :-- | :-- | :-- |
| `exp_geometry.py` | anisotropy, cone effect, OOD inputs (gray/noise), single-direction energy | ~70% variance in one direction; even gray/noise land in the cone (cos ~0.66) → bias is encoder-intrinsic |
| `exp_attribution.py` | which component creates the bias; base vs fd-lite checkpoint | bias comes from the DINOv3 last block; identical in base and fd-lite |
| `exp_roundtrip_debiased.py` | re-score the `roundtrip_prior` A→B→A with the bias projected out | raw cosine ~0.8 is inflated; residual (content-only) cosine is 0.23–0.61 |

`../metrics/geometry_report.json` holds the measured numbers.

## Prior round-trip (application)

| script | what it shows |
| :-- | :-- |
| `exp_stepwise.py` | lion→cat→lion, every diffusion step, tracking content-cosine to the original |
| `exp_stepwise5.py` | the same stepwise round trip across 5 animal/flower classes |
| `exp_controlled.py` | **segmentation-controlled** round trip: a U²-Net (rembg) foreground mask anchors the original background at every step, so only the masked subject is edited |

### Segmentation control result (5 examples, edit_angle 85°)

Compositing the original background back in at each step stops the composition
from drifting through the round trip:

| | mean content-cos to original | mean PSNR |
| :-- | :--: | :--: |
| **controlled** (mask-anchored) | **+0.75** | **15.3 dB** |
| uncontrolled | +0.23 | 8.7 dB |

Numbers per example in `../metrics/controlled.json` and `../metrics/stepwise5.json`.

**Caveat.** Where the rembg mask was poor or tiny (peacock 4%, penguin 18%) the
high score is partly just copying the original — a mask-quality artifact, not pure
model skill. The masked *subject* still degrades at 85°; a lower edit angle or
latent-space (RePaint-style) compositing would keep it cleaner.

## Dashboard

`../dashboard/index.html` is a self-contained page (figures in `img/`, galleries
fetched from `controlled.json` / `stepwise5.json`) visualizing all of the above.
Serve it with `python -m http.server` from `../dashboard/`.
