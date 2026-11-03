# Bias in the spherical encoder's latent space

**Question.** Is there a systematic *bias* — a dominant shared direction / anisotropy
— in the sphere encoder's spherical latent space? We look into the latent space
**directly** (encode many images, inspect the geometry, and decode constructed
latents) instead of always seeding from a single image.

**Mechanism under test.** The model spherifies with `f = vector_rms_norm(zero_mean=False)`,
so the shared component of `enc(x)` is *not* removed before projecting onto the
hypersphere. If the encoder's outputs share a common component, every image latent
lands in a narrow cone around one direction — "the bias". The generator's own prior
is `f(randn)`, which is (near-)isotropic, so it serves as the unbiased baseline.

## What we measure

```
u_i = enc(x_i)      raw encoder output              [N, D]
z_i = f(u_i)        spherified latent the decoder consumes (unit RMS)
r_j = f(randn)      the sampling prior (isotropic baseline)
```

- mean-direction energy `||mean_i(z_i/|z_i|)||`  (0 = isotropic, 1 = colinear cone)
- mean pairwise cosine between distinct `z_i`, vs between `r_j`
- fraction of variance in a single direction (top-1 uncentered Gram eigenvalue)
- `cos(z_i, bias_dir)` where `bias_dir` = normalized mean latent
- decode the latent space directly: the bias direction, random sphere points, and
  each latent with the bias projected out.

## Result (49 diverse animals, 98,304-dim latent)

| measure | encoded images | noise prior `f(randn)` | isotropic ideal |
| :-- | :--: | :--: | :--: |
| mean pairwise cosine (distinct images) | **+0.69** | +0.00 | ~0.003 |
| mean-direction energy (0=iso, 1=cone) | **0.836** | 0.143 | 0 |
| variance in a single direction (top-1 Gram) | **70%** | 2.1% | — |
| … after removing that direction | 8.4% | — | — |
| `cos(latent, bias_dir)` | 0.84  [0.73, 0.88] | — | — |

**The bias is real and large.** Every image's spherical latent sits in a narrow
cone: any two images are ~69% aligned, and a *single* bias direction holds ~70% of
the latent variance (vs 2% for the isotropic noise prior).

**It is a structured pattern, not a scalar offset.** The raw per-latent DC scalar is
~0, so `zero_mean` would not remove it; the bias is intrinsic to the frozen
DINOv3 + learned adapter encoder (the classic transformer/DINO "cone effect"), and
spherify merely preserves it.

**The bias is largely content-free** (see `samples/results/montage_latent_probe.jpg`,
rows: `bias dir | random sphere | encoded | encoded-minus-bias`):
- decoding the bias direction alone gives a faint, generic "animal-on-ground" ghost
  that is nearly invariant to class conditioning;
- removing the dominant 70%-variance direction and decoding leaves every image fully
  recognizable — `decode(z)` vs `decode(z - bias)` differ by only ~19.8 dB, and
  reconstruction to the original drops just 20.0 → 17.0 dB (−3 dB);
- random isotropic sphere points decode to junk — coherent images live only inside
  the biased cone, which the decoder has learned to expect.

So image identity lives in the thin, more-isotropic residual; the big shared
direction is a mostly-content-free pedestal.

## Implication

Cosine similarity in this space is **bias-inflated**: image-to-image cosines are high
(~0.7–0.9) mainly because of the shared direction, so cosine is a loose measure of
content. A fairer "prior preservation" metric (e.g. for the `roundtrip_prior`
experiment) should center/whiten out the bias direction first and compare the
residual.

## Reproduce

```bash
PYTHONPATH=$PWD python latent_bias/latent_bias.py \
    --exp_dir workspace/experiments/sphere2-base-imagenet-256px \
    --items   latent_bias/items.json \
    --out_dir workspace/latent_bias/out
```

## Files

```
latent_bias/
├── README.md               this file
├── latent_bias.py          the probe (geometry stats + direct latent decodes)
├── items.json              image manifest
├── metrics/bias_stats.json measured numbers
└── samples/
    ├── inputs/             example input crops
    └── results/            montage_latent_probe.jpg
```
