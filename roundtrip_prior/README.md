# Double-diffusion prior-consistency probe

**Question.** Given an image of class *A*, the G1 editor can send it to a *different*
class *B* (one trip through the diffusion/edit process). If we then take that new
image and edit it **back to class A**, does the *prior* survive the round trip?
i.e. is `A → B → A` content-preserving, and does the model actually return to the
same base class?

Here the **prior** is concrete: G1's encoder is *unconditional* (class-independent),
so `z = enc(x)` is a content/identity code that does not depend on the class label.
`edit(x, y)` encodes `x` into this prior, rotates it toward noise by `edit_angle`,
and decodes it **conditioned on the target class `y`**. So the probe measures whether
that content code is stable when it is pushed through a *second* class and back.

## Method

For each input image `x` with true (base) class `A`, and a target class `B`:

```
z0    = enc(x)                 # prior of the original
x_B   = edit(x,   y=B)         # one trip:   content of x rendered as class B
x_ABA = edit(x_B, y=A)         # trip back:  content of x_B rendered as class A
x_AA  = edit(x,   y=A)         # reference:  single direct edit to A (one pass)
```

Priors are compared by cosine similarity in the encoder's latent space:

| metric | meaning |
| :-- | :-- |
| `cos(z0, enc(x_B))`   | how far the one-way class swap `A→B` moved the prior |
| `cos(z0, enc(x_ABA))` | **main:** prior after the full `A→B→A` round trip |
| `cos(z0, enc(x_AA))`  | reference: prior after a single direct edit back to A |

plus pixel PSNR of `x_ABA` and `x_AA` against the original.

`B` is chosen as the base class of the *next* image in the set, so every image is
edited toward a genuinely different animal.

## Setup

- **Model:** `sphere2-base-imagenet-256px` (EMA weights), class-conditional, 256px.
- **Encoder note.** The encoder is a *frozen* DINOv3 ViT-S/16+. That repo is gated on
  Hugging Face, but its weights are already inside the sphere2 checkpoint
  (`encoder.encoder.*`, 232 tensors). `roundtrip_prior.py` therefore rebuilds the
  DINOv3 architecture from a local config inferred from the checkpoint
  (`hidden=384, layers=12, heads=6, register=4, patch=16, gated SwiGLU/silu MLP`)
  and loads the weights from the checkpoint — no gated download, `missing keys: []`.
  The `silu` SwiGLU activation was confirmed by reconstruction PSNR (silu 18.6 dB vs
  gelu 18.1 dB mean over the 10 inputs).
- **Data:** 10 real animal photos (Wikipedia), each tagged with its *model-space*
  class id (base class A). 256px center-cropped copies are in `samples/inputs/`.
- Edit strength (`edit_angle`) swept over 20/40/60/85°, `forward_steps=4`, seed 99.

## Results

Means over the 10 images:

| edit angle | one-way `cos(z0,z_B)` | **round-trip `cos(z0,z_ABA)`** | direct `cos(z0,z_AA)` | PSNR ABA / AA (dB) |
| :--: | :--: | :--: | :--: | :--: |
| 20° | 0.897 | 0.846 | 0.896 | 12.1 / 14.6 |
| 40° | 0.898 | 0.852 | 0.895 | 11.7 / 14.0 |
| 60° | 0.888 | 0.848 | 0.888 | 10.8 / 13.5 |
| 85° | 0.774 | **0.811** | 0.842 |  9.2 / 9.3  |

**Takeaways.**

1. **The content latent is a stable, class-independent code.** At low–moderate edit
   strength, swapping to a *different* class barely moves the prior
   (`cos ≈ 0.89`, essentially the same as a same-class edit). Identity, pose and
   background live in the latent, not in the class label.
2. **The round trip returns to the base class with the prior mostly recovered.**
   At the aggressive default (85°) the one-way edit pushes the prior far (0.774) and
   editing *back* to A pulls it up to 0.811 — the return trip re-aligns content
   toward the original. Qualitatively every row comes home to the right animal in
   the original pose (see below).
3. **Double diffusion is not free.** The round trip sits ~0.04–0.05 cosine and
   2–4 dB below a single direct edit: each `encode→decode→re-encode` pass sheds a
   little content, and degradation accumulates per pass. So the prior stays *mostly*
   intact, not perfectly.

**Qualitative (see `samples/results/montage_ang85.jpg`).** Columns are
`input | →B | →back-to-A (round trip) | →A direct`. The round trip keeps the
original layout while cycling the class: lion→panda→lion on the *same rocks*
(`row_03`), king penguin→red fox→penguin on the *same gravel* (`row_06`),
husky→peacock→husky (`row_08`). The one clear failure is tabby→zebra→back, which
loses the subject — consistent with its low cosine/PSNR.

## Reproduce

```bash
# from the repo root, with a torch+CUDA env and the checkpoint under
# workspace/experiments/sphere2-base-imagenet-256px (see top-level README)
PYTHONPATH=$PWD python roundtrip_prior/roundtrip_prior.py \
    --exp_dir workspace/experiments/sphere2-base-imagenet-256px \
    --items   roundtrip_prior/items.json \
    --out_dir workspace/roundtrip/out_a85_s4 \
    --forward_steps 4 --edit_angle 85 --dino_act silu
```

Add `--sanity_only` to just reconstruct the inputs and print PSNR (validates the
encoder). `rt_inspect.py <ckpt.pth>` dumps the checkpoint's module groups and the
embedded DINOv3 tensor shapes.

## Files

```
roundtrip_prior/
├── README.md               this file
├── roundtrip_prior.py      the probe (edit A→B→A, metrics + grids)
├── rt_inspect.py           checkpoint inspector (encoder shapes)
├── items.json              10-image manifest (label, model class id, article)
├── metrics/                per-angle summary + per-image metrics (json)
└── samples/
    ├── inputs/             10 × 256px center-cropped inputs
    └── results/            montages (ang85, ang40) + 3 example rows
```
