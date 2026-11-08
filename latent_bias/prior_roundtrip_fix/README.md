# Improving the prior through the A→B→A round trip

**Problem.** When we edit an image A→B (e.g. lion→zebra) and then back B→A, the
model does **not** return to the original prior: the content-only (bias-removed)
round-trip cosine is only ~0.24. Worse, going A→B repaints the **whole scene**, not
just the subject. This directory collects everything we tried to fix that, ending
with a method that recovers the prior **mask-free**, trained on **20 images of a
single class** (lions, ImageNet `n02129165`, class id 294).

The metric throughout is the **bias-removed (residual) round-trip content-cosine**
between `encode(A)` and `encode(A→B→A)`, averaged over several target classes B
(zebra / tabby / flamingo), on **held-out** lions. See `../README.md` for why the
bias must be projected out before comparing.

## Why the whole scene changes (root cause)

`edit()` does `encode(x,null) → spherify(·, 85°) → decode(·, class B)`. Two things
are **global**, neither localized to the subject:
1. the noise injection `z ← cos α·z + sin α·e` adds the same `e` to **every** token;
2. the class label B is fed **globally** through adaLN to every token.
So every token is pushed to noise and re-rendered as B. Nothing tells the background
tokens to stay. The fix has to make **both** of these spatial.

## What we tried (in order)

| # | method | mask at inference? | round-trip cos | verdict |
| :-- | :-- | :--: | :--: | :-- |
| baseline | plain `edit()` | — | 0.24 | prior drifts; scene repainted |
| Tier 1 | **latent token-masking** (pin background tokens to the clean latent, rembg mask) | yes | **0.83** | works, but needs a mask |
| Tier 3 | differentiable **cycle-consistency** LoRA | no | 0.36 | real gain, but A→B under-edits (cheats) |
| Tier 4 | cycle + outward **foreground-distillation** + bg-preservation LoRA | no | 0.35 | edits honest+strong, but plateaus; bg still drifts |
| Tier 5 | **spatial-conditioning gate** `g·emb(B)+(1−g)·emb(null)` | no | 0.25 | **failed** — gate read the 85°-noised latent (≈pure noise) so it couldn't localize; and gating conditioning alone leaves the global *noise* untouched |
| **Tier 6** | **learned latent-masking** | **no** | **0.73** | **the fix** |

Earlier, broader LoRA attempts (`../experiments`, 325-image class-agnostic distill)
gave no gain or **bleached** outputs — scaling LoRA capacity was the wrong lever.

## Tier 6 — learned latent-masking (the fix)

Combine the two things that worked, and fix Tier 5's two bugs:
- a tiny gate `g(z₀) ∈ [0,1]` per token reads the **clean** content latent
  `z₀ = f(encode(x,null))` (not the noised latent — that was Tier 5's bug), so it
  can actually see the subject;
- `g` drives **both** global factors spatially, i.e. Tier 1's latent-masking but
  with a *learned* gate instead of the rembg mask:
  - latent:  `z ← f( g·(noised z) + (1−g)·z₀ )`  → background tokens keep the clean scene,
  - conditioning: `cond = g·emb(B) + (1−g)·emb(null)` → background gets null class.

Only the ~66 K-param gate is trained (encoder+decoder frozen), on 20 lions, with:
`L_gate` (g vs downsampled rembg mask) + outward foreground-distillation to the base
model's B edit + background preservation + cycle-consistency + a DINOv3-feature term.
At inference **no mask is used** — `g` is predicted from the latent.

**Result (mask-free, held-out lions):** round-trip content-cos **0.29 → 0.73**
(Tier 1 with a real mask = 0.83). The gate loss fell 0.52 → 0.07 — i.e. the gate
**learned the lion silhouette from the latent alone** (see
`figures/tier6_learned_mask_lion.png`, last panel). The A→B→A trip keeps the
original rock+grass scene and returns a lion, while the base model drifts to a new
scene (panel 3).

## Files

```
prior_roundtrip_fix/
├── README.md
├── gate.pth                 trained Tier-6 gate (66K params)
├── scripts/
│   ├── exp_common.py        shared model-load / metrics helpers
│   ├── exp_latent_masked.py Tier 1 (training-free, rembg mask)
│   ├── tier3_prep.py        pull 20+4 lion images + rembg masks from ImageNet
│   ├── tier3_train.py       cycle-consistency LoRA
│   ├── tier4_train.py       cycle + fg-distillation LoRA
│   ├── tier5_train.py       spatial-conditioning gate (negative result)
│   └── tier6_train.py       learned latent-masking  ← the fix
├── metrics/  tier1,3,4,5,6.json   (per-example round-trip cosines)
└── figures/  one lion A→B→A strip per tier (+ gate map for 5/6)
```

Train data: 20 lion crops (256²) + rembg masks; held-out: 4 more ImageNet lions +
the original round-trip lion. Hardware: 1× RTX A6000, `.venv_sphere2`.

## Takeaways

- The apparent prior drift is caused by **two** global operations (noise + class
  conditioning); fixing only one (Tier 5) does nothing.
- A **learned, latent-read gate** can reproduce a salient-object mask from the
  spherical latent with no segmentation model at inference, from only 20 examples.
- Specializing to one class with a cycle + mask objective recovers the prior nearly
  to the masked upper bound, mask-free.
