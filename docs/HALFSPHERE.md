# halfsphere: half-image completion on the Sphere Encoder 2 latent sphere

Given the left (or top) half of an image, find latents on SE2's sphere whose decodings agree
with it, and through them the missing half. The prior on the sphere is flat, so the answer is a
set (the fiber), not a point. Three solver families, all optimisation only, no training:

| # | solution | plan label | file | what it does |
|---|---|---|---|---|
| 1 | **Repaint** | S2 | `halfsphere/repaint.py` | encode, rotate toward fixed noise by angle a_k, decode, pin the observed half; repeat with a_k decreasing |
| 2 | **Ridge Map** | S4 | `halfsphere/ridgemap.py` | matrix-free Jacobian tools at a solution: fiber dimension, completion modes, Repaint's contraction rate, LM refinement, block CCA |
| 3 | **Steer** | S1 | `halfsphere/steer.py` | Riemannian gradient descent on the sphere with an encode-decode cycle regulariser |

The solution plan (written before the code) calls them S2, S4 and S1; seed tags inside
`sphere.rng.fold_in` keep those labels so noise draws match runs made before the renaming, and
`records.read_rows` maps rows of those older runs to the new names.

Shared pieces: `backbone.py` (the only file that touches the authors' `G1`), `geometry.py`,
`masks.py`, `data.py`, `metrics.py`, `records.py`. What was checked against the authors' code
is in `docs/reconciliation.md`.

## Setup

The code expects the repo root as the working directory and on `PYTHONPATH`, checkpoints under
`workspace/experiments/<job_dir>/` (as the repo README describes), the gated
`facebook/dinov3-vits16plus-pretrain-lvd1689m` in the Hugging Face cache, and validation images
under `workspace/datasets/<name>/` in the layout `sphere/loader.py` reads. Extra packages beyond
`requirements.txt`: `pytest` and `scipy` (tests and the Lanczos cross-check).

## Run

```bash
python -m pytest tests/                                   # CPU, dummy backbone
python scripts/smoke_se2.py                               # real-weight contract checks
python scripts/smoke_se2.py --only_equivalence True       # rule: reproduce sample.py exactly
python scripts/run_repaint.py --stage coarse --n_images 64   # Repaint schedule sweep
python scripts/summarize_repaint.py workspace/runs/repaint_coarse/<run>
python scripts/run_ridgemap.py h2|h3|modes|lm|h1 ...          # Ridge Map analyses
python scripts/run_steer.py --configs '[{"lam_cyc": 0.1}]'    # Steer
```

Every run writes `workspace/runs/<experiment>/<stamp>-<sha>/` with `run.json` (config, git
state, seeds, checkpoint sha256, environment) and `rows.jsonl` (one line per image, method,
config and seed). `workspace/` is gitignored.

## Conventions

Images `[B,3,H,W]` in `[-1,1]`; latents `[B,256,384]` fp32 on the radius-sqrt(L) sphere;
angles in degrees at every interface; Ridge Map runs in fp32 inside
`bb.differentiable_attention()`; a completion is always reported with how it was selected.
