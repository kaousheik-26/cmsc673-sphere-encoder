# Results: half-image completion on the SE2 sphere (first pass, 2026-10-08)

Model `sphere2-base-flowers-256px` (EMA, frozen; nothing was trained). Flowers-102 validation,
left half observed unless stated. Every number below is recomputed from the run directories by
`scripts/compile_results.py`; the results sheet and the three solution sheets are built from the
same output. The ImageNet model, the planned primary, is not covered: its validation set needs
dataset access not yet granted.

Three solutions, in the order they build on each other (the solution plan's labels in brackets):

1. **Repaint** (S2): forget a little, let the decoder repaint, restore the visible half; repeat.
2. **Ridge Map** (S4): linearise the decoder at one answer and map which directions keep the
   visible half fixed; also diagnoses Repaint and polishes any answer.
3. **Steer** (S1): steer the latent directly by Riemannian gradient descent on the sphere.

## How every completion was selected

| method | selection | seeds |
|---|---|---|
| mirror | observed half reflected into the missing half, pinned | none |
| oracle | `D(v(x), 0)` of the TRUE image, observed half pinned; told the answer, so an upper reference | none |
| Repaint A | K=4 steps at constant 50 deg, mirror fill, one fixed noise draw, final alpha=0 pass, pinned. Chosen from 156 schedules as a Pareto point (LPIPS_B vs KID) | `fold_in(0, "s2", batch_start, draw)` |
| Repaint B | the plan's default: K=12, 80 to 20 deg linear; otherwise as A | same |
| Repaint sets | 8 independent noise draws of A or B | draws 0..7 |
| Repaint-tok | observed tokens pinned at a small angle, missing tokens at a large one, effective angle to the decoder (Proposition 9) | same |
| Ridge Map LM | 10 accept/reject Levenberg-Marquardt steps, 3 deg trust region, from Repaint A outputs or from the true latent | none |
| Steer | Riemannian descent from the encoded mirror fill (restart 0) and from rotations of it toward random latents by 60 deg (restarts 1+); the restart with the lowest observed-half data loss is reported, pinned | `fold_in(0, "s1", batch_start)` |

Seed tags keep the plan's labels so draws reproduce runs made before the solvers were named.

## Headline findings

### Solution 1: Repaint

1. **Realistic but not settling.** Over 156 schedules on 64 images, KID is ten times below mirror
   fill, but the loop never settles (the latent still moves about 2.4 deg per step) and leaves a
   seam (2.4 or more), because the decoder redraws the observed half slightly shifted. More steps
   score worse.
2. **Pinning observed tokens (Repaint-tok) did not help**, although the token blocks are strongly
   dependent (Ridge Map, H1); it is not kept.

### Solution 2: Ridge Map

3. **The fiber is enormous (H2 supported).** At the true latent the observed-half Jacobian has
   numerical rank about 276 (eps 1e-2) and 683 (eps 1e-3) of 98,303 directions, so roughly 97,600
   directions leave the observed half unchanged to first order. Validated: Lanczos matches
   ARPACK to 2e-7, and SLQ is unchanged from 80 to 200 Lanczos steps.
4. **Left and right token blocks are strongly dependent (H1 supported)**: top canonical correlation
   0.999 against a shuffled null of 0.48.
5. **Repaint is not contractive (H3 falsified).** Median contraction above 1 at every constant angle
   from 10 to 80 deg, which explains finding 1.
6. **Completion modes and LM polish work as designed**: modes change the hidden half far more than
   the visible half (leakage median 0.022); 10 LM steps lift Repaint A's observed-half PSNR from
   20.2 to 26.4 dB.

### Solution 3: Steer

7. **Steer fits the observed half better than the true latent does**: 25.8 dB before pinning against
   21.3 dB for the oracle, and about half the seam of Repaint. It stays close to its mirror start.
8. **The encoder in the loop did not help (H5 falsified on this checkpoint).** R_cyc pulls latents
   toward the data-token manifold but leaves the completion no better; decoder-only Steer matches
   Repaint. This checkpoint was trained without latent consistency, so ImageNet must be checked.
9. **Diversity (H4 mixed).** Steer over 8 restarts is both more diverse and closer to the truth than
   Repaint A; Repaint B is the most diverse but furthest from the truth.

### Across solutions

10. **Distance to the truth barely separates methods** (LPIPS_B 0.61 to 0.68 for every method),
    which is what a 97,600-dimensional fiber predicts. On Flowers the mirror fill is a strong
    distortion baseline because centre-cropped flowers are close to left-right symmetric.

## Plan acceptance criteria that did not hold

Repaint: seam below 1.5 (best 2.43), LPIPS_B below mirror fill (0.627 vs 0.623), and trajectory
angles shrinking to zero (they plateau near 2.4 deg). The reasons are above; none is a code fault
(the Repaint invariants, determinism and pinning, are tested).

## Reproduce

```bash
python scripts/run_repaint.py --stage coarse --n_images 64
python scripts/run_ridgemap.py h1 | h2 | h3 | modes | lm | check
python scripts/run_steer.py --configs '[...]'
python scripts/compile_results.py --manifest manifest.json --out compiled/
```
The exact run directories used are listed in the manifest kept with the results sheet. Runs made
before 2026-10-10 sit under the plan's labels (`s2_*`, `s4_*`, `s1`); `records.read_rows` reads
them under the new names, and reruns under the new names reproduced them exactly.
