# Novel views with a frozen Sphere Encoder 2

Can a pretrained **Sphere Encoder 2 (SE2)** show an object from a new camera, given only one
photo? SE2's encoder and decoder stay **frozen** in everything below; no network is trained.

- **Route 1: warp-and-fill.** Move the photo to the new camera with monocular depth and let SE2
  fill what the photo cannot see. 5 GSO objects: **15.7 → 17.5 dB PSNR, 0.310 → 0.243 LPIPS**
  from five fixes; 15.8–16.2 dB on real photos (CO3D).
- **Route 2: latent search.** For a new view, search the SE2 latent that the frozen decoder turns
  into that view. Such a latent exists for every view we tried, and it beats SE2 re-drawing the
  true view (LPIPS 0.02–0.03 vs 0.11). We study how the latent has to move, from 30 to 10,000
  objects.

## Route 1: warp-and-fill

Given **one photo of an object** and a set of new cameras, we generate the object's views
from those cameras using SE2 as a generative prior. Geometry comes from a monocular depth
network; SE2 fills what the input photo cannot see.

### Method

```
input photo ──► MoGe-2 depth ──► scale fix ──► 3D points ─┐
                                                          ▼
  for each new camera (15° steps, outward from the input, alternating sides):
     1. splat the point cloud into the camera (z-buffer)  → image with holes
     2. clean the mask, pre-fill holes inside the object hull, white outside
     3. SE2 fills the holes:  x ← D(rotate(F(E(x)), e, α), α)  for α = 75°, 45°, 25°,
        pulling the known (warped) pixels back after every step
     4. depth of the new view (MoGe-2), fused to the rendered depth at the seams,
        and only clean new pixels are added to the point cloud
  optional: fit a 3D Gaussian splat to all generated views → consistent video
```

Details:
- **Depth and scale.** MoGe-2 gives affine depth for the input photo. The metric scale comes
  from the known orbit radius: the geometric mean of a "q90 limb" rule and a "unit-box" rule
  (`orbit_unitbox_scale`). On 30 GSO objects its scale error is 3.6 % (the first q95 rule: 6.4 %).
  For real photos (CO3D) the scale is a scale-only fit to CO3D's MVS depth; without MVS
  depth, the orbit centre is the point all camera axes look at.
- **Warp.** Points are splatted with a z-buffer (`geometry/warp.py`). Pixels that no point
  covers are unknown.
- **Mask hygiene and pre-fill.** Small holes are inpainted; the remaining unknown region is
  pre-filled with nearest-neighbour colour *inside the convex hull of the warped object* and
  white outside it, with an 8 px unknown band around the hull.
- **SE2 projector with data consistency.** SE2 maps an image to a point on a sphere (`F(E(x))`).
  Rotating that point toward a fixed noise point `e` by an angle α and decoding gives a cleaner,
  more natural image; larger α changes more. We use a short schedule α = 75°, 45°, 25° (one step
  each) and after every step put the known warped pixels back (HQS, λ = 1). The same noise `e`
  is shared by all views.
- **Adding new points.** A new pixel joins the point cloud only if it is non-white, lies inside
  the dilated known hull, and passes an edge filter (3×3 relative depth jump > 4 %, 2 px
  erosion). Its depth is MoGe-2 on the generated view, aligned to the rendered depth with a
  global scale + shift plus a smooth (harmonic / Laplace) offset across the seam.
- **Optional 3D splat.** `gsplat` fit to the source + generated views (no view-dependent colour,
  LPIPS term 0.2) for a smooth, 3D-consistent video (`--video`).

### Results

#### Five fixes, one object at a time (5 GSO objects, RTX A4000)

Each object is run on its own: one input photo (azimuth 0°, elevation 30°), 12 target cameras
(±15° … ±90°), compared with the true renders. PSNR / LPIPS (VGG), white background.

| object | first version | final version | 3D splat of our views (held-out cameras inside the arc) |
|---|---|---|---|
| hat | 14.74 / 0.317 | **18.05 / 0.229** | 19.22 dB |
| bell | 13.37 / 0.310 | **17.06 / 0.211** | 17.98 dB |
| shoe | 16.52 / 0.304 | **17.67 / 0.234** | 18.63 dB |
| chicken toy | 18.53 / 0.294 | **19.29 / 0.245** | 20.52 dB |
| toaster | 15.39 / 0.325 | **15.53 / 0.294** | 16.69 dB |
| **mean** | **15.71 / 0.310** | **17.52 / 0.243** | 18.61 dB |

The fixes, in the order they were found:

| fix | problem it solved | effect |
|---|---|---|
| hull-aware pre-fill | plain nearest-neighbour pre-fill smeared colour into the background; SE2 turned that halo into texture, and later views inherited it | LPIPS 0.222 → 0.134 on warped inputs (probe P5) |
| depth scale rule | the first rule left the point cloud 9 % too small | scale error 6.4 % → 3.6 % (30 objects) |
| edge filter | flying pixels at silhouettes became speckles in later views | cleaner silhouettes |
| harmonic depth fusion | visible seams between the depth of different views | smaller seams |
| short schedule 75°/45°/25° | the long schedule (75°×3, 45°×2, 25°×2) drifted from the known pixels | better on 5/5 objects (LPIPS) |

Figures: `results/warp_r1/figs/hat_ring_first_version.jpg` and `hat_ring_after_fixes.jpg`
(hat from −90° to +90°).

#### Same code on RTX A5000, mean of 3 seeds

| object | hat | bell | chicken toy | shoe | toaster | mean |
|---|---|---|---|---|---|---|
| PSNR | 18.43 | 18.94 | 18.49 | 18.63 | 16.16 | **18.13** (LPIPS 0.223) |

The warp-and-fill chain is autoregressive (each view builds on the previous ones), so tiny
numeric differences between GPU types grow: one object can move by up to 1.9 dB between an
A4000 and an A5000 run. Compare variants only on one GPU type and over several seeds.

#### Where the remaining error comes from (oracle tests)

| setting (7 dev objects, first version) | view PSNR | view LPIPS |
|---|---|---|
| MoGe-2 depth everywhere | 15.68 | 0.329 |
| true depth for the input photo only | 16.84 | 0.316 |
| true depth for the generated views only | 17.24 | 0.234 |
| true depth everywhere (with colour match + feathered paste) | 18.97 | 0.212 |

With true depth everywhere the final hat reaches 20.07 dB / 0.167. Depth of the *generated*
views is the largest fixable loss. A 3D splat fitted to the *true* views at the same cameras
reaches 36–43 dB, so the splat itself is not the limit.

#### Real photos (CO3D teddy bears)

| sequence | PSNR | LPIPS | note |
|---|---|---|---|
| 34_1479_4753 | 16.17 | 0.234 | scale from CO3D MVS depth |
| 187_20215_38541 | 15.80 | 0.243 | scale from CO3D MVS depth |
| 598_91820_182418 | 12.80 | 0.354 | no MVS depth; scale from the camera axes |

Five real target frames per sequence (about 15° to 90° away). Figure:
`results/warp_r1/figs/co3d_teddy_real_photos.jpg` (top: real photos, bottom: ours).

#### Probe P5: SE2 on warped inputs (why data consistency is needed)

Repeatedly applying the SE2 projector to a warped image *without* putting the known pixels back
makes the result drift away from the truth at every angle (LPIPS 0.22 → 0.26–0.37 after 10
steps; α = 85° is destructive), even though realism (KID) improves. Hence: strong data
consistency, few projector steps, and large α only where pixels are unknown.

#### What did not help

Colour matching of the generated pixels, feathered pasting of the known pixels, λ = 0.3,
10° steps instead of 15°, a predicted class label instead of the null label, a final
consistency sweep with the splat, latent refinement against the known pixels (100 or 300
steps), more mask erosion (4 or 6 px instead of 2: 17.51 and 16.69 dB vs 18.13), a stricter
non-white threshold (18.17 vs 18.13 dB, within noise).

#### Known failure mode

Past about 45°, the point cloud collects pale rim pixels from earlier filled views, and later
views inherit them (`results/warp_r1/figs/failure_pale_points_past_45deg.jpg`; columns: warp,
pre-fill, output, true view). Quality is best near the input view.

### How to run

Requirements: PyTorch 2.6 (CUDA 12.4), `gsplat` 1.5.3, `nvdiffrast`, MoGe-2, `lpips`,
`torch-fidelity`, `timm`, `opencv-python`, `imageio[ffmpeg]`. The SE2 checkpoint
`sphere2-large-imagenet-fd-lite-256px` goes in `<workspace>/experiments/`.

Paths are set at the top of three files: `DATA` in `se3d/data/render_gso.py` and
`se3d/gs/gt_ceiling.py`, and `DEFAULT_WORKSPACE` in `se3d/se2/wrapper.py`.

Run everything from the repository root.

```bash
# data: GSO30 renders (SyncDreamer cameras, 32-view ring, dev orbit, extrapolated views)
python -m se3d.data.render_gso --protocols sd16 ring32 dev_orbit ext8
python -m se3d.data.align_gso30            # align our meshes to the SyncDreamer renders
python -m se3d.data.prep_co3d_dev           # CO3D teddy-bear dev frames (optional)

# warp-and-fill on dev objects (default settings = final version), one object per run
python se3d/routes/r1_warp_fill.py --objects hat --out results/r1/hat
# with a smooth 3D-splat video (true | our splat | splat of the true views)
python se3d/routes/r1_warp_fill.py --objects hat --gs_sh 0 --gs_lpips 0.2 --gs_iters 4000 --video --out results/r1/hat

# evaluation protocols: SyncDreamer GSO30 cameras, or real CO3D photos
python -m se3d.routes.r1_protocols --protocol gso30 --out results/r1_gso30
python -m se3d.routes.r1_protocols --protocol co3d --out results/r1_co3d
```

Every field of `R1Config` (in `r1_warp_fill.py`) is also a command-line flag, for example
`--schedule 75 45 25`, `--pc_edge_thresh 0.04`, `--pc_erode 2`, `--depth_fusion harmonic`,
`--depth_scale geo_q90_aabb`, `--seed 0`.

Debug probes: `python -m se3d.probes.depth_check --object hat` (MoGe-2 accuracy),
`python -m se3d.probes.scale_check` (scale rules on 30 objects),
`python -m se3d.probes.r1_depth_debug` (per-view depth errors), and
`python -m se3d.probes.p5_projection` (SE2 projector on warped inputs).

SLURM helper: `se3d/scripts/submit.sh NAME HOURS MEM_GB "command" [min_vram_gb]`.

### Files

| path | content |
|---|---|
| `routes/r1_warp_fill.py` | the warp-and-fill pipeline (`R1Config`, `R1`), GSO dev driver |
| `routes/r1_protocols.py` | GSO30 (SyncDreamer cameras) and CO3D real-photo protocols |
| `geometry/` | cameras, mesh renderer (nvdiffrast), z-buffer warp, mask hygiene and pre-fill, MoGe-2 depth and depth fusion, point cloud |
| `se2/wrapper.py` | gradient-capable wrapper around the frozen SE2 encoder / decoder |
| `gs/` | Gaussian-splat fitting (gsplat) and the true-view splat ceiling |
| `data/` | GSO renders, alignment to the SyncDreamer renders, CO3D dev frames |
| `eval/` | PSNR / SSIM / LPIPS / KID, side-by-side videos |
| `probes/` | depth, scale and projector probes used to debug the pipeline |
| `results/warp_r1/` | metrics (JSON) of the runs above and the figures |

## Route 2: latent search

For a source photo and the **true** view from a new camera, we search the SE2 latent that the
frozen decoder turns into that view. This uses the true view, so it is an analysis tool and an
upper bound, not a way to predict new views. It answers two questions: can the frozen decoder
draw new views at all, and how must the latent move when the camera moves?

### Method

- `z0 = F(E(x_src))` is the source photo's latent on the sphere; `z_gt = F(E(x_tgt))` is the
  encoder's latent of the true view.
- We optimise a latent `u` (Adam, lr 1e-2), use `z = spherify(u)`, and minimise
  LPIPS (VGG) + L1 between `D(z)` and the true view. An optional pull `μ · 2(1 − cos(z, z0))`
  keeps the latent close to the source latent. We keep the best step (LPIPS AlexNet, checked
  every 50 steps).
- Start points: `z0`, `z_gt`, a token-warped `z0` (latent tokens moved with depth,
  `probes/p4_token_warp.py`), or random noise.
- Checks: decode the searched latent again after rotating it toward noise by 10° / 20° / 40°,
  and encode its image again (does the encoder map it back?).
- At scale: one source view per object, a fixed set of camera changes, μ = 1, 150 steps
  (enough: LPIPS 0.026 at 150 steps vs 0.019 at 500).

### Results

#### 1. Every new view has a good latent (30 GSO objects × 5 camera turns, 15°–90°, 1,000 steps)

| start point | LPIPS of the searched latent | beats SE2 re-draw | angle to `z0` | angle to `z_gt` | LPIPS after re-decoding at 20° |
|---|---|---|---|---|---|
| source latent `z0` | 0.020 | 150 / 150 | 13.6° | 20.2° | 0.178 |
| true view's latent `z_gt` | 0.019 | 150 / 150 | 19.9° | 11.6° | 0.086 |
| token-warped `z0` | 0.020 | 150 / 150 | 27.0° | 27.6° | 0.157 |
| random noise | 0.023 | 150 / 150 | 90.1° | 90.0° | 0.348 |

SE2 re-drawing the true view (`D(F(E(x_tgt)))`) scores 0.110; `z0` and `z_gt` are 16.0° apart.
With the pull μ = 1 the searched latent stays only 3.5° from `z0` at the same quality (0.018);
μ = 10 gives 0.038 and μ = 100 gives 0.09–0.11.

- The frozen decoder can draw every target view, better than SE2's own re-draw of it.
- Starting from the source latent, the search finds a small nudge; it does not move toward the
  true view's own latent.
- Latents near the encoder's output survive re-decoding best.

Figure: `results/latent_search/figs/search_grid_30_objects.jpg` (rows: source, true view,
SE2 re-draw, searched latent, searched latent re-decoded at 20°).

#### 2. Is a camera change linear in the latent? (100 GSO-300 objects, +17° turn)

Ridge maps fitted on 80 objects, tested on 20 by decoding (LPIPS):

| prediction of the new view's latent | LPIPS |
|---|---|
| source latent (no change) | 0.219 |
| source + mean move | 0.217 |
| ridge map: source latent → searched move | 0.192 |
| ridge map: source latent → move to `z_gt` | 0.215 |
| searched latent (oracle) | 0.020 |
| SE2 re-draw of the true view | 0.131 |

Applying the +17° map twice gives 0.493, while a direct +34° map gives 0.235 (no change: 0.255).
The top 10 principal directions explain 18 % of the searched moves (34 % of the moves to
`z_gt`). There is no useful linear rule.

#### 3. Pattern study on 200 objects (GSO-300 orbits, 11 azimuth turns from ±17° to 171°)

Search from `z0` with μ = 1, 500 steps; LPIPS of the searched latents 0.021–0.024
(SE2 re-draw 0.117–0.121).

- **Size.** The searched latent is 3.4°–4.1° from `z0` for every camera turn (17°: 3.4°,
  85°: 4.0°). The encoder's own latent of the true view moves more and grows with the turn:
  12.5° (17°), 16.4° (34°), 18.8° (51°), 21.3° (85°), 15.4° (171°; back views of symmetric
  objects look alike).
- **Shared direction.** Weak: the mean of the unit moves has length 0.16 (random: 0.07), and two
  objects' moves have a mean cosine of 0.02. The part shared by all objects is the average over
  tokens; it holds about 1.3 % of the move and is the same for every camera turn, so it is not
  a rotation.
- **Sign.** The moves for +d and −d are not opposite (cosine +0.23; +0.32 for the encoder's own
  latents).
- **Where.** About half of the move sits on the object's tokens, which are 26 % of all tokens.
- **Linear tools on unseen objects.** Fitted on 160 objects, tested on 40 others: at 85°, no
  change 0.320, mean move 0.320, ridge 0.314–0.318 (searched latent 0.023).
- Applying the mean move with growing strength never turns the object; at 16× it breaks the
  image (`results/latent_search/figs/average_move_does_not_turn_the_object.jpg`; columns: input,
  strength 0, 1, 4, 16, true view).

#### 4. Searched views as 3D videos (5 GSO objects)

72 cameras on two rings (10° and 40° elevation), search from `z0` with μ = 1 (400 steps), then a
Gaussian splat fitted to the 72 decoded views and rendered on a 120-camera circle between the
rings:

| object | searched views (PSNR / LPIPS) | 3D splat on new cameras (PSNR / LPIPS) |
|---|---|---|
| stacking bear | 32.5 / 0.019 | 34.3 / 0.045 |
| sneaker | 29.5 / 0.021 | 30.7 / 0.048 |
| doll family | 28.4 / 0.023 | 30.3 / 0.054 |
| paint bottles | 24.7 / 0.048 | 25.3 / 0.114 |
| LEGO set | 23.4 / 0.040 | 23.7 / 0.098 |

Figures: `results/latent_search/figs/search_bear_72_cameras.jpg` and
`search_sneaker_72_cameras.jpg` (rows: true view, decoded searched latent).

#### 5. Latent search on 10,000 objects

9,991 Objaverse CC-BY objects (G-Objaverse renders), one source view each, 15 camera changes:
turns of ±15°, ±30°, ±45°, ±60°, ±90°, ±120° and 180° at the same height, plus two views from
lower and closer (same direction, and turned 90°). About 150,000 searches, μ = 1, 150 steps.

| camera change | searched latent | SE2 re-draw of the true view |
|---|---|---|
| ±15° | 0.020 | 0.116 |
| ±45° | 0.027 | 0.113 |
| ±90° | 0.032 | 0.117 |
| ±120° | 0.031 | 0.113 |
| 180° | 0.023 | 0.120 |
| lower and closer, same direction | 0.019 | 0.122 |
| lower and closer, turned 90° | 0.032 | 0.119 |
| **all 15** | **0.027** | **0.115** |

The searched latent beats SE2's re-draw in 99.4 % of the searches. The searched latents
(about 60 GB) are kept outside the repository.

### How to run

Run from the repository root (paths as in Route 1).

```bash
# 1. search on the 30 GSO dev objects, all start points, and the pull sweep
python -m se3d.routes.r2_latent_search --protocol dev_orbit --inits z0 zgt warp random --steps 1000 --out results/r2/inits
python -m se3d.routes.r2_latent_search --protocol dev_orbit --inits z0 --mus 1 10 100 --steps 1000 --out results/r2/mu
python -m se3d.routes.r2_analysis --run results/r2/inits

# 2. linear test on the GSO-300 orbits (render them with: python -m se3d.data.render_gso --gso300)
python -m se3d.routes.r2_latent_search --protocol sv3d21 --inits z0 --deltas 17.142857 --steps 1000 --out results/r2/step1
python -m se3d.routes.r2_latent_search --protocol sv3d21 --inits z0 --deltas 34.285714 --steps 1000 --out results/r2/step2
python -m se3d.routes.r2_linear --run1 results/r2/step1 --run2 results/r2/step2 --out results/r2/linear

# 3. pattern study: searches in folders results/r2_scale200/part*/ (one or more runs of
#    r2_latent_search --protocol sv3d21 --inits z0 --mus 1 --steps 500), then
python -m se3d.routes.r2_pattern --runs results/r2_scale200 --out results/r2_pattern200

# 4. 3D videos of the searched views
python -m se3d.data.render_object_views --object STACKING_BEAR_V04KKgGBn2A --views 64 --out data/object_views
python -m se3d.routes.r2_video --root data/object_views --object STACKING_BEAR_V04KKgGBn2A --out results/r2_video/bear

# 5. search on 10,000 Objaverse objects (one job per part)
python -m se3d.data.prep_objaverse licenses
python -m se3d.data.prep_objaverse download --start 0 --end 10000
ls -d data/objaverse_ccby/* > objaverse_list.txt
python -m se3d.routes.r2_search_scale --list objaverse_list.txt --part 0 --n_parts 40 --out results/r2_scale10k
#    second set of camera changes: add --targets 2 4 8 12 22 20 16 27 30
```

### Files

| path | content |
|---|---|
| `routes/r2_latent_search.py` | the latent search (start points, pull, Riemannian option, re-decoding checks) |
| `routes/r2_analysis.py` | analyses of a search run (fidelity, landing point, token transport, re-decoding) |
| `routes/r2_linear.py` | linear test: PCA of the moves, ridge maps, composition |
| `routes/r2_pattern.py` | pattern study over many objects (size, shared direction, sign, linear tools on unseen objects) |
| `routes/r2_video.py` | search on many cameras, then a Gaussian-splat video |
| `routes/r2_search_scale.py` | large-scale search over thousands of objects, split into parts |
| `se2/sphere_ops.py` | sphere maths: slerp, log / exp maps, tangent projection, angles |
| `probes/p4_token_warp.py` | latent tokens moved with depth (one start point) |
| `data/prep_objaverse.py`, `data/render_object_views.py`, `data/images.py` | Objaverse CC-BY download, many-camera renders of one object, image loading |
| `results/latent_search/` | metrics (JSON) of the runs above and the figures |
