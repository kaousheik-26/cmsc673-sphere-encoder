# Warp-and-fill novel views with a frozen Sphere Encoder 2

Given **one photo of an object** and a set of new cameras, we generate the object's views
from those cameras using the pretrained **Sphere Encoder 2 (SE2)** as a generative prior.
SE2's encoder and decoder stay **frozen**; nothing is trained. Geometry comes from a
monocular depth network; SE2 fills what the input photo cannot see.

Main result (5 GSO objects, each run separately, 12 new cameras from ±15° to ±90°):
**15.7 → 17.5 dB PSNR and 0.310 → 0.243 LPIPS** between the first working version and the
final version, from five fixes described below. On real photos (CO3D teddy bears) the same
pipeline gives 15.8–16.2 dB.

## Method

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

## Results

### Five fixes, one object at a time (5 GSO objects, RTX A4000)

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

### Same code on RTX A5000, mean of 3 seeds

| object | hat | bell | chicken toy | shoe | toaster | mean |
|---|---|---|---|---|---|---|
| PSNR | 18.43 | 18.94 | 18.49 | 18.63 | 16.16 | **18.13** (LPIPS 0.223) |

The warp-and-fill chain is autoregressive (each view builds on the previous ones), so tiny
numeric differences between GPU types grow: one object can move by up to 1.9 dB between an
A4000 and an A5000 run. Compare variants only on one GPU type and over several seeds.

### Where the remaining error comes from (oracle tests)

| setting (7 dev objects, first version) | view PSNR | view LPIPS |
|---|---|---|
| MoGe-2 depth everywhere | 15.68 | 0.329 |
| true depth for the input photo only | 16.84 | 0.316 |
| true depth for the generated views only | 17.24 | 0.234 |
| true depth everywhere (with colour match + feathered paste) | 18.97 | 0.212 |

With true depth everywhere the final hat reaches 20.07 dB / 0.167. Depth of the *generated*
views is the largest fixable loss. A 3D splat fitted to the *true* views at the same cameras
reaches 36–43 dB, so the splat itself is not the limit.

### Real photos (CO3D teddy bears)

| sequence | PSNR | LPIPS | note |
|---|---|---|---|
| 34_1479_4753 | 16.17 | 0.234 | scale from CO3D MVS depth |
| 187_20215_38541 | 15.80 | 0.243 | scale from CO3D MVS depth |
| 598_91820_182418 | 12.80 | 0.354 | no MVS depth; scale from the camera axes |

Five real target frames per sequence (about 15° to 90° away). Figure:
`results/warp_r1/figs/co3d_teddy_real_photos.jpg` (top: real photos, bottom: ours).

### Probe P5: SE2 on warped inputs (why data consistency is needed)

Repeatedly applying the SE2 projector to a warped image *without* putting the known pixels back
makes the result drift away from the truth at every angle (LPIPS 0.22 → 0.26–0.37 after 10
steps; α = 85° is destructive), even though realism (KID) improves. Hence: strong data
consistency, few projector steps, and large α only where pixels are unknown.

### What did not help

Colour matching of the generated pixels, feathered pasting of the known pixels, λ = 0.3,
10° steps instead of 15°, a predicted class label instead of the null label, a final
consistency sweep with the splat, latent refinement against the known pixels (100 or 300
steps), more mask erosion (4 or 6 px instead of 2: 17.51 and 16.69 dB vs 18.13), a stricter
non-white threshold (18.17 vs 18.13 dB, within noise).

### Known failure mode

Past about 45°, the point cloud collects pale rim pixels from earlier filled views, and later
views inherit them (`results/warp_r1/figs/failure_pale_points_past_45deg.jpg`; columns: warp,
pre-fill, output, true view). Quality is best near the input view.

## How to run

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

## Files

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
