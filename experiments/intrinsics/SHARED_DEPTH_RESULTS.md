# Shared RGB-to-depth transformations

**Finding: preliminary partial success for a shared depth transformation;
no demonstrated advantage for geodesic movement over normalized addition.**


## What was tested

Starting from **RGB** images, encode and normalize their latents, apply one
shared learned transformation, and decode toward corresponding pseudo-depth
maps. At inference the shared transformation receives only the RGB latent.
Each trained direction applies to every test image; there is no per-image
fitting in the shared-method results.

- Model: Sphere Encoder 2 base ImageNet 256px, strict EMA loading.
- Labels: null class conditioning; decoder angle fixed at zero.
- Latent: global sphere, `[B,256,384]`.
- Editing parameters: one direction tensor plus one size scalar, 98,305
  parameters; encoder and decoder frozen.
- Data: 120 images in 12 categories; 72 train / 24 validation / 24 test.
  Each category contributes 6/2/2 images. The test images are unseen by the
  edit optimizer, but the categories are represented in training.
- File paths and exact cropped-image hashes are disjoint. Near duplicates
  were not audited, and this is not unseen-category evaluation.
- Targets: relative-depth predictions from Depth Anything V2 Small. Normalize per image at
  percentiles 1 and 99 and repeat in three channels. These are teacher
  predictions, not measured physical depth.
- Two methods, three seeds (673/674/675), 400 minibatch steps each.
  Pixel MSE objective; batch size 4; Adam. The initial direction is the
  mean RGB-to-target latent difference over training images, plus seeded
  noise. This initialization also uses only training data.
- Validation MSE selects the best parameters at 50-step intervals; test
  images do not select parameters. Each seed is retained and reported.

## Held-out results

MSE is measured against normalized three-channel depth targets; lower is
better. Correlation uses channel-averaged images; higher is better. Metrics
use unclipped decoder outputs.

Seed SD is variation over three runs, not a
confidence interval for the population of images.

| Method | Test MSE (mean plus/minus seed SD) | Grayscale correlation |
| --- | ---: | ---: |
| Unedited RGB latent | 0.239226 | -0.314815 |
| Random spherical directions at learned sizes | 0.225977 | — |
| Training mean-depth template | 0.084110 | — |
| Per-image target-mean constant (oracle baseline) | 0.092520 | Undefined |
| Learned shared geodesic direction | **0.039513 ± 0.003921** | **0.766986** |
| Learned shared normalized-addition direction | **0.036905 ± 0.002874** | **0.782416** |

The learned spherical edit reduces MSE by about **53%** relative to the
training mean-depth template.

Normalized addition has lower average MSE in this experiment. An exploratory
bootstrap over the twelve categories gives a 95% interval of approximately
`[-0.000178, 0.005861]` for geodesic MSE minus addition MSE. This includes
zero and does not establish a reliable difference. Both methods satisfy the
sphere constraint; addition is followed by normalization.

| Method | Seed | Selected step | Validation MSE | Test MSE |
| --- | ---: | ---: | ---: | ---: |
| Geodesic | 673 | 200 | 0.027365 | 0.043712 |
| Geodesic | 674 | 400 | 0.022891 | 0.035948 |
| Geodesic | 675 | 400 | 0.025650 | 0.038880 |
| Addition | 673 | 400 | 0.027807 | 0.040223 |
| Addition | 674 | 350 | 0.024309 | 0.035164 |
| Addition | 675 | 400 | 0.027045 | 0.035329 |

Learned geodesic angles were 31.54–32.23 degrees. Mean relative-radius error
on test images was below `4.6e-8` in all three geodesic runs. Addition scales
were 0.849–0.875; these are offset scales, not angles.

## Visual findings and limitations

Held-out output grids show approximate foreground shapes, silhouettes, and
background-depth gradients. They also show residual image texture, some
color, blurred boundaries, and geometric errors. The church-interior images
are a notable failure: the output does not recover the target's ceiling
depth pattern. This is **not** reliable depth estimation for every scene.

Geodesic training MSE averaged `0.006430`, while test MSE averaged `0.039513`.
There is substantial overfitting / generalization gap. The small dataset,
same-category split, one checkpoint, pseudo targets, and fixed conditioning
limit the finding. Other initializations, categories, checkpoints, and target
encodings were not tested. This is not a universal direction claim.

Four test images received separate, target-informed per-image optimization
for 150 steps as a reference. Their mean MSE was `0.000447`, substantially
lower than the shared methods. This reference uses each test target and
starts from its encoded-target direction; it is not a deployable RGB-only
predictor or a guaranteed optimum. It demonstrates considerable remaining
capacity when sharing is removed.

## Visual results

- [Shared geodesic edit](assets/shared-depth/geodesic_seed673_overview.png)
- [Shared normalized addition](assets/shared-depth/addition_seed673_overview.png)
- [Target-informed per-image references](assets/shared-depth/per_image_overview.png)
- [Geodesic transfer to unseen categories](assets/depth-unseen-categories/geodesic_unseen_overview.png)
- [Addition transfer to unseen categories](assets/depth-unseen-categories/addition_unseen_overview.png)
