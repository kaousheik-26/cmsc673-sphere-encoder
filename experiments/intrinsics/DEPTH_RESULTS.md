# Depth feasibility pilot


RGB images are center-cropped to 256px before estimating depth. Predictions
are normalized per image using the 1st and 99th percentiles, clipped to
`[0,1]`, and repeated in three channels. Raw predictions and normalization
bounds are retained. Display normalization keeps the estimator's value
orientation; the values are not distances in meters.

Sphere Encoder checkpoint: base ImageNet 256px, EMA, strict loading, null
class conditioning, decoder angle fixed at zero. The feasibility path is:

**Depth estimator target → frozen encoder → spherical normalization → frozen decoder.**

This explicitly supplies the depth map to Sphere Encoder; it does not ask
Sphere Encoder to infer depth from RGB.

## Measured results

Mean across the 12 images, without fitting scale/shift to reconstruction:

| Method | RGB MSE vs target | Grayscale MSE | Grayscale Pearson correlation |
| --- | ---: | ---: | ---: |
| Depth-map autoencoder reconstruction | 0.005325 | 0.003827 | 0.987227 |
| RGB reconstruction compared with target | 0.259215 | 0.252461 | -0.322951 |
| Grayscale RGB compared with target | 0.252965 | 0.252965 | -0.332012 |
| Per-image target-mean constant | 0.087314 | 0.087314 | Undefined |

Depth-map reconstruction beat the constant baseline on **12/12 images**.
The comparison does not establish shared-direction generalization. The
constant baseline uses the target mean and is only a feasibility reference.
Grayscale metrics average the three channels. RGB MSE includes color errors.
Metrics use unclipped decoder outputs; PNG visualizations clip to `[0,1]`.

The cliff image was the worst depth-map reconstruction (RGB MSE `0.026279`),
with an unwanted blue tint. Mean channel spread across the pilot was
`0.029087`, so reconstruction is not perfectly grayscale. Some contours and
gradients are also altered. The chosen target representation nevertheless
appears usable for a subsequent latent editing experiment.

## Interpretation and next experiment

The frozen model can approximately reconstruct this grayscale pseudo-depth
encoding. This resolves a preliminary target-compatibility question. It
does not show that RGB latents contain an accessible depth direction, that
the teacher is geometrically correct, or that edits will generalize.

## Visual results

[Full depth feasibility grid](assets/depth-feasibility/overview.png) compares supplied teacher depth targets with their reconstructions.
