# Frozen Sphere Encoder intrinsic-property experiments

## Outcome

**The planned depth, normal, and semantic-segmentation experiments are
complete. Their hypotheses have mixed outcomes.** Shared depth transformations
have promising evidence, normal transformations show modest improvement,
and reliable semantic segmentation through a shared decoder edit was not
demonstrated. Geodesic movement did not consistently outperform addition
followed by normalization.

It does not establish a universal intrinsic direction or a complete
inverse-problem restoration method. The results below are measured teacher
agreement, not ground-truth physical or semantic accuracy.


## Data and controls

Existing RGB images were read from ImageNet21k validation data. No input data or checkpoint files were changed. Twelve categories
contributed ten images each: 72 editor-training images, 24 validation images,
and 24 held-out test images. Another twelve categories contributed two images
each to a separate 24-image category-held-out evaluation set. There are
**144 distinct cropped images** in the completed property suite.

Filename and exact cropped-image SHA-256 duplicates are rejected. Near
duplicates were not audited. "Unseen category" means absent from editing
training; overlap with the encoder/decoder's original pretraining data was
not audited. All images remain from the same ImageNet21k source, so this is
not an out-of-domain evaluation.

The Sphere2 base ImageNet 256px checkpoint was loaded strictly using EMA
weights. All encoder and decoder parameters remained frozen, verified by
the absence of parameter gradients. Null class conditioning and decoder
angle zero were fixed throughout. The `[B,256,384]` latents used the global
sphere convention. Each shared edit has one direction and one movement-size
scalar (98,305 trained parameters). No per-image direction or target is used
when predicting with a shared edit.

Each method used three seeds, 400 Adam minibatch steps, and training-only
mean latent-displacement initialization with seeded noise. Validation MSE
selected the RGB-loss edits; validation pooled mIoU selected categorical
segmentation edits. Test sets did not select checkpoints. Four target-informed
per-image references per property were also optimized for 150 steps; these
are diagnostic references, not deployable shared models or guaranteed optima.

## Depth: promising approximate extraction

Targets came from
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf),
RGB images are cropped
before target generation. Predicted relative depth uses per-image 1st/99th
percentile normalization and three identical channels. These values are not
distances in meters.

| Method | Held out image MSE | New-category MSE | New-category correlation |
| --- | ---: | ---: | ---: |
| Training mean depth template | 0.08411 | 0.05311 | 0.7445 |
| Random geodesic directions at learned sizes | 0.22598 | 0.20097 | -0.1305 |
| Learned shared geodesic direction | **0.03951** | **0.02690** | **0.8782** |
| Learned shared normalized addition | **0.03691** | **0.02585** | **0.8813** |
| Encode and reconstruct supplied depth target | — | 0.00675 | 0.9847 |

The saved directions transferred to the new categories **without retraining**.
Geodesic output MSE was `0.02690` against its own targets and `0.06752`
against cyclically shuffled targets. This supports image-specific agreement,
not simply outputting the same template. The four target-informed per-image
new-category references averaged MSE `0.000429`.

The new-category set scored better than the original test set, but it contains
different scenes and may be easier. This does not prove arbitrary-domain
generalization. Outputs still contain residual texture, color, blurred edges,
and geometric mistakes. The original indoor church images were particularly
poor. The original train–test MSE gap remains substantial.

## Surface normals: modest positive evidence

An independent
[Marigold Normals v1.1 teacher](https://huggingface.co/prs-eth/marigold-normals-v1-1)
generated normal vectors.
Settings: fp16, four denoising
steps, ensemble size one, 512px processing, and fixed per-image seeds.
Vectors are unit-normalized and encoded as `(n+1)/2`. Evaluation decodes and
normalizes the vectors, with no axis alignment or fitted rotation. This is
an exploratory teacher protocol, not a benchmark-quality ground-truth run.

| Method | Held-out mean angle error | New-category mean angle error |
| --- | ---: | ---: |
| Constant front-facing vector | 30.83° | 37.86° |
| Constant training mean vector | 30.31° | 36.79° |
| Training spatial mean-normal template | 27.91° | 30.83° |
| Learned shared geodesic direction | **24.08°** | **27.35°** |
| Learned shared normalized addition | **23.29°** | **27.63°** |
| Encode and reconstruct supplied normal target | 35.13° | 49.12° |

The shared direction improves on the spatial template by about 3.48° on
new categories, but the remaining orientation error is large. Only about
20.9% of new-category pixels are within 11.25° for the geodesic method.
The target-informed per-image references averaged 4.04° on four images.
Direct normal-target encoding is a poor route even though per-image decoder
optimization can reach much better outputs. Background/sky orientation can
be ambiguous; no invalid-region ground-truth mask is available here.

## Semantic segmentation: shared decoder editing unsuccessful

Teacher:
[SegFormer B0 ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512). Its 150 label IDs use
fixed unique bit-interleaved RGB codes. Decoder outputs are assigned to the
nearest palette color. Pixel agreement and pooled mIoU are computed against
teacher labels; mIoU includes classes with nonzero union, including spurious
predicted classes. Thus small amounts of many false labels can strongly
reduce mIoU. Both pixel agreement and mIoU are reported.

The first edit objective was palette RGB MSE. A follow-up used cross entropy
over differentiable palette-distance logits at 64px, with a small RGB term,
selected by validation mIoU. The second loss did not solve the problem.

| Method | New-category pixel agreement | New-category mIoU |
| --- | ---: | ---: |
| Training majority-label baseline | 14.58% | 0.00297 |
| Shared geodesic edit, RGB loss | 3.98% | 0.00234 |
| Shared addition edit, RGB loss | 3.50% | 0.00217 |
| Shared geodesic edit, categorical loss | 6.03% | 0.00343 |
| Shared addition edit, categorical loss | 6.44% | 0.00418 |
| Encode and reconstruct supplied palette target | 11.70% | 0.01034 |
| Frozen-latent linear classifier (separate probe) | **39.22%** | **0.08427** |

Shared edits fail to beat the majority baseline on pixel agreement and
have negligible semantic overlap. The palette target is not faithfully
reconstructed by the autoencoder either. A gray or colored region output
therefore cannot be counted as successful segmentation.

The separate probe trained only a 1x1 linear classifier on cached frozen
RGB token latents, with no decoder editing. Three seeds and validation mIoU
selection gave held-out pixel agreement 50.98% / mIoU 0.09762 and new-category
agreement 39.22% / mIoU 0.08427. This is evidence of accessible semantic label
information, not successful segmentation through a shared decoder offset.
The probe is itself limited and is not a high-quality segmentation system.

Per-image target-informed segmentation optimization can achieve high dominant
pixel agreement on four examples, but still introduces many spurious classes
and low mIoU. Alternative target encodings and more expressive transformations
were not tested; the negative result does not prove their impossibility.

## Visual results

The [visual results gallery](assets/README.md) contains 20 saved figures with source folders and evaluation context.

- [Depth on unseen categories](assets/depth-unseen-categories/geodesic_unseen_overview.png)
- [Surface normals on unseen categories](assets/surface-normals/geodesic_unseen_overview.png)
- [Segmentation with RGB loss](assets/segmentation-rgb-loss/geodesic_unseen_overview.png)
- [Segmentation with categorical loss](assets/segmentation-categorical-loss/geodesic_unseen_overview.png)
- [Separate frozen-latent classifier](assets/segmentation-probe/unseen_120.png) (RGB, teacher, prediction)

Per-image controls in the gallery are target-informed optimization references, not shared predictions.
