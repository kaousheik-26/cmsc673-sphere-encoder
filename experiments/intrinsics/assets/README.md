# Visual results

Twenty saved figures from the intrinsic-property experiments. Overview grids retain
the examples produced by the evaluation scripts, including imperfect outputs.
Teacher targets are model predictions, not measured ground truth.

Geodesic and addition grids use seed 673. “Unseen” means categories absent from
edit training; test images come from categories used in training.
Per-image depth controls use target information during optimization. The linear
segmentation probe is a separate classifier on frozen latents, not a decoder edit.
Probe columns are RGB input, teacher segmentation, and classifier prediction.

## Figures

### Latent traversals

Source: `outputs/job-7801680/results-smoke-02/`.

- [00_n02123045_10001_traversal.png](latent-traversals/00_n02123045_10001_traversal.png)
- [02_n01443537_10014_traversal.png](latent-traversals/02_n01443537_10014_traversal.png)

### Depth feasibility

Source: `outputs/job-7801755/results-depth-01/`.

- [overview.png](depth-feasibility/overview.png)

### Shared depth

Source: `outputs/job-7801794/results-shared-01/`.

- [geodesic_seed673_overview.png](shared-depth/geodesic_seed673_overview.png)
- [addition_seed673_overview.png](shared-depth/addition_seed673_overview.png)
- [per_image_overview.png](shared-depth/per_image_overview.png)

### Depth unseen categories

Source: `outputs/property-validation/results-property-depth-01/`.

- [geodesic_unseen_overview.png](depth-unseen-categories/geodesic_unseen_overview.png)
- [addition_unseen_overview.png](depth-unseen-categories/addition_unseen_overview.png)

### Surface normals

Source: `outputs/property-validation/results-property-normals-01/`.

- [geodesic_test_overview.png](surface-normals/geodesic_test_overview.png)
- [addition_test_overview.png](surface-normals/addition_test_overview.png)
- [geodesic_unseen_overview.png](surface-normals/geodesic_unseen_overview.png)
- [addition_unseen_overview.png](surface-normals/addition_unseen_overview.png)

### Segmentation rgb loss

Source: `outputs/property-validation/results-property-segmentation-01/`.

- [geodesic_unseen_overview.png](segmentation-rgb-loss/geodesic_unseen_overview.png)
- [addition_unseen_overview.png](segmentation-rgb-loss/addition_unseen_overview.png)

### Segmentation categorical loss

Source: `outputs/property-validation/results-segmentation-categorical-01/`.

- [geodesic_unseen_overview.png](segmentation-categorical-loss/geodesic_unseen_overview.png)
- [addition_unseen_overview.png](segmentation-categorical-loss/addition_unseen_overview.png)

### Segmentation probe

Source: `outputs/property-validation/results-segmentation-probe-01/`.

- [unseen_120.png](segmentation-probe/unseen_120.png)
- [unseen_126.png](segmentation-probe/unseen_126.png)

### Depth per image controls

Source: `outputs/property-validation/results-additional-controls-01/`.

- [depth_reference_120.png](depth-per-image-controls/depth_reference_120.png)
- [depth_reference_126.png](depth-per-image-controls/depth_reference_126.png)
