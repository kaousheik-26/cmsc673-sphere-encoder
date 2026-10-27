# Intrinsic-property experiments

Experiments on extracting relative depth, surface normals, and semantic
segmentation from frozen Sphere Encoder latents. The encoder and decoder stay
fixed; training learns a shared latent direction and movement size, comparing
geodesic movement with addition followed by normalization.

The completed study found promising shared depth transformations, modest normal
improvements, and no reliable shared segmentation edit. Geodesic movement did
not consistently outperform normalized addition. Evaluation measures agreement
with teacher-model predictions, rather than ground-truth accuracy.

## Results

- [Final report](FINAL_REPORT.md): findings, evaluation protocol, and limitations.
- [Depth feasibility results](DEPTH_RESULTS.md): reconstruction of supplied depth targets.
- [Detailed depth evaluation](SHARED_DEPTH_RESULTS.md): experimental setup and controls.

Browse the [visual results gallery](assets/README.md) for 20 comparison figures.

## Setup

Run commands from the repository root. Model experiments require a CUDA GPU,
a pretrained Sphere Encoder checkpoint with its `cfg.json`, and access to input
images. Checkpoints, datasets, and generated tensor caches are not included.

Use a separate environment for inference. The dependencies used in the GPU
experiments are listed in [requirements-inference.txt](requirements-inference.txt);
choose a PyTorch build compatible with your CUDA environment.

```sh
python -m venv .venv-intrinsics
source .venv-intrinsics/bin/activate
python -m pip install -r experiments/intrinsics/requirements-inference.txt
```

Teacher weights are downloaded on first use. Replace the example paths below
with your local paths and choose a new output directory for each run.

## Usage

### Compare latent traversals

Generate image grids comparing geodesic and normalized-addition edits:

```sh
python -m experiments.intrinsics.traverse \
  --checkpoint /path/to/model/ckpt/ep0299.pth \
  --image /path/to/image.jpg \
  --output-dir experiments/intrinsics/outputs/traversal-01 \
  --encoder-from-checkpoint --seed 673
```

`cfg.json` is resolved beside the checkpoint or above `ckpt/`; use `--config`
to override it. `--encoder-from-checkpoint` supports DINOv3 ViT-S/16+ only.
Outputs include comparison grids and metadata with measured angles and latent norms.

### Train a shared depth transformation

Provide a JSON manifest with `image` and `split` fields. Use separate images
for the `train`, `validation`, and `test` splits:

```json
[
  {"image": "/path/to/train.jpg", "split": "train"},
  {"image": "/path/to/validation.jpg", "split": "validation"},
  {"image": "/path/to/test.jpg", "split": "test"}
]
```


The run saves paired tensors, learned transformations, comparison images, and
JSON metrics.

### Additional experiments

| Module | Purpose |
| --- | --- |
| `depth_feasibility` | Test reconstruction of teacher-generated depth targets. |
| `complete_properties` | Evaluate depth on held-out categories and train normal or segmentation edits. |
| `categorical_segmentation` | Train segmentation edits with a class-aware loss. |
| `segmentation_probe` | Train a linear semantic classifier on frozen latents. |
| `audit_controls` | Run constant-normal, random-depth, and per-image depth controls. |

Inspect each module's required inputs with
`python -m experiments.intrinsics.<module> --help`. Later stages consume saved
outputs from earlier experiments. The bundled [pilot12.json](pilot12.json)
contains cluster-specific paths; replace them before using it elsewhere.

## Tests

The geometry and metric tests run without a checkpoint or GPU. Install the test
dependencies from [requirements.txt](requirements.txt) in a separate environment,
then run:

```sh
python -m pytest experiments/intrinsics/test_*.py -q
```
