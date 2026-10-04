## Sphere Encoder 2

### Installation

```bash
pip install -r requirements.txt
```

### Model Zoo

Download checkpoints from [here](https://huggingface.co/tomg-group-umd/sphere2) and place each folder under `workspace/experiments`.

| Model | flowers-256px | flowers-512px | imagenet-256px | imagenet-512px |
| :---- | :-----------: | :-----------: | :------------: | :------------: |
| Base  | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-base-flowers-256px) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-base-flowers-512px) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-base-imagenet-256px) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-base-imagenet-512px) |
| Large | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-large-flowers-256px) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-large-flowers-512px) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-large-imagenet-256px) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-large-imagenet-512px) |

Models trained with $\mathcal{L}_{\mathrm{fd\text{-}lite}}$:

| Model | imagenet-256px | imagenet-512px |
| :---- | :------------: | :------------: |
| Base  | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-base-imagenet-256px-fdlite) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-base-imagenet-512px-fdlite) |
| Large | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-large-imagenet-256px-fdlite) | [ckpt](https://huggingface.co/tomg-group-umd/sphere2/tree/main/sphere2-large-imagenet-512px-fdlite) |

### Sampling

```bash
bash scripts/sample_flowers.sh
```

---

### ImageNet

Organize [ImageNet](https://image-net.org/download) dataset as follows:

```
workspace/datasets/imagenet
├── train.json
├── val.json
└── images
    ├── train
    │   ├── n01440764
    │   └── ...
    └── val
        ├── n01440764
        └── ...
```

Json files can be downloaded from [here](https://huggingface.co/tomg-group-umd/sphere2).

### Training

```bash
bash scripts/train_imagenet_512px.sh
```

### Evaluation