import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms

import sphere.rng as rng
from sphere.loader import ListDataset, center_crop_arr


def image_dataset(name: str, image_size: int, dev_dir: str = "workspace", split: str = "val") -> ListDataset:
    tf = transforms.Compose(
        [
            transforms.Lambda(lambda im: center_crop_arr(im, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return ListDataset(osp.join(dev_dir, "datasets", name), split=split, transform=tf)


def balanced_subset(ds: ListDataset, n: int, seed: int = 0) -> list:
    by_cls = defaultdict(list)
    for i, item in enumerate(ds.list):
        by_cls[int(item["class_id"])].append(i)
    g = np.random.default_rng(rng.fold_in(seed, "subset") % (2**32))
    classes = sorted(by_cls)
    g.shuffle(classes)
    queues = {c: list(g.permutation(by_cls[c])) for c in classes}
    out = []
    while len(out) < n and any(queues.values()):
        for c in classes:
            if queues[c] and len(out) < n:
                out.append(int(queues[c].pop(0)))
    return out


def load_batch(ds: ListDataset, idx: list, device="cuda"):
    xs, cs, ids = [], [], []
    for i in idx:
        x, c, _ = ds[i]
        xs.append(x)
        cs.append(c)
        ids.append(ds.list[i]["image_path"])
    return torch.stack(xs).to(device), torch.tensor(cs, device=device), ids
