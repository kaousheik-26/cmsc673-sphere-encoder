import io
import json
import logging
import math
import os
import random
import zipfile
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)


class ListDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        max_samples=-1,
        load_from_zip=False,
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        json_path = os.path.join(root, f"{split}.json")

        if split == "test" and not os.path.exists(json_path):
            # test split is not available
            json_path = os.path.join(root, "val.json")

        assert os.path.exists(json_path), f"{json_path} does not exist"

        with open(json_path, "r") as f:
            data = [json.loads(line.strip()) for line in f]

        assert len(data) > 0
        assert "class_id" in data[0]
        assert "class_name" in data[0]
        assert "image_path" in data[0]

        self.list = data

        if max_samples > 0:
            self.list = sample_subset(data, max_samples)

        self.load_from_zip = load_from_zip
        self.archives = {}

    def __getitem__(self, index):
        item = self.list[index]

        image_path = str(item["image_path"])
        class_id = int(item["class_id"])
        class_name = str(item["class_name"])

        is_absolute_path = False
        if "is_absolute_path" in item:
            is_absolute_path = item["is_absolute_path"]

        if is_absolute_path:
            absolute_image_path = image_path
        else:
            absolute_image_path = os.path.join(self.root, image_path)

        if self.load_from_zip:

            # lazy loading: open each archive only once per worker,
            # keyed by zip name since the dataset may span multiple zips
            zip_name = image_path.split("/")[0]
            archive = self.archives.get(zip_name)
            if archive is None:
                zip_path = os.path.join(self.root, f"{zip_name}.zip")
                archive = zipfile.ZipFile(zip_path, "r")
                self.archives[zip_name] = archive

            img_bytes = archive.read(image_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img = default_loader(absolute_image_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, class_id, class_name

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return (
            f"root = {self.root}, "
            f"split={self.split}, "
            f"transform={self.transform}"
        )


def get_dataset_cls(dataset_name: str):
    """
    maps a dataset name onto the `Dataset` class that knows how to read it
    """
    if dataset_name in ["cifar-10", "cifar-100"]:
        from torchvision import datasets

        return datasets.__dict__[dataset_name.upper().replace("-", "")]

    # load images from a list of file paths and labels
    return ListDataset


def sample_subset(data: list, max_samples: int, seed: int = 99):
    cnt_per_class = defaultdict(int)

    # get total number of classes
    for item in data:
        class_id = int(item["class_id"])
        cnt_per_class[class_id] = 0

    # shuffle with the identical order on every rank
    random.Random(seed).shuffle(data)

    # estimate number of samples per class
    num_samples_per_class = int(max_samples / len(cnt_per_class))

    # sample
    _list = []
    for item in data:
        class_id = int(item["class_id"])
        if cnt_per_class[class_id] < num_samples_per_class:
            _list.append(item)
            cnt_per_class[class_id] += 1

    logger.info(f"random sampling partial data: {len(_list)}")
    return _list


def create_dataset(
    dataset_cls,
    root,
    split="train",
    concat_train_val_splits=False,
    **kwargs,
):
    """
    creates a dataset regardless of whether it expects `train` or `split`
    """
    import inspect

    sig = inspect.signature(dataset_cls)
    params = sig.parameters

    # drop the options this dataset knows nothing about. classes that take
    # **kwargs (ours) absorb them; torchvision's do not and would raise
    takes_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if not takes_var_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    if "train" in params:
        # expects a boolean flag
        is_train = split.lower() == "train"
        ds = dataset_cls(root=root, train=is_train, **kwargs)

    elif "split" in params:
        # expects a string like 'train' or 'test' or 'val'
        if concat_train_val_splits and split == "train":
            # `max_samples` budgets the concatenation as a whole, not each split:
            # spend half of it on val first, then hand train the budget val could
            # not fill, so a small val split never shrinks the total
            budget = kwargs.get("max_samples", -1)
            val_kwargs = (
                {**kwargs, "max_samples": budget // 2} if budget > 0 else kwargs
            )

            ds_val = dataset_cls(root=root, split="val", **val_kwargs)
            if budget > 0:
                kwargs = {**kwargs, "max_samples": budget - len(ds_val)}

            ds = dataset_cls(root=root, split=split, **kwargs)
            ds = torch.utils.data.ConcatDataset([ds, ds_val])
        else:
            ds = dataset_cls(root=root, split=split, **kwargs)

    else:
        raise TypeError(f"{dataset_cls.__name__} does not accept `train` or `split`")

    return ds


def create_loader(
    dataset_cls,
    dataset_dir,
    image_size,
    patch_size,
    max_samples=-1,
    interp_mode=transforms.InterpolationMode.BICUBIC,
    rot_degrees=0,
    crop_mode="center",
    flip_image=True,
    color_jitter=False,
    extra_padding=False,
    drop_last=True,
    shuffle=True,
    concat_train_val_splits=False,
    ddp_world_size=1,
    ddp_rank=0,
    batch_size_per_rank=16,
    pin_mem=True,
    num_workers=1,
    train_only=False,
    load_from_zip=False,
):
    assert crop_mode in ["random", "center", "random_adm", "center_adm"]

    augs = []  # augmentations

    if crop_mode == "random_adm":
        augs.append(partial(random_crop_arr, image_size=image_size))

    elif crop_mode == "center_adm":
        augs.append(partial(center_crop_arr, image_size=image_size))

    elif crop_mode == "random":
        resize_size = (
            image_size + int(image_size / patch_size / 2)
            if extra_padding
            else image_size
        )
        augs.append(transforms.Resize(resize_size, interpolation=interp_mode))
        augs.append(transforms.RandomCrop(image_size))

    elif crop_mode == "center":
        resize_size = (
            image_size + int(image_size / patch_size / 2)
            if extra_padding
            else image_size
        )
        augs.append(transforms.Resize(resize_size, interpolation=interp_mode))
        if rot_degrees > 0:
            augs.append(
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            degrees=rot_degrees,
                            expand=True,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=0.3,
                )
            )
        augs.append(transforms.CenterCrop(image_size))

    else:
        raise NotImplementedError

    if flip_image:
        augs.append(transforms.RandomHorizontalFlip())

    if color_jitter:
        augs.append(transforms.ColorJitter(0.05, 0.05, 0.05, 0.02))

    aug_mean, aug_std = [0.5] * 3, [0.5] * 3

    train_dataset = create_dataset(
        dataset_cls,
        root=dataset_dir,
        split="train",
        concat_train_val_splits=concat_train_val_splits,
        download=True,
        transform=transforms.Compose(
            [
                *augs,
                transforms.ToTensor(),
                transforms.Normalize(aug_mean, aug_std),
            ]
        ),
        max_samples=max_samples,
        load_from_zip=load_from_zip,
    )
    logger.info(f"number of samples in train dataset: {len(train_dataset)}")
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=shuffle,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_rank,
        sampler=train_sampler,
        pin_memory=pin_mem,
        num_workers=num_workers,
        drop_last=drop_last,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    if train_only:
        return train_loader, train_sampler

    test_dataset = create_dataset(
        dataset_cls,
        root=dataset_dir,
        split="test",
        download=True,
        transform=transforms.Compose(
            [
                partial(center_crop_arr, image_size=image_size),
                transforms.ToTensor(),
                transforms.Normalize(aug_mean, aug_std),
            ]
        ),
    )
    logger.info(f"number of samples in test dataset: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_rank,
        sampler=torch.utils.data.DistributedSampler(
            test_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=False,
        ),
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    vis_loader = DataLoader(
        create_dataset(
            dataset_cls,
            root=dataset_dir,
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    partial(center_crop_arr, image_size=image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(aug_mean, aug_std),
                ]
            ),
        ),
        batch_size=batch_size_per_rank,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(ddp_rank),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    vis_loader = cycle(vis_loader)

    return train_loader, test_loader, vis_loader, train_sampler


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    center cropping implementation from adm
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.Resampling.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    random cropping implementation from adm
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L146
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def resize_arr(pil_image: Image.Image, image_size: int):

    if pil_image.size[0] == image_size and pil_image.size[1] == image_size:
        return pil_image

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.Resampling.BOX,
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )
    return pil_image


def cycle(dataloader):
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            # when the epoch ends, discard the old iterator
            # and create a brand new one. nothing is cached
            iterator = iter(dataloader)
