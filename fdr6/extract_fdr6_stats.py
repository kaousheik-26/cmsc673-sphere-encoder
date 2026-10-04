"""
run:
    ./run.sh fdr6/extract_fdr6_stats.py --dataset_name flowers-102 --image_size 256
    ./run.sh fdr6/extract_fdr6_stats.py --dataset_name imagenet --image_size 512
    ./run.sh fdr6/extract_fdr6_stats.py --dataset_name imagenet --imagenet_ref full
    ./run.sh fdr6/extract_fdr6_stats.py --dataset_name imagenet --imagenet_ref full --encoders inception

extract features with the six FDr-6 encoders on the reference set of a dataset,
then save one (mu, sigma) stats file per encoder. optionally also compute the
held-out normalizer FD(heldout, ref) per encoder (needed to turn raw FD into
FDr) and save it as a json.

the reference set:
    imagenet : --imagenet_ref rand-50k: 50K raw images sampled from the train
               split (fid_refs/ref_images_imagenet_rawpx), source tag "rand-50k"
               --imagenet_ref full (default): the whole 1.28M train split through
               sphere.loader (datasets/imagenet/train.json), source tag "full".
               this is what the FD-Loss paper stats use, so FD(val, ref) lands
               near the paper's normalizers (inception 1.68, dinov2 14.19, ...)
               instead of the ~1.3x inflated 50K-vs-50K values
    others   : the full train+val split through sphere.loader, source tag "extr"

the held-out split is the dataset's "test" split (falls back to val.json).
for imagenet that is the 50K val set, disjoint from the reference. for the
small datasets the reference already concatenates train+val, so the held-out
split overlaps the reference; the normalizer is still written but flagged in
the json.
"""

import argparse
import glob
import logging
import os
import os.path as osp
from functools import partial

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cli_utils import get_device_type, get_dist_backend, set_device, str2bool
from fdr6.fdr6_utils import (
    ENCODER_LABELS,
    autocast_dtype,
    build_encoders,
    extract_multi_stats,
    frechet_distance,
    get_encoder_specs,
    save_stats,
    save_valfd,
)
from sphere.metric import DistributedEvalSampler
from sphere.loader import center_crop_arr, create_dataset, get_dataset_cls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# fmt: off
parser = argparse.ArgumentParser(description="FDr-6 reference statistics")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="fid_stats")
parser.add_argument("--data_dir", type=str, default="datasets")
parser.add_argument("--ref_image_dir", type=str, default="fid_refs")
# --- dataset
parser.add_argument("--dataset_name", type=str, default="flowers-102", choices=["flowers-102", "imagenet"])
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--imagenet_ref", type=str, default="full", choices=["rand-50k", "full"], help="imagenet only: reference set = the 50K random subset shared with FID, or the full 1.28M train split")
parser.add_argument("--load_from_zip", type=str2bool, default=False, help="read the full imagenet train split from zip archives (see sphere.loader.ListDataset)")
parser.add_argument("--batch_size_per_rank", type=int, default=50)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--device", type=str, default=get_device_type())
# --- encoders
parser.add_argument("--encoders", type=str, nargs="+", default=None, choices=ENCODER_LABELS, help="subset of the six encoders to extract (default: all). the valfd json is merged into an existing one, so subsets can be run one at a time")
# --- held-out normalizer
parser.add_argument("--compute_valfd", type=str2bool, default=True)
# --- inception-v3 ckpt
parser.add_argument("--inception_weight_path", type=str, default="workspace/pretrained/fid_pretrained_models/weights-inception-2015-12-05-6726825d.pth")
cli_args = parser.parse_args()
# fmt: on
# -----------------------------------------------------------------------------


class FlatImageFolder(Dataset):
    """images live directly under `folder` (no per-class subdirs)."""

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".JPEG", ".bmp", ".webp")

    def __init__(self, folder, transform):
        self.paths = sorted(
            p for p in glob.glob(osp.join(folder, "*")) if p.endswith(self.IMG_EXTS)
        )
        assert len(self.paths) > 0, f"no images found under {folder}"
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0


def build_transform(image_size):
    # ADM center crop, then [-1, 1]: the input contract of every fdr6 encoder
    return transforms.Compose(
        [
            transforms.Lambda(partial(center_crop_arr, image_size=image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def build_reference_dataset(args):
    """returns (dataset, source_tag)."""
    transform = build_transform(args.image_size)
    if args.dataset_name == "imagenet" and args.imagenet_ref == "full":
        ds = create_dataset(
            get_dataset_cls(args.dataset_name),
            root=osp.join(args.dev_dir, args.data_dir, args.dataset_name),
            split="train",
            concat_train_val_splits=False,
            transform=transform,
            load_from_zip=args.load_from_zip,
        )
        return ds, "full"
    if args.dataset_name == "imagenet":
        raw_dir = osp.join(
            args.dev_dir, args.ref_image_dir, "ref_images_imagenet_rawpx"
        )
        assert osp.exists(raw_dir), (
            f"{raw_dir} not found; --imagenet_ref rand-50k needs the 50K "
            f"reference images there (or use --imagenet_ref full)"
        )
        return FlatImageFolder(raw_dir, transform), "rand-50k"

    dataset_cls = get_dataset_cls(args.dataset_name)
    ds = create_dataset(
        dataset_cls,
        root=osp.join(args.dev_dir, args.data_dir, args.dataset_name),
        split="train",
        concat_train_val_splits=True,
        download=True,
        transform=transform,
    )
    return ds, "extr"


def build_heldout_dataset(args):
    """the dataset's test split (ListDataset falls back to val.json)."""
    dataset_cls = get_dataset_cls(args.dataset_name)
    return create_dataset(
        dataset_cls,
        root=osp.join(args.dev_dir, args.data_dir, args.dataset_name),
        split="test",
        download=True,
        transform=build_transform(args.image_size),
    )


def build_loader(ds, args, ddp_rank, ddp_world_size):
    # Keep every reference/held-out image exactly once across ranks.
    sampler = DistributedEvalSampler(
        ds,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size_per_rank,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def main(args):
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device_type = get_device_type()
    device = set_device(device_type, ddp_local_rank)
    dist.init_process_group(
        backend=get_dist_backend(device_type), device_id=ddp_local_rank
    )
    ddp_rank0 = ddp_rank == 0

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    specs = get_encoder_specs(args.image_size)
    if args.encoders is not None:
        specs = [s for s in specs if s.label in args.encoders]
    if ddp_rank0:
        for s in specs:
            logger.info(f"encoder {s.label}: {s.model_name} @ {s.target_size}")
    encoders = build_encoders(
        specs, device=device, inception_weight_path=args.inception_weight_path
    )

    stats_dir = osp.join(args.dev_dir, args.out_dir)

    # ---- reference stats ----
    ref_ds, source = build_reference_dataset(args)
    logger.info(f"reference set: {len(ref_ds)} images, source tag: {source}")
    ref_loader = build_loader(ref_ds, args, ddp_rank, ddp_world_size)
    ref_stats = extract_multi_stats(
        ref_loader, encoders, device, desc=f"ref {args.dataset_name}"
    )
    if ddp_rank0:
        save_stats(
            ref_stats, specs, stats_dir, source, args.dataset_name, args.image_size
        )
    dist.barrier()

    # ---- held-out normalizer ----
    if args.compute_valfd:
        heldout_ds = build_heldout_dataset(args)
        logger.info(f"held-out set: {len(heldout_ds)} images")
        heldout_loader = build_loader(heldout_ds, args, ddp_rank, ddp_world_size)
        heldout_stats = extract_multi_stats(
            heldout_loader, encoders, device, desc=f"heldout {args.dataset_name}"
        )
        if ddp_rank0:
            valfd = {}
            for s in specs:
                valfd[s.label] = frechet_distance(
                    heldout_stats[s.label]["mu"],
                    heldout_stats[s.label]["sigma"],
                    ref_stats[s.label]["mu"],
                    ref_stats[s.label]["sigma"],
                )
                logger.info(f"valFD {s.label}: {valfd[s.label]:.6f}")
            overlaps_ref = args.dataset_name != "imagenet"
            if overlaps_ref:
                logger.warning(
                    "the held-out split overlaps the reference set (train+val "
                    "concatenated); FDr values for this dataset are only "
                    "comparable to each other, not to the paper"
                )
            save_valfd(
                valfd,
                stats_dir,
                source,
                args.dataset_name,
                args.image_size,
                meta={
                    "encoders": {s.label: [s.model_name, s.target_size] for s in specs},
                    "n_ref": ref_stats[specs[0].label]["n"],
                    "n_heldout": heldout_stats[specs[0].label]["n"],
                    "heldout_overlaps_ref": overlaps_ref,
                    "autocast_dtype": str(autocast_dtype(device)),
                },
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main(cli_args)
