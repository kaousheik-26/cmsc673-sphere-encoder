import argparse
import datetime
import glob
import json
import logging
import os
import os.path as osp
import random
import shutil
import string
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch_fidelity
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import sphere.rng as rng
from cli_utils import get_device_type, get_dist_backend, set_device, str2bool, cfg_sweep
from fdr6.fdr6_utils import (
    ENCODER_LABELS,
    ISC_LOGITS_KEY,
    build_encoders,
    compute_fdr6,
    extract_multi_stats,
    fdr6_table_columns,
    get_encoder_specs,
    load_ref_stats,
    load_valfd,
)
from sphere.builder import build_model
from sphere.ema import ModuleEMA
from sphere.loader import create_dataset, resize_arr
from sphere.utils import load_ckpt, save_image, save_tensors_to_images
from sphere.metric import DistributedEvalSampler, compute_isc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlatImageFolder(torch.utils.data.Dataset):
    """images live directly under `root` (no per-class subdirs)."""

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".JPEG", ".bmp", ".webp")

    def __init__(self, root, transform=None):
        self.paths = sorted(
            p for p in glob.glob(osp.join(root, "*")) if p.endswith(self.IMG_EXTS)
        )
        assert len(self.paths) > 0, f"no images found under {root}"
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = datasets.folder.pil_loader(self.paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


# -----------------------------------------------------------------------------
# fmt: off
parser = argparse.ArgumentParser(description="G1 FDr-6 Evaluation")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="evaluation")
parser.add_argument("--data_dir", type=str, default="datasets")
parser.add_argument("--job_dir", type=str, default=None)
# --- score an existing folder instead of generating
parser.add_argument("--eval_imgs_dir", type=str, default=None)
# --- generation
parser.add_argument("--load_ckpt_strict", type=str2bool, default=True)
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--num_eval_samples", type=int, default=50 * 1000)
parser.add_argument("--batch_size_per_rank", type=int, default=25)
parser.add_argument("--metric_batch_size_per_rank", type=int, default=None, help="batch size per rank for the six judge encoders over the saved images; defaults to --batch_size_per_rank (lower it for 512px where the ConvNeXt judge OOMs on 11GB cards)")
parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers per rank for the metric loaders; each rank forks this many processes, so keep it low on nodes with many ranks (host RAM, not GPU, is the limit)")
parser.add_argument("--forward_steps", type=int, nargs="+", default=[1, 4])
parser.add_argument("--cache_sampling_noise", type=str2bool, default=True)
parser.add_argument("--sampling_init_angle", type=float, default=85.0)
parser.add_argument("--sampling_loop_angle", type=float, default=None, help="constant angle of the sampling loop after the first step; defaults to --sampling_init_angle")
parser.add_argument("--seed_sampling", type=str2bool, default=False)
parser.add_argument("--use_ema_model", type=str2bool, default=False)
parser.add_argument("--compile_model", type=str2bool, default=True)
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
# --- guidance
parser.add_argument("--use_cfg", type=str2bool, default=False)
parser.add_argument("--cfg_min", type=float, default=0.0)
parser.add_argument("--cfg_max", type=float, default=5.0)
parser.add_argument("--cfg_gap", type=float, default=1.0)
parser.add_argument("--cfg_position", type=str, default="angle", choices=["angle"])
# --- metrics
parser.add_argument("--fid_stats_used_from", type=str, default="full", choices=["jit", "adm", "extr", "rand-50k", "full"], help="source tag of the FDr-6 reference stats; full = the whole 1.28M imagenet train split; missing stats cause an error")
parser.add_argument("--fid_stats_dir", type=str, default="fid_stats")
parser.add_argument("--fid_ref_dir", type=str, default="fid_refs")
parser.add_argument("--report_fid", type=str, nargs="+", default=["gfid"])
parser.add_argument("--report_precision_recall", type=str2bool, default=True, help="precision / recall / f-score via torch_fidelity against the reference image folder")
parser.add_argument("--inception_weight_path", type=str, default="workspace/pretrained/fid_pretrained_models/weights-inception-2015-12-05-6726825d.pth")
# --- flops
parser.add_argument("--report_flops", type=str2bool, default=False, help="measure the GFLOPs of one generated image (batch size 1, no cfg) with fvcore and exit; no checkpoint or reference stats needed")
parser.add_argument("--flops_steps", type=int, default=1, help="sampling steps the FLOPs are measured over")
# --- saving
parser.add_argument("--save_grid_images", type=str2bool, default=True)
parser.add_argument("--num_snapshot_samples", type=int, default=256)
parser.add_argument("--rm_folder_after_eval", type=str2bool, default=True)
parser.add_argument("--save_folder_suffix", type=str, default="")
cli_args = parser.parse_args()
if cli_args.sampling_loop_angle is None:
    cli_args.sampling_loop_angle = cli_args.sampling_init_angle
if cli_args.metric_batch_size_per_rank is None:
    cli_args.metric_batch_size_per_rank = cli_args.batch_size_per_rank
# fmt: on
# -----------------------------------------------------------------------------


def main(cli_args):
    exp_dir = osp.join(cli_args.dev_dir, "experiments", cli_args.job_dir)

    # merge cfg.json with cli overrides
    logger.info(f"load cfg from {exp_dir}")
    with open(osp.join(exp_dir, "cfg.json"), "r") as fio:
        cfg_args = json.load(fio)
    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device_type = get_device_type()
    device = set_device(device_type, ddp_local_rank)
    dist.init_process_group(
        backend=get_dist_backend(device_type),
        # device_id must be an accelerator; leave it unset for cpu (gloo) runs
        device_id=device if device_type != "cpu" else None,
        timeout=datetime.timedelta(hours=2),
    )
    ddp_rank0 = ddp_rank == 0
    seed_offset = ddp_rank

    seed = None
    if args.seed_sampling:
        seed = 99
        torch.manual_seed(seed + seed_offset)

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    # ---- flops only: no checkpoint, reference stats or judges needed ----
    # every rank must take this branch: rank 0 alone would leave the others in
    # the eval path, and tearing down the process group under them deadlocks
    # the barriers inside load_ckpt.
    if args.report_flops:
        if ddp_rank0:
            model = build_model(args, for_train=False)
            model.to(dtype=ptdtype, device=device, memory_format=torch.channels_last)
            model.eval().requires_grad_(False)
            run_to_measure_flops(
                model, steps=args.flops_steps, device=device, ptdtype=ptdtype
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    out_dir = osp.join(args.dev_dir, args.out_dir, args.job_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"output dir for images: {out_dir}")

    tabl_dir = osp.join(
        exp_dir,
        "eval" if args.save_folder_suffix == "" else f"eval_{args.save_folder_suffix}",
    )
    os.makedirs(tabl_dir, exist_ok=True)
    logger.info(f"output dir for tables: {tabl_dir}")

    if args.dataset_name in ["flowers-102"]:
        args.fid_stats_used_from = "extr"
        logger.info(
            f"override fid_stats_used_from to {args.fid_stats_used_from} for {args.dataset_name}"
        )

    # ---- FDr-6 reference stats + normalizer ----
    specs = get_encoder_specs(args.image_size)
    stats_dir = osp.join(args.dev_dir, args.fid_stats_dir)
    ref_stats = load_ref_stats(
        specs, stats_dir, args.fid_stats_used_from, args.dataset_name, args.image_size
    )
    valfd = load_valfd(
        stats_dir, args.fid_stats_used_from, args.dataset_name, args.image_size
    )
    if valfd is None:
        logger.warning("no valfd normalizer found; the table will report raw FD only")
    else:
        logger.info(f"valfd normalizer: {valfd}")
    encoders = build_encoders(
        specs, device=device, inception_weight_path=args.inception_weight_path
    )

    metric_kwargs = dict(
        dataset_name=args.dataset_name,
        image_size=args.image_size,
        batch_size_per_rank=args.metric_batch_size_per_rank,
        tabl_dir=tabl_dir,
        encoders=encoders,
        ref_stats=ref_stats,
        valfd=valfd,
        fid_stats_used_from=args.fid_stats_used_from,
        fid_ref_dir=osp.join(args.dev_dir, args.fid_ref_dir),
        report_prc=args.report_precision_recall,
        device=device,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    # ---- score an existing folder and exit ----
    if args.eval_imgs_dir is not None:
        assert osp.isdir(args.eval_imgs_dir), f"not a dir: {args.eval_imgs_dir}"
        gen_imgs_dir = args.eval_imgs_dir
        parsed = parse_folder_name(args.eval_imgs_dir)
        calc_metrics(
            task_mode="generation",
            num_eval_samples=None,
            gen_imgs_dir=gen_imgs_dir,
            num_workers=args.num_workers,
            ckpt_epoch=parsed.get("pth", "external"),
            use_ema=parsed.get("ema", args.use_ema_model),
            forward_steps=parsed.get("steps", "-"),
            cfg=parsed.get("cfg", "-"),
            cfg_position=parsed.get("cfg_position", "-"),
            cache_sampling_noise=parsed.get("cache", "-"),
            sampling_init_angle=parsed.get("init", "-"),
            sampling_loop_angle=parsed.get("loop", parsed.get("amax", "-")),
            seed_sampling=args.seed_sampling,
            **metric_kwargs,
        )
        dist.barrier()
        dist.destroy_process_group()
        return

    snapshot_save_dir = None
    if args.save_grid_images:
        snapshot_save_dir = osp.join(
            exp_dir,
            (
                "eval_snapshot"
                if args.save_folder_suffix == ""
                else f"eval_snapshot_{args.save_folder_suffix}"
            ),
        )
        os.makedirs(snapshot_save_dir, exist_ok=True)
        logger.info(f"output snapshot dir: {snapshot_save_dir}")

    # ---- loader for the reconstruction task ----
    loader = None
    if "rfid" in args.report_fid:
        Ts = transforms.Compose(
            [
                transforms.Lambda(partial(resize_arr, image_size=args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if args.dataset_name in ["cifar-10", "cifar-100"]:
            dataset_cls = datasets.__dict__[args.dataset_name.upper().replace("-", "")]
            ds = create_dataset(
                dataset_cls,
                root=osp.join(args.dev_dir, args.data_dir, args.dataset_name),
                split="train",
                download=True,
                transform=Ts,
            )
        else:
            ref_imgs_dir = osp.join(
                args.dev_dir,
                args.fid_ref_dir,
                f"ref_images_{args.dataset_name}_{args.image_size}px",
            )
            assert osp.exists(
                ref_imgs_dir
            ), f"reference images not found: {ref_imgs_dir}"
            ds = FlatImageFolder(root=ref_imgs_dir, transform=Ts)
        sampler = DistributedEvalSampler(
            ds,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size_per_rank,
            sampler=sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    # ---- model ----
    model = build_model(args, for_train=False)
    model.to(dtype=ptdtype, device=device, memory_format=torch.channels_last)

    ema_model = ModuleEMA(model)
    ema_model.eval().requires_grad_(False)

    ckpt_dir = osp.join(exp_dir, "ckpt")
    ckpts = sorted(glob.glob(osp.join(ckpt_dir, "*.pth")))
    if not ckpts:
        raise ValueError("no checkpoints to eval")
    load_from = (
        ckpts[-1] if args.ckpt_fname is None else osp.join(ckpt_dir, args.ckpt_fname)
    )
    assert osp.exists(load_from), f"ckpt not found: {load_from}"
    logger.info(f"load checkpoint from: {load_from}")
    ckpt_epoch = osp.basename(load_from).replace(".pth", "")

    load_ckpt(
        model,
        ckpt_path=load_from,
        ema_model=ema_model,
        strict=args.load_ckpt_strict,
        override_model_with_ema=args.use_ema_model,
        verbose=True,
    )
    if args.compile_model:
        model = torch.compile(model)
        logger.info("model is compiled")
    model.eval().requires_grad_(False)

    eval_kwargs = dict(
        model=model,
        loader=loader,
        use_ema=args.use_ema_model,
        num_classes=args.num_classes if args.cond_generator else 0,
        save_dir=out_dir,
        ckpt_epoch=ckpt_epoch,
        save_snapshot=args.save_grid_images,
        num_snapshot_samples=args.num_snapshot_samples,
        snapshot_save_dir=snapshot_save_dir,
        seed_sampling=args.seed_sampling,
        seed=seed,
        ptdtype=ptdtype,
        metric_kwargs=metric_kwargs,
    )

    if "rfid" in args.report_fid:
        evaluate(args, task_mode="reconstruction", **eval_kwargs)
        dist.barrier()

    if "gfid" in args.report_fid:
        cfg_vals = cfg_sweep(args.cfg_min, args.cfg_max, args.cfg_gap)
        if not args.use_cfg:
            # guidance off: 0 degrees of angle cfg
            cfg_vals = [0.0]

        for cfg in cfg_vals:
            for step in args.forward_steps:
                evaluate(
                    args,
                    task_mode="generation",
                    forward_steps=step,
                    cfg=cfg,
                    cfg_position=args.cfg_position,
                    cache_sampling_noise=args.cache_sampling_noise,
                    sampling_init_angle=args.sampling_init_angle,
                    sampling_loop_angle=args.sampling_loop_angle,
                    **eval_kwargs,
                )

    dist.destroy_process_group()


def parse_folder_name(path):
    name = osp.basename(osp.normpath(path))
    if name in ("gens", "recs"):
        name = osp.basename(osp.dirname(osp.normpath(path)))
    out = {}
    for tok in name.split("_"):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "cfg" and "-" in v:
            cfg, pos = v.split("-", 1)
            out["cfg"], out["cfg_position"] = cfg, pos
        else:
            out[k] = v
    return out


@torch.inference_mode()
def evaluate(
    args,
    task_mode,
    model,
    loader=None,
    use_ema=False,
    forward_steps=1,
    num_classes=0,
    cache_sampling_noise=False,
    sampling_init_angle=89.0,
    sampling_loop_angle=89.0,
    cfg=1.0,
    cfg_position="angle",
    save_dir=None,
    ckpt_epoch=None,
    save_snapshot=False,
    num_snapshot_samples=128,
    snapshot_save_dir=None,
    seed_sampling=False,
    seed=99,
    ptdtype=torch.bfloat16,
    metric_kwargs=None,
):
    assert task_mode in ["generation", "reconstruction"]
    device = metric_kwargs["device"]
    ddp_rank = metric_kwargs["ddp_rank"]
    ddp_world_size = metric_kwargs["ddp_world_size"]
    image_size = metric_kwargs["image_size"]
    dataset_name = metric_kwargs["dataset_name"]

    is_rec = task_mode == "reconstruction"
    sub_fold_name = "recs" if is_rec else "gens"
    suffix_key = "_rec.png" if is_rec else "_gen.png"
    icon = "🥨" if is_rec else "🍺"

    save_fold_name = (
        f"imgs"
        f"_px={image_size}"
        f"_pth={ckpt_epoch}"
        f"_ema={use_ema}"
        f"_cfg={cfg}-{cfg_position}"
        f"_steps={forward_steps}"
        f"_cache={cache_sampling_noise}"
        f"_init={sampling_init_angle}"
        f"_loop={sampling_loop_angle}"
    )
    if is_rec:
        rand_tag = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        tag_holder = [rand_tag]
        dist.broadcast_object_list(tag_holder, src=0)
        save_fold_name += f"_rec{tag_holder[0]}"

    if save_snapshot:
        snapshot_root = snapshot_save_dir if snapshot_save_dir else save_dir
        snapshot_img_path = osp.join(snapshot_root, save_fold_name + suffix_key)

    save_dir = osp.join(save_dir, save_fold_name)
    gen_imgs_dir = osp.join(save_dir, sub_fold_name)
    if ddp_rank == 0:
        if osp.exists(gen_imgs_dir):
            shutil.rmtree(gen_imgs_dir, ignore_errors=True)
        os.makedirs(gen_imgs_dir, exist_ok=True)
    dist.barrier()
    logger.info(f"save output images to: {gen_imgs_dir}")

    if is_rec:
        rec_iter = iter(loader)
        num_batches_per_rank = len(loader)
    else:
        assert (
            args.num_eval_samples % (ddp_world_size * args.batch_size_per_rank) == 0
        ), (
            f"got num_eval_samples={args.num_eval_samples}, "
            f"world_size={ddp_world_size}, "
            f"batch_size_per_rank={args.batch_size_per_rank}"
        )
        num_batches_per_rank = int(
            args.num_eval_samples / ddp_world_size / args.batch_size_per_rank
        )

    class_ids = None
    if num_classes > 0:
        assert args.num_eval_samples % num_classes == 0
        assert model.use_modulation is True
        per_class = args.num_eval_samples // num_classes
        class_ids = np.arange(0, num_classes).repeat(per_class)
        logger.info(f"total classes: {len(class_ids)}, samples per class: {per_class}")

    dist.barrier()
    logger.info(f"start eval for {task_mode} on {dataset_name} {icon}")

    cnt = 0
    clss = None
    device_type = device if isinstance(device, str) else device.type
    pbar = tqdm(range(num_batches_per_rank), total=num_batches_per_rank)
    for batch_idx in pbar:
        start_idx = (
            batch_idx * args.batch_size_per_rank * ddp_world_size
            + ddp_rank * args.batch_size_per_rank
        )
        end_idx = start_idx + args.batch_size_per_rank

        with torch.autocast(device_type=device_type, dtype=ptdtype):
            if is_rec:
                imgs = next(rec_iter)[0].to(device)
                if num_classes > 0:
                    clss = torch.full(
                        (imgs.shape[0],), num_classes, dtype=torch.long, device=device
                    )
                outs = model.reconstruct(imgs, clss, sampling=False)
            else:
                with (
                    torch.random.fork_rng(devices=[device])
                    if seed_sampling
                    else nullcontext()
                ):
                    if seed_sampling:
                        torch.manual_seed(rng.fold_in(seed, ddp_rank, batch_idx))
                    if num_classes > 0:
                        clss = torch.tensor(class_ids[start_idx:end_idx]).to(
                            device=device, dtype=torch.long
                        )
                    _, outs = model.generate(
                        batch_size=args.batch_size_per_rank,
                        y=clss,
                        cfg=cfg,
                        cfg_position=cfg_position,
                        forward_steps=forward_steps,
                        cache_sampling_noise=cache_sampling_noise,
                        sampling_init_angle=sampling_init_angle,
                        sampling_loop_angle=sampling_loop_angle,
                        device=device,
                    )

        cnt += outs.shape[0]
        pbar.set_description(f"{task_mode}: generated {cnt} images on rank {ddp_rank}")
        save_image(
            x=outs,
            batch_idx=batch_idx,
            ddp_rank=ddp_rank,
            save_dir=gen_imgs_dir,
            force_image_size=args.image_size,
        )
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "xpu":
            torch.xpu.empty_cache()

    dist.barrier()

    calc_metrics(
        task_mode=task_mode,
        num_eval_samples=args.num_eval_samples,
        gen_imgs_dir=gen_imgs_dir,
        num_workers=args.num_workers,
        ckpt_epoch=ckpt_epoch,
        forward_steps=forward_steps,
        seed_sampling=seed_sampling,
        cache_sampling_noise=cache_sampling_noise,
        use_ema=use_ema,
        sampling_init_angle=sampling_init_angle,
        sampling_loop_angle=sampling_loop_angle,
        cfg=cfg,
        cfg_position=cfg_position,
        **metric_kwargs,
    )
    dist.barrier()

    if save_snapshot and ddp_rank == 0:
        imgs = glob.glob(osp.join(gen_imgs_dir, "*.png"))
        if len(imgs) > 0:
            imgs = np.random.choice(
                imgs, min(len(imgs), num_snapshot_samples), replace=False
            )
            imgs = [datasets.folder.pil_loader(img) for img in imgs]
            imgs = [torch.from_numpy(np.array(img)) for img in imgs]
            imgs = torch.stack(imgs, dim=0).permute(0, 3, 1, 2) / 255.0
            save_tensors_to_images(
                imgs,
                path=snapshot_img_path,
                nrow=max(8, int(num_snapshot_samples / 128 * 8)),
                max_nimgs=num_snapshot_samples,
            )
            logger.info(f"save snapshot image to {snapshot_img_path}")
    dist.barrier()

    if args.rm_folder_after_eval:
        shutil.rmtree(gen_imgs_dir, ignore_errors=True)
        logger.info(f"removed generated images folder: {gen_imgs_dir}")
    dist.barrier()


def calc_metrics(
    task_mode,
    dataset_name,
    image_size,
    num_eval_samples,
    batch_size_per_rank,
    ckpt_epoch,
    gen_imgs_dir,
    tabl_dir,
    encoders,
    ref_stats,
    valfd,
    fid_stats_used_from,
    fid_ref_dir=None,
    report_prc=False,
    sampling_init_angle=89.0,
    sampling_loop_angle=90.0,
    use_ema=False,
    forward_steps=1,
    seed_sampling=False,
    cache_sampling_noise=False,
    cfg=1.0,
    cfg_position="angle",
    device=get_device_type(),
    ddp_rank=0,
    ddp_world_size=1,
    num_workers=4,
):
    """six-encoder feature extraction over the generated images (all ranks),
    then FDr-6 and precision / recall / f-score on rank 0, appended to the
    table file."""
    ddp_rank0 = ddp_rank == 0

    # saved PNGs are uint8; ToTensor + Normalize maps them to [-1, 1]
    Ts = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ds = FlatImageFolder(root=gen_imgs_dir, transform=Ts)
    num_imgs = len(ds)
    if task_mode == "generation" and num_eval_samples is not None:
        assert num_imgs == num_eval_samples, f"{num_imgs} != {num_eval_samples}"

    # Score every image exactly once, including an uneven final shard
    sampler = (
        DistributedEvalSampler(
            ds,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
        )
        if ddp_world_size > 1
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size_per_rank,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    gen_stats = extract_multi_stats(
        loader, encoders, device, desc="fdr6 features", collect_isc_logits=True
    )
    if not ddp_rank0:
        return
    n_scored = gen_stats[ENCODER_LABELS[0]]["n"]
    logger.info(f"total number of images to eval: {num_imgs} (scored: {n_scored})")

    metrics = compute_fdr6(gen_stats, ref_stats, valfd)
    for k in fdr6_table_columns():
        logger.info(f"{k}: {metrics[k]}")

    # ---- inception score from the same inception pass ----
    isc_mean, isc_std = compute_isc(torch.from_numpy(gen_stats[ISC_LOGITS_KEY]))
    logger.info(f"isc_mean: {isc_mean}, isc_std: {isc_std}")

    # ---- precision / recall / f-score via torch_fidelity ----
    report_prc = fid_ref_dir is not None and report_prc
    if report_prc:
        ref_imgs_dir = osp.join(
            fid_ref_dir, f"ref_images_{dataset_name}_{image_size}px"
        )
        if not osp.exists(ref_imgs_dir):
            report_prc = False
            logger.warning(
                f"reference images not found: {ref_imgs_dir}, "
                f"skip report precision/recall/f-score"
            )

    prc, rcl, fsc = "-", "-", "-"  # N/A
    if report_prc:
        logger.info("start calculating P/R/F-Score")
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=gen_imgs_dir,
            input2=ref_imgs_dir,
            # torch_fidelity only supports a cuda/cpu switch (no xpu), so this
            # falls back to its CPU path when the run isn't on a CUDA device.
            cuda=(device if isinstance(device, str) else device.type) == "cuda",
            isc=False,
            fid=False,
            kid=False,
            prc=True,
            verbose=True,
        )
        prc = metrics_dict["precision"]
        rcl = metrics_dict["recall"]
        fsc = metrics_dict["f_score"]
        logger.info(f"precision: {prc}, recall: {rcl}, f-score: {fsc}")

    metric_cols = fdr6_table_columns()
    headers = metric_cols + [
        "isc_mean",
        "isc_std",
        "forward_steps",
        "cfg",
        "cfg_position",
        "image_size",
        "num_imgs",
        "sampling_init_angle",
        "sampling_loop_angle",
        "task_mode",
        "use_ema",
        "seed_sampling",
        "cache_sampling_noise",
        "fid_stats_used_from",
    ]
    row = [metrics[k] for k in metric_cols] + [
        isc_mean,
        isc_std,
        forward_steps,
        cfg,
        cfg_position,
        image_size,
        n_scored,
        sampling_init_angle,
        sampling_loop_angle,
        task_mode,
        use_ema,
        seed_sampling,
        cache_sampling_noise,
        fid_stats_used_from,
    ]
    if report_prc:
        headers += ["precision", "recall", "f-score"]
        row += [prc, rcl, fsc]

    log_table = tabulate([row], headers=headers, tablefmt="pipe")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpt_path = f"ckpt: {ckpt_epoch}, time: {now}"
    task = "gen" if task_mode == "generation" else "rec"
    file_path = f"{tabl_dir}/eval_fdr6_tabl_{ckpt_epoch}_{task}_ema={use_ema}.txt"
    with open(file_path, "a") as f:
        f.write("\n" + ckpt_path + "\n-----\n" + log_table + "\n")
    logger.info(f"appended FDr-6 table to {file_path}")


class FvcoreWrapper(torch.nn.Module):
    """fvcore traces forward(input); route that to the sampling loop."""

    def __init__(self, model, gen_kwargs):
        super().__init__()
        self.model = model
        self.gen_kwargs = gen_kwargs

    def forward(self, dummy_input):
        # the dummy input is only there to satisfy fvcore's api
        return self.model.generate(**self.gen_kwargs)


@torch.inference_mode()
def run_to_measure_flops(
    model, steps=1, device=get_device_type(), ptdtype=torch.bfloat16
):
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    flops_model = FvcoreWrapper(
        model,
        gen_kwargs={
            "batch_size": 1,
            "y": torch.zeros(1, dtype=torch.long, device=device),
            "cfg": 0.0,
            "cfg_position": "angle",
            "forward_steps": steps,
            "device": device,
        },
    )

    device_type = device if isinstance(device, str) else device.type
    with torch.autocast(device_type=device_type, dtype=ptdtype):
        dummy_input = torch.randn(1).to(device=device, dtype=ptdtype)
        flops = FlopCountAnalysis(flops_model, dummy_input)
        flops.unsupported_ops_warnings(False)
        gflops = flops.total() / 1e9

    logger.info(f"total GFLOPs: {gflops:.3f}")
    logger.info(flop_count_table(flops, max_depth=2))
    return gflops


if __name__ == "__main__":
    main(cli_args)
