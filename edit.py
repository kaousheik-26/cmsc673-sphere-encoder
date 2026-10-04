"""
edit.py: a minimal playground for the g1 model.

--edit_mode class: encode the input image, then decode its latent
    conditioned on other ImageNet classes with the multi-step sampling loop
    (G1.edit). rows: [input | class id | edited image].
--edit_mode interp: two random noises, decode --num_interp points on the
    slerp path between them, with the class embeddings of --interp_classes
    interpolated alongside (--class_interp; G1.interpolate). no input image
    needed. rows are trials, columns go from class a to class b.
one grid per --forward_steps value.

usage (single gpu):
    torchrun --nproc_per_node=1 edit.py --job_dir <exp> --input_image <path> \
        --class_of_interests 384 535 923 --forward_steps 1 4 --edit_angle 85
    torchrun --nproc_per_node=1 edit.py --job_dir <exp> --edit_mode interp \
        --interp_classes 384 535 --num_interp 8
"""

import argparse
import datetime
import glob
import json
import os
import os.path as osp
import logging
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torchvision import datasets

from sphere.loader import center_crop_arr
from sphere.builder import build_model
from sphere.ema import ModuleEMA
from sphere.utils import load_ckpt, make_label_tiles, save_tensors_to_images
from cli_utils import str2bool, get_device_type, get_dist_backend, set_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# fmt: off
parser = argparse.ArgumentParser(description="G1 Image Editing")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="image_editing")
parser.add_argument("--job_dir", type=str, default=None)
# --- input
parser.add_argument("--input_image", type=str, default=None, help="path to an input image; if omitted, a random noise image is used instead")
# --- editing task
parser.add_argument("--edit_mode", type=str, default="class", choices=["class", "interp"], help="class: decode the input's latent conditioned on other classes; interp: interpolate between random samples of two classes")
# --- interpolation between two classes
parser.add_argument("--interp_classes", type=int, nargs=2, default=[384, 535], help="interp: the two class ids (endpoints)")
parser.add_argument("--num_interp", type=int, default=8, help="interp: number of images per row, endpoints included")
parser.add_argument("--class_interp", type=str, default="lerp", choices=["lerp", "slerp", "hard"], help="interp: how the class conditioning moves along the path; lerp/slerp mix the two class embeddings, hard switches label at the midpoint")
parser.add_argument("--interp_fix_latent", type=str2bool, default=False, help="interp: sample one point and keep it for every column, so only the class embedding is interpolated")
parser.add_argument("--num_trials", type=int, default=4, help="interp: rows per grid, each with different noise")
# --- class editing: decode the input's latent conditioned on other classes
# default: animals, in the model's class_id space (see datasets/imagenet/folder_to_id_to_label.json)
#   384 tabby, 535 golden_retriever, 909 Siberian_husky, 585 red_fox,
#   634 tiger, 294 lion, 976 giant_panda, 923 zebra
parser.add_argument("--class_of_interests", type=int, nargs="+", default=[384, 535, 909, 585, 634, 294, 976, 923], help="target class ids (model's class_id column); one edited image per class")
parser.add_argument("--class_label_file", type=str, default="workspace/datasets/imagenet/folder_to_id_to_label.json", help="jsonl of {class_id, label}; used to print class names on the grid tiles when present")
parser.add_argument("--forward_steps", type=int, nargs="+", default=[4], help="total decode steps (step 0 is the edit, the rest refine); one grid per value")
parser.add_argument("--edit_angle", type=float, default=85.0, help="degrees of noise injected at the first step (class: 0 = class-swapped reconstruction; interp: init angle)")
parser.add_argument("--sampling_loop_angle", type=float, default=None, help="constant angle of the sampling loop after the first step; defaults to --edit_angle")
parser.add_argument("--cache_sampling_noise", type=str2bool, default=True)
parser.add_argument("--save_step_images", type=str2bool, default=False, help="also save one grid per step: <name>_step=<t>.png")
# --- model
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--load_ckpt_strict", type=str2bool, default=True)
parser.add_argument("--compile_model", type=str2bool, default=False)
parser.add_argument("--use_ema_model", type=str2bool, default=True)
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--seed_mode", type=str, default="fix", choices=["fix", "random"], help="fix: seed every run with --seed (same noise each run); random: draw a fresh seed per run (logged and written into the file names as _seed=<n>)")
parser.add_argument("--seed", type=int, default=99, help="base seed when --seed_mode fix; the rank is added on top")
# --- saving
parser.add_argument("--save_name", type=str, default=None, help="output png stem; defaults to edit_pth=<ckpt>_... built from the settings")
cli_args = parser.parse_args()
if cli_args.sampling_loop_angle is None:
    cli_args.sampling_loop_angle = cli_args.edit_angle
# fmt: on
# -----------------------------------------------------------------------------


def load_image_to_tensor(path, image_size):
    """load an image as a [3, H, W] tensor in [-1, 1], center-cropped to image_size"""
    x = datasets.folder.pil_loader(path)
    x = np.array(center_crop_arr(x, image_size))
    x = x.astype(np.float32) / 255.0  # [0, 1]
    x = torch.from_numpy(x).permute(2, 0, 1)  # [3, H, W]
    x = x * 2 - 1  # [-1, 1]
    return x


def load_class_labels(path):
    """class_id -> label from the jsonl mapping file; {} when the file is missing"""
    if path is None or not osp.exists(path):
        return {}
    labels = {}
    with open(path, "r") as fio:
        for line in fio:
            line = line.strip()
            if line:
                row = json.loads(line)
                labels[int(row["class_id"])] = row["label"]
    return labels


def save_grid(tensors, path, nrow=0, max_nimgs=64):
    """save a [B, 3, H, W] tensor (or a list of them, concatenated along the
    row) in [0, 1] as one png grid"""
    save_tensors_to_images(tensors, path=path, nrow=nrow, max_nimgs=max_nimgs)
    logger.info(f"saved {path}")


def setup(cli_args):
    """merge cfg.json with cli args, init ddp, pick device/dtype, make out_dir"""
    exp_dir = osp.join(cli_args.dev_dir, "experiments", cli_args.job_dir)

    # load config from exp folder and let cli args override it
    logger.info(f"load cfg from {exp_dir}")
    with open(osp.join(exp_dir, "cfg.json"), "r") as fio:
        cfg_args = json.load(fio)
    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    # compatibility for older configs
    args.data_dir = "datasets"
    args.interp_mode = "bicubic"

    # ddp init
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device_type = get_device_type()
    device = set_device(device_type, ddp_local_rank)
    dist.init_process_group(
        backend=get_dist_backend(device_type),
        device_id=device,
        timeout=datetime.timedelta(hours=2),
    )

    # seed: fixed from --seed, or a fresh one per run. rank 0 draws the
    # random seed and broadcasts it so every rank agrees on the tag
    if args.seed_mode == "random":
        seed_t = torch.tensor([int.from_bytes(os.urandom(4), "little")], device=device)
        dist.broadcast(seed_t, src=0)
        seed = int(seed_t.item())
        seed_tag = f"_seed={seed}"
    else:
        seed = args.seed
        seed_tag = ""
    torch.manual_seed(seed + ddp_rank)
    logger.info(f"seed_mode={args.seed_mode}, seed={seed} (+rank)")
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    logger.info(f"using dtype: {ptdtype}, device: {device}, ddp_rank: {ddp_rank}")

    out_dir = osp.join(args.dev_dir, args.out_dir, args.job_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"create output folder: {out_dir}")

    ctx = SimpleNamespace(
        exp_dir=exp_dir,
        out_dir=out_dir,
        device=device,
        device_type=device_type,
        ptdtype=ptdtype,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp_rank0=ddp_rank == 0,
        seed=seed,
        seed_tag=seed_tag,
    )
    return args, ctx


def load_model(args, ctx):
    """build the g1 model and load the latest (or the given) checkpoint"""
    model = build_model(args, for_train=False)
    model.to(dtype=ctx.ptdtype, device=ctx.device, memory_format=torch.channels_last)
    logger.info(model)

    ema_model = ModuleEMA(model)
    ema_model.eval().requires_grad_(False)

    ckpt_dir = osp.join(ctx.exp_dir, "ckpt")
    ckpts = sorted(glob.glob(osp.join(ckpt_dir, "*.pth")))
    if len(ckpts) == 0:
        raise ValueError(f"no checkpoints in {ckpt_dir}")
    load_from = (
        ckpts[-1] if args.ckpt_fname is None else osp.join(ckpt_dir, args.ckpt_fname)
    )
    assert osp.exists(load_from), load_from
    logger.info(f"load ckpt: {load_from}")
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
        logger.info("compiling the model...")
        model = torch.compile(model)

    model.eval().requires_grad_(False)
    return model, ckpt_epoch


def run_class_edit(args, ctx, model, ckpt_epoch, x):
    """task 1: decode the input's latent conditioned on other classes"""
    assert args.num_classes > 0, "class editing needs a class-conditional model"
    # one row per target class: the same input, a different label
    classes = args.class_of_interests
    for c in classes:
        assert (
            0 <= c < args.num_classes
        ), f"class {c} out of range [0, {args.num_classes})"
    y = torch.tensor(classes, device=ctx.device, dtype=torch.long)
    x_in = x.expand(len(classes), -1, -1, -1).contiguous()

    h, w = x.shape[-2:]
    names = load_class_labels(args.class_label_file)
    tile_text = [f"{c} {names[c]}" if c in names else str(c) for c in classes]
    logger.info(f"target classes: {tile_text}")
    tiles = make_label_tiles(
        tile_text, h, w, font_size=max(10, h // 14)
    )  # [K, 3, H, W]
    tiles = tiles.to(device=ctx.device)
    input_col = x_in * 0.5 + 0.5

    for fwd_step in args.forward_steps:
        with torch.autocast(device_type=ctx.device_type, dtype=ctx.ptdtype):
            x_edit, step_images = model.edit(
                x_in,
                y,
                forward_steps=fwd_step,
                edit_angle=args.edit_angle,
                sampling_loop_angle=args.sampling_loop_angle,
                cache_sampling_noise=args.cache_sampling_noise,
                return_step_images=True,
            )

        save_name = args.save_name or (
            f"edit"
            f"_pth={ckpt_epoch}"
            f"_ema={args.use_ema_model}"
            f"_steps={fwd_step}"
            f"_eang={args.edit_angle}"
            f"_loop={args.sampling_loop_angle}"
            f"_cache={args.cache_sampling_noise}"
        )

        save_name += ctx.seed_tag
        if ctx.ddp_rank0:
            # each row: [input | class id | edited]
            cols = [input_col, tiles, x_edit.float()]
            save_grid(
                cols, path=osp.join(ctx.out_dir, save_name + ".png"), nrow=len(cols)
            )
            if args.save_step_images:
                for t, x_t in enumerate(step_images):
                    cols = [input_col, tiles, x_t.float()]
                    save_grid(
                        cols,
                        path=osp.join(ctx.out_dir, save_name + f"_step={t}.png"),
                        nrow=len(cols),
                    )


def run_interp(args, ctx, model, ckpt_epoch):
    """decode the slerp path between two random noises. one grid per
    --forward_steps; rows are trials, columns go from class a to class b"""
    K = args.num_trials
    ca, cb = args.interp_classes
    for c in (ca, cb):
        assert (
            0 <= c < args.num_classes
        ), f"class {c} out of range [0, {args.num_classes})"
    names = load_class_labels(args.class_label_file)
    logger.info(f"interpolating {ca} {names.get(ca, '')} -> {cb} {names.get(cb, '')}")
    y_a = torch.full((K,), ca, device=ctx.device, dtype=torch.long)
    y_b = torch.full((K,), cb, device=ctx.device, dtype=torch.long)

    for fwd_step in args.forward_steps:
        with torch.autocast(device_type=ctx.device_type, dtype=ctx.ptdtype):
            outs = model.interpolate(
                y_a,
                y_b,
                num_interp=args.num_interp,
                forward_steps=fwd_step,
                init_angle=args.edit_angle,
                sampling_loop_angle=args.sampling_loop_angle,
                cache_sampling_noise=args.cache_sampling_noise,
                class_interp=args.class_interp,
                fix_latent=args.interp_fix_latent,
                device=ctx.device,
            )

        save_name = args.save_name or (
            f"interp"
            f"_pth={ckpt_epoch}"
            f"_ema={args.use_ema_model}"
            f"_cls={ca}-{cb}"
            f"_n={args.num_interp}"
            f"_cinterp={args.class_interp}"
            f"_fixlat={args.interp_fix_latent}"
            f"_init={args.edit_angle}"
            f"_steps={fwd_step}"
            f"_loop={args.sampling_loop_angle}"
            f"_cache={args.cache_sampling_noise}"
        )

        save_name += ctx.seed_tag
        if ctx.ddp_rank0:
            # rows: trials; columns: class a -> ... -> class b
            cols = [o.float() for o in outs]
            save_grid(
                cols,
                path=osp.join(ctx.out_dir, save_name + ".png"),
                nrow=len(cols),
                max_nimgs=K * len(cols),
            )


def main(cli_args):
    args, ctx = setup(cli_args)
    model, ckpt_epoch = load_model(args, ctx)

    # input: a real image if given, otherwise a random noise image
    if args.input_image is not None:
        x = load_image_to_tensor(args.input_image, args.image_size)  # [-1, 1]
        x = x.unsqueeze(0).to(device=ctx.device)
        logger.info(f"loaded input image {args.input_image}: {tuple(x.shape)}")
    else:
        x = torch.randn((1, 3, args.image_size, args.image_size), device=ctx.device)
        logger.info(f"no --input_image given, using random noise: {tuple(x.shape)}")

    if args.edit_mode == "class":
        run_class_edit(args, ctx, model, ckpt_epoch, x)
    elif args.edit_mode == "interp":
        run_interp(args, ctx, model, ckpt_epoch)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(cli_args)
