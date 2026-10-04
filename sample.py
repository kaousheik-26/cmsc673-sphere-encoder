import argparse
import datetime
import time
import glob
import json
import os
import os.path as osp
import shutil
import logging
from contextlib import nullcontext
from types import SimpleNamespace

from cli_utils import str2bool, get_device_type, get_dist_backend, set_device, cfg_sweep
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import sphere.rng as rng
from sphere.loader import create_loader, create_dataset, cycle, get_dataset_cls
from torchvision import transforms
from sphere.builder import build_model
from sphere.utils import (
    load_ckpt,
    make_label_tiles,
    save_image,
    save_tensors_to_images,
    nn_concat_all_gather,
)
from sphere.ema import ModuleEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# fmt: off

def save_grid(x, path, nrow, max_nimgs, slice_rows=0):
    """save a [B, C, H, W] batch as one grid, or as several grids of
    `slice_rows * nrow` images each (<stem>_part=<k>.png) when slice_rows > 0"""
    if slice_rows <= 0:
        save_tensors_to_images(x, path=path, nrow=nrow, max_nimgs=max_nimgs)
        return
    x = x[:max_nimgs]
    per_part = slice_rows * nrow
    stem, ext = osp.splitext(path)
    for k, start in enumerate(range(0, x.shape[0], per_part)):
        save_tensors_to_images(
            x[start : start + per_part],
            path=f"{stem}_part={k:03d}{ext}",
            nrow=nrow,
            max_nimgs=per_part,
        )


parser = argparse.ArgumentParser(description="G1 Generation")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="visualization")
parser.add_argument("--job_dir", type=str, default=None)
# --- reconstruction
parser.add_argument("--reconstruction_mode", type=str2bool, default=False)
parser.add_argument("--num_rec_samples", type=int, default=8)
parser.add_argument("--gap_angle_deg", type=float, default=10)
parser.add_argument("--continue_sampling", type=str2bool, default=False)
# --- generation
parser.add_argument("--load_ckpt_strict", type=str2bool, default=True)
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--num_gen_samples", type=int, default=64)
parser.add_argument("--batch_size_per_rank", type=int, default=16)
parser.add_argument("--class_of_interests", type=int, nargs="+", default=[100])
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--compile_model", type=str2bool, default=True)
parser.add_argument("--random_sample_classes", type=str2bool, default=False)
parser.add_argument("--sample_all_classes", type=str2bool, default=False, help="loop over every class of the dataset (0 .. num_classes-1), ignoring --class_of_interests")
parser.add_argument("--sample_all_classes_start", type=int, default=0, help="first class index (inclusive) when --sample_all_classes is on")
parser.add_argument("--sample_all_classes_end", type=int, default=-1, help="last class index (exclusive) when --sample_all_classes is on; -1 means num_classes")
parser.add_argument("--forward_steps", type=int, nargs="+", default=[4])
parser.add_argument("--cache_sampling_noise", type=str2bool, default=True)
parser.add_argument("--seed_sampling", type=str2bool, default=False)
parser.add_argument("--seed_offset", type=int, default=0, help="added to the base seed (99) when --seed_sampling is on, so a different but still reproducible set of samples is drawn; a nonzero offset is appended to the output names as _seed=<offset>")
parser.add_argument("--use_ema_model", type=str2bool, default=False)
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
# --- class picking mode
parser.add_argument("--class_picking_mode", type=str2bool, default=False)
parser.add_argument("--num_imgs_per_class", type=int, default=5)
parser.add_argument("--classes_per_sheet", type=int, default=50)
# --- guidance
parser.add_argument("--use_cfg", type=str2bool, default=False)
parser.add_argument("--cfg_min", type=float, default=0.0)
parser.add_argument("--cfg_max", type=float, default=5.0)
parser.add_argument("--cfg_gap", type=float, default=5.0)
parser.add_argument("--cfg_position", type=str, default="angle", choices=["angle"])
# --- saving
parser.add_argument("--save_grid_images", type=str2bool, default=True)
parser.add_argument("--save_step_images", type=str2bool, default=False)
parser.add_argument("--grid_nrow", type=int, default=8)
parser.add_argument("--grid_slice_rows", type=int, default=0, help="if > 0, slice each saved grid into multiple files of <grid_slice_rows> x <grid_nrow> images each, named <name>_part=<k>.png (avoids one huge png at 512px)")
parser.add_argument("--save_real_images", type=str2bool, default=False, help="also save a grid of real train images of the same class (same layout) next to each generated grid, as <name>_real.png")
# --- other
parser.add_argument("--override_configs", type=str2bool, default=False)
parser.add_argument("--sampling_init_angle", type=float, default=85.0)
parser.add_argument("--sampling_loop_angle", type=float, default=None, help="constant angle of the sampling loop after the first step; defaults to --sampling_init_angle")
# --- speed benchmark
parser.add_argument("--benchmark", type=str2bool, default=False, help="time noise -> uint8 generation on one GPU and write a timing JSON instead of saving images; uses --batch_size_per_rank, the first --forward_steps, and --cfg_min when --use_cfg")
parser.add_argument("--bench_warmup", type=int, default=3, help="untimed batches before timing (covers torch.compile)")
parser.add_argument("--bench_iters", type=int, default=10, help="timed batches")
parser.add_argument("--bench_out", type=str, default="", help="timing JSON path (default: <out_dir>/timing-<settings>-bs<bsz>.json)")
cli_args = parser.parse_args()
if cli_args.sampling_loop_angle is None:
    cli_args.sampling_loop_angle = cli_args.sampling_init_angle
# fmt: on
# -----------------------------------------------------------------------------


def run_benchmark(
    args, model, ckpt_epoch, out_dir, device, device_type, ptdtype, ddp_world_size
):
    """Warmup, then time noise -> uint8 image batches with the eval settings (bf16 autocast,
    torch.compile if enabled, cached sampling noise), and write the timings to a JSON file.
    """
    if ddp_world_size != 1:
        raise ValueError("--benchmark runs on a single process")
    bsz = args.batch_size_per_rank
    fwd_step = args.forward_steps[0]
    cfg = args.cfg_min if args.use_cfg else 0.0
    n_batches = args.bench_warmup + args.bench_iters
    labels = (
        torch.from_numpy(
            np.random.default_rng(0).choice(args.num_classes, size=bsz * n_batches)
        )
        .long()
        .to(device)
    )

    def sync():
        if device_type == "cuda":
            torch.cuda.synchronize(device)

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(n_batches):
        y = labels[i * bsz : (i + 1) * bsz]
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device_type, dtype=ptdtype):
            _, x = model.generate(
                batch_size=bsz,
                y=y,
                cfg=cfg,
                cfg_position=args.cfg_position,
                forward_steps=fwd_step,
                cache_sampling_noise=args.cache_sampling_noise,
                sampling_init_angle=args.sampling_init_angle,
                sampling_loop_angle=args.sampling_loop_angle,
                return_step_images=False,
                device=device,
            )
        pixels = (
            (x.float() * 255).clamp(0, 255).round().to(torch.uint8)
        )  # noqa: F841  (generate returns [0, 1])
        sync()
        if i >= args.bench_warmup:
            times.append(time.perf_counter() - t0)
        logger.info(
            f"benchmark batch {i + 1}/{n_batches}{' (warmup)' if i < args.bench_warmup else ''}"
        )

    times_np = np.asarray(times, dtype=np.float64)
    per_image = times_np / bsz
    streams = 2 if cfg > 0 else 1  # angle cfg runs a second, unconditional stream
    result = dict(
        batch_size=bsz,
        timed_batches=len(times),
        num_images=int(bsz * len(times)),
        sec_per_image_mean=float(per_image.mean()),
        sec_per_image_std=float(per_image.std()),
        images_per_sec=float(bsz * len(times) / times_np.sum()),
        sec_per_batch=[float(t) for t in times],
        peak_mem_gb=(
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if device_type == "cuda"
            else 0.0
        ),
        gpu=torch.cuda.get_device_name() if device_type == "cuda" else device_type,
        torch=torch.__version__,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        family="g1",
        model=f"G1-{args.vit_dec_model_size}/{args.patch_size}",
        img_size=args.image_size,
        precision=args.dtype.replace("float16", "f16").replace("float32", "fp32"),
        tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        compiled=bool(args.compile_model),
        sampler="loop",
        steps=fwd_step,
        guidance=dict(
            method="angle-cfg" if cfg > 0 else "none", scale=cfg, uncond_pass=cfg > 0
        ),
        # main = decoder calls (T, or 2T with angle cfg); the frozen DINOv3 encoder runs T-1 times per stream
        nfe_per_image=dict(main=fwd_step * streams, encoder=(fwd_step - 1) * streams),
        encoder=str(getattr(args, "load_pretrained_encoder", args.vit_enc_model_size)),
        sampling=dict(
            init_angle=args.sampling_init_angle,
            loop_angle=args.sampling_loop_angle,
            cache_noise=bool(args.cache_sampling_noise),
        ),
        job_dir=args.job_dir,
        ckpt=ckpt_epoch,
        ema=bool(args.use_ema_model),
        warmup_batches=args.bench_warmup,
    )
    tag = f"steps{fwd_step}-cfg{cfg}-init{args.sampling_init_angle}-loop{args.sampling_loop_angle}-res{args.image_size}"
    path = args.bench_out or osp.join(out_dir, f"timing-{tag}-bs{bsz}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    nfe = sum(result["nfe_per_image"].values())
    logger.info(
        f"Benchmark {result['model']} @ {args.image_size}px bs{bsz}: {result['sec_per_image_mean']:.4f} +- "
        f"{result['sec_per_image_std']:.4f} s/img, {result['images_per_sec']:.3f} img/s, {nfe} NFE "
        f"({1000 * result['sec_per_image_mean'] / nfe:.1f} ms/NFE), peak {result['peak_mem_gb']:.2f} GB on {result['gpu']}"
    )
    logger.info(f"wrote {path}")


def main(cli_args):
    # setup dirs
    exp_dir = osp.join(cli_args.dev_dir, "experiments", cli_args.job_dir)

    # prepare to merge config
    cli_args_dict = vars(cli_args)

    # load config from exp folder
    logger.info(f"load cfg from {exp_dir}")
    config_path = os.path.join(exp_dir, "cfg.json")
    with open(config_path, "r") as fio:
        cfg_args = json.load(fio)

    # let cli args override config file args
    cfg_args.update(cli_args_dict)

    # convert to namespace for easy access
    args = SimpleNamespace(**cfg_args)
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    # compatibility for older configs
    args.data_dir = "datasets"
    args.interp_mode = "bicubic"

    # override configs
    if args.override_configs:
        pass

    # various inits, derived attributes, I/O setup
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
    ddp_rank0 = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed

    # seed: base 99 plus a user offset, so the same offset always reproduces
    # the same grids and a different offset gives a fresh set
    seed = None
    seed_tag = ""
    if args.seed_sampling:
        seed = 99 + args.seed_offset
        torch.manual_seed(seed + seed_offset)
        if args.seed_offset != 0:
            seed_tag = f"_seed={args.seed_offset}"
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    logger.info(f"using dtype: {ptdtype}, device: {device}, ddp_rank: {ddp_rank}")

    # create output folders
    out_dir = osp.join(args.dev_dir, args.out_dir, args.job_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"create output folder: {out_dir}")

    # if args.dataset_name in [
    #     "cifar-10",
    #     "animal-faces",
    #     "flowers-102",
    # ]:
    #     args.random_sample_classes = True
    #     logger.info(f"enable random sampling classes for {args.dataset_name}")

    if args.random_sample_classes:
        args.class_of_interests = [0]  # dummy for one loop
    elif args.sample_all_classes:
        assert (
            args.num_classes > 0
        ), "--sample_all_classes needs a class-conditional model"
        start = args.sample_all_classes_start
        end = (
            args.num_classes
            if args.sample_all_classes_end < 0
            else args.sample_all_classes_end
        )
        assert (
            0 <= start < end <= args.num_classes
        ), f"bad class range [{start}, {end}) for num_classes={args.num_classes}"
        args.class_of_interests = list(range(start, end))
        logger.info(
            f"sampling classes [{start}, {end}) ({len(args.class_of_interests)} of "
            f"{args.num_classes}) of {args.dataset_name}"
        )

    # build model
    model = build_model(args, for_train=False)
    model.to(dtype=ptdtype, device=device, memory_format=torch.channels_last)
    logger.info(model)

    ema_model = ModuleEMA(model)
    ema_model.eval().requires_grad_(False)

    # load ckpt path
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(ckpts) == 0:
        raise ValueError("no checkpoints to eval")
    ckpts = sorted(ckpts)

    # optionally load from a specific ckpt
    load_from = (
        ckpts[-1]
        if args.ckpt_fname is None
        else os.path.join(ckpt_dir, args.ckpt_fname)
    )
    assert os.path.exists(load_from)
    logger.info(f"find the latest ckpt: {load_from}")
    ckpt_epoch = osp.basename(load_from).replace(".pth", "")

    # load ckpt
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

    if args.benchmark:
        run_benchmark(
            args,
            model,
            ckpt_epoch,
            out_dir,
            device,
            device_type,
            ptdtype,
            ddp_world_size,
        )
        dist.destroy_process_group()
        return

    if args.reconstruction_mode:
        # one row per class of interest, unless classes are sampled randomly
        if not args.random_sample_classes:
            args.num_rec_samples = len(args.class_of_interests)
        # pad to a multiple of the world size; the extra rows are dropped
        # after the all-gather
        args.num_gen_samples = int(
            np.ceil(args.num_rec_samples / ddp_world_size) * ddp_world_size
        )
        args.batch_size_per_rank = 1

    if not args.class_picking_mode:
        args.batch_size_per_rank = min(
            args.batch_size_per_rank, args.num_gen_samples // ddp_world_size
        )
        assert args.num_gen_samples % (ddp_world_size * args.batch_size_per_rank) == 0
        num_batches_per_rank = int(
            args.num_gen_samples / ddp_world_size / args.batch_size_per_rank
        )

    if args.reconstruction_mode:
        # build dataloader
        dataset_cls = get_dataset_cls(args.dataset_name)

        loader = create_loader(
            dataset_cls,
            osp.join(args.dev_dir, args.data_dir, args.dataset_name),
            args.image_size,
            args.patch_size,
            max_samples=args.max_samples,
            interp_mode=args.interp_mode,
            rot_degrees=args.rot_degrees,
            crop_mode=args.crop_mode,
            flip_image=args.flip_image,
            extra_padding=args.extra_padding,
            concat_train_val_splits=args.concat_train_val_splits,
            ddp_world_size=ddp_world_size,
            ddp_rank=ddp_rank,
            batch_size_per_rank=args.batch_size_per_rank,
            num_workers=args.num_workers,
            load_from_zip=args.load_from_zip,
        )[2]
        loader = cycle(loader)

        # each row is one image with different angles
        max_angle_deg = int(args.noise_sigma_max_angle)
        rec_angles = list(range(0, 80, args.gap_angle_deg)) + list(
            range(80, max_angle_deg + 1, 1)
        )
        args.grid_nrow = len(rec_angles) + 1  # including the original image
        rec_clss_tag = (
            "all"
            if args.random_sample_classes
            else "-".join(f"{c:05d}" for c in args.class_of_interests)
        )

        dist.barrier()
        logger.info("start reconstructing images 🏞️")
        recon_imgs = []

        # each rank owns a contiguous chunk of the class list, so that the
        # all-gathered rows keep the order of `class_of_interests`
        if args.random_sample_classes:
            rank_targets = [None] * num_batches_per_rank
        else:
            padded = args.class_of_interests + [args.class_of_interests[-1]] * (
                args.num_gen_samples - args.num_rec_samples
            )
            rank_targets = padded[
                ddp_rank * num_batches_per_rank : (ddp_rank + 1) * num_batches_per_rank
            ]

        for target in rank_targets:
            # draw from the loader until an image of the target class shows up
            while True:
                imgs, clss = next(loader)[:2]
                if target is None or int(clss[0]) == target:
                    break

            imgs = imgs.to(device, non_blocking=True)
            clss = clss.to(device, non_blocking=True)
            recon_imgs.append(imgs * 0.5 + 0.5)

            for a in rec_angles:
                noise_scaler = a / max_angle_deg
                with torch.autocast(device_type=device_type, dtype=ptdtype):
                    recs = model.reconstruct(
                        imgs,
                        clss,
                        noise_scaler=noise_scaler,
                        sampling=True,
                        cache_sampling_noise=args.cache_sampling_noise,
                        sampling_max_angle=max_angle_deg,
                        continue_sampling=noise_scaler > 0.5 and args.continue_sampling,
                    )
                    recon_imgs.append(recs)

        dist.barrier()

        recon_imgs = torch.cat(recon_imgs, dim=0)
        recon_imgs = nn_concat_all_gather(recon_imgs)
        save_tensors_to_images(
            recon_imgs,
            path=osp.join(
                out_dir,
                f"recs_pth={ckpt_epoch}_"
                + f"cache={args.cache_sampling_noise}_"
                + f"clss={rec_clss_tag}"
                + seed_tag
                + ".png",
            ),
            nrow=args.grid_nrow,
            max_nimgs=args.num_rec_samples * args.grid_nrow,
        )
        dist.destroy_process_group()
        return

    if args.use_cfg:
        cfg_vals = cfg_sweep(args.cfg_min, args.cfg_max, args.cfg_gap)
    else:
        # guidance off: 0 degrees of angle cfg
        cfg_vals = [0.0]

    if args.class_picking_mode:
        run_class_picking(
            args,
            model,
            cfg_vals,
            ckpt_epoch,
            out_dir,
            device,
            device_type,
            ptdtype,
            ddp_rank,
            ddp_world_size,
            seed=seed,
            seed_tag=seed_tag,
        )
        dist.destroy_process_group()
        return

    # real images of the same class, laid out like the generated grid, so the
    # two can be compared side by side. Loaded on rank 0 only, once per class.
    real_ds, real_idx_by_class, real_done = None, None, set()
    if args.save_real_images and args.save_grid_images and ddp_rank0:
        assert (
            not args.random_sample_classes
        ), "--save_real_images needs fixed classes (random_sample_classes=False)"
        real_ds, real_idx_by_class = build_real_class_index(args)

    # stack loops
    for try_id in range(args.num_trials):
        for cfg in cfg_vals:
            for fwd_step in args.forward_steps:
                for clss in args.class_of_interests:
                    # create the sub folder to save images
                    save_name = (
                        f"imgs"
                        f"_try={try_id:02d}"
                        f"_clss={clss:05d}"
                        f"_pth={ckpt_epoch}"
                        f"_ema={args.use_ema_model}"
                        f"_cfg={cfg}-{args.cfg_position}"
                        f"_steps={fwd_step}"
                        f"_cache={args.cache_sampling_noise}"
                        f"_init={args.sampling_init_angle}"
                        f"_loop={args.sampling_loop_angle}"
                    )

                    save_name += seed_tag

                    if not args.save_grid_images:
                        # save images in this folder
                        sub_dir = osp.join(out_dir, save_name)
                        if osp.exists(sub_dir):
                            shutil.rmtree(sub_dir)
                        os.makedirs(sub_dir, exist_ok=True)
                    else:
                        # save grid image in this path
                        save_path = osp.join(out_dir, save_name + ".png")

                    # prepare class conditioning if needed
                    clss = torch.tensor(
                        [clss] * args.batch_size_per_rank,
                        dtype=torch.long,
                        device=device,
                    )
                    if args.save_grid_images:
                        gen_images = []
                    if args.save_step_images:
                        # one bucket per sampling step (h + forward steps)
                        step_gen_images = [[] for _ in range(fwd_step)]

                    dist.barrier()
                    logger.info("start sampling images 🍭")

                    pbar = tqdm(range(num_batches_per_rank), total=num_batches_per_rank)
                    for batch_idx in pbar:

                        with (
                            torch.random.fork_rng(devices=[device])
                            if args.seed_sampling
                            else nullcontext()
                        ):
                            if args.seed_sampling:
                                torch.manual_seed(
                                    rng.fold_in(seed, ddp_rank, batch_idx)
                                )

                            t0 = time.perf_counter()
                            with torch.autocast(device_type=device_type, dtype=ptdtype):
                                gen_out = model.generate(
                                    batch_size=args.batch_size_per_rank,
                                    y=None if args.random_sample_classes else clss,
                                    cfg=cfg,
                                    cfg_position=args.cfg_position,
                                    forward_steps=fwd_step,
                                    cache_sampling_noise=args.cache_sampling_noise,
                                    sampling_init_angle=args.sampling_init_angle,
                                    sampling_loop_angle=args.sampling_loop_angle,
                                    return_step_images=args.save_step_images,
                                    device=device,
                                )
                            if args.save_step_images:
                                x_gen_1_step, x_gen_n_step, x_gen_steps = gen_out
                                for i, step_img in enumerate(x_gen_steps):
                                    step_gen_images[i].append(step_img)
                            else:
                                x_gen_1_step, x_gen_n_step = gen_out
                            gen_time = time.perf_counter() - t0
                            # logger.info(
                            #     f"Batch {batch_idx+1}/{num_batches_per_rank}, "
                            #     f"Generated {args.batch_size_per_rank} images in {gen_time:.2f}s, "
                            #     f"{args.batch_size_per_rank / gen_time:.2f} img/s"
                            # )

                        if args.save_grid_images:
                            gen_images.append(x_gen_n_step)
                        else:
                            save_image(
                                x=x_gen_n_step,
                                batch_idx=batch_idx,
                                ddp_rank=ddp_rank,
                                save_dir=sub_dir,
                            )

                    dist.barrier()
                    if args.save_grid_images:
                        gen_images = torch.cat(gen_images, dim=0)
                        gen_images = nn_concat_all_gather(gen_images)

                        # cifar image in the paper, max 32 in a row
                        save_grid(
                            gen_images,
                            path=save_path,
                            nrow=args.grid_nrow,
                            max_nimgs=args.num_gen_samples,
                            slice_rows=args.grid_slice_rows,
                        )

                        # the real-image counterpart: one grid per class, not
                        # per (try, cfg, step), since it does not depend on them
                        clss_id = int(clss[0])
                        if real_ds is not None and clss_id not in real_done:
                            real_done.add(clss_id)
                            real_images = load_real_images(
                                real_ds,
                                real_idx_by_class,
                                clss_id,
                                args.num_gen_samples,
                                seed=seed if args.seed_sampling else 0,
                            )
                            real_path = osp.join(
                                out_dir,
                                f"imgs_real_clss={clss_id:05d}"
                                f"_n={args.num_gen_samples}{seed_tag}.png",
                            )
                            save_grid(
                                real_images,
                                path=real_path,
                                nrow=args.grid_nrow,
                                max_nimgs=args.num_gen_samples,
                                slice_rows=args.grid_slice_rows,
                            )
                            logger.info(f"saved real-image grid to {real_path}")

                    if args.save_step_images:
                        # one grid per sampling step, saved into a folder that
                        # shares the final image's base name
                        step_dir = osp.join(out_dir, save_name)
                        os.makedirs(step_dir, exist_ok=True)
                        for i, step_imgs in enumerate(step_gen_images):
                            step_imgs = torch.cat(step_imgs, dim=0)
                            step_imgs = nn_concat_all_gather(step_imgs)
                            save_grid(
                                step_imgs,
                                path=osp.join(step_dir, f"step={i:03d}.png"),
                                nrow=args.grid_nrow,
                                max_nimgs=args.num_gen_samples,
                                slice_rows=args.grid_slice_rows,
                            )

    dist.destroy_process_group()
    return


def build_real_class_index(args):
    """train dataset with the eval-style transform (resize + center crop, no
    flip) and a class_id -> [indices] map, built from the metadata list so no
    image is decoded here."""
    dataset_cls = get_dataset_cls(args.dataset_name)
    interp = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
    }.get(str(args.interp_mode), transforms.InterpolationMode.BICUBIC)
    ds = create_dataset(
        dataset_cls,
        root=osp.join(args.dev_dir, args.data_dir, args.dataset_name),
        split="train",
        concat_train_val_splits=args.concat_train_val_splits,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=interp),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
            ]
        ),
        load_from_zip=args.load_from_zip,
    )

    idx_by_class = {}
    if hasattr(ds, "list"):  # ListDataset: metadata rows
        labels = [int(item["class_id"]) for item in ds.list]
    elif hasattr(ds, "targets"):  # torchvision cifar
        labels = [int(t) for t in ds.targets]
    else:
        raise NotImplementedError(f"no class index for {type(ds).__name__}")
    for i, c in enumerate(labels):
        idx_by_class.setdefault(c, []).append(i)
    logger.info(
        f"real-image index: {len(labels)} images over {len(idx_by_class)} classes"
    )
    return ds, idx_by_class


def load_real_images(ds, idx_by_class, class_id, n, seed=0):
    """n real images of class_id as a [n, 3, H, W] tensor in [0, 1], drawn
    with a fixed seed so the same class always gives the same grid."""
    idx = list(idx_by_class.get(class_id, []))
    assert idx, f"no train images for class {class_id}"
    g = np.random.default_rng(seed + class_id)
    pick = g.choice(idx, size=min(n, len(idx)), replace=False)
    imgs = [ds[int(i)][0] for i in pick]
    return torch.stack(imgs, dim=0)


def run_class_picking(
    args,
    model,
    cfg_vals,
    ckpt_epoch,
    out_dir,
    device,
    device_type,
    ptdtype,
    ddp_rank,
    ddp_world_size,
    seed=None,
    seed_tag="",
):
    """
    generate `num_imgs_per_class` images for every class and save contact
    sheets: each row is [class id tile | img_1 ... img_k], `classes_per_sheet`
    rows per sheet. used to pick classes for visualization.
    """
    assert args.num_classes > 0, "class picking needs a class-conditional model"
    num_classes = args.num_classes
    k = args.num_imgs_per_class
    bs = args.batch_size_per_rank
    total = num_classes * k

    # global batch index g = b * world_size + rank, so that concatenating the
    # all-gathered outputs batch by batch restores the original label order.
    num_global_batches = int(np.ceil(total / bs))
    num_batches_per_rank = int(np.ceil(num_global_batches / ddp_world_size))
    padded_total = num_batches_per_rank * ddp_world_size * bs

    all_labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(k)
    all_labels = torch.cat(
        [all_labels, torch.zeros(padded_total - total, dtype=torch.long)]
    )
    all_labels = all_labels.view(num_batches_per_rank, ddp_world_size, bs)

    for cfg in cfg_vals:
        for fwd_step in args.forward_steps:
            save_name = (
                f"picking"
                f"_k={k}"
                f"_pth={ckpt_epoch}"
                f"_ema={args.use_ema_model}"
                f"_cfg={cfg}-{args.cfg_position}"
                f"_steps={fwd_step}"
                f"_cache={args.cache_sampling_noise}"
                f"_init={args.sampling_init_angle}"
                f"_loop={args.sampling_loop_angle}"
            )
            save_name += seed_tag

            pick_dir = osp.join(out_dir, save_name)
            os.makedirs(pick_dir, exist_ok=True)

            dist.barrier()
            logger.info(f"start class picking for {num_classes} classes 🎯")

            gen_images = []  # kept on cpu, already in global order
            pbar = tqdm(range(num_batches_per_rank), total=num_batches_per_rank)
            for batch_idx in pbar:
                y = all_labels[batch_idx, ddp_rank].to(device)
                global_batch_idx = batch_idx * ddp_world_size + ddp_rank

                with (
                    torch.random.fork_rng(devices=[device])
                    if seed is not None
                    else nullcontext()
                ):
                    if seed is not None:
                        torch.manual_seed(rng.fold_in(seed, "pick", global_batch_idx))
                    with torch.autocast(device_type=device_type, dtype=ptdtype):
                        _, x_gen_n_step = model.generate(
                            batch_size=bs,
                            y=y,
                            cfg=cfg,
                            cfg_position=args.cfg_position,
                            forward_steps=fwd_step,
                            cache_sampling_noise=args.cache_sampling_noise,
                            sampling_init_angle=args.sampling_init_angle,
                            sampling_loop_angle=args.sampling_loop_angle,
                            return_step_images=False,
                            device=device,
                        )
                # gather this batch from all ranks and move it off the gpu
                gen_images.append(nn_concat_all_gather(x_gen_n_step.float()).cpu())

            dist.barrier()
            gen_images = torch.cat(gen_images, dim=0)[:total]  # (C * k, 3, H, W)
            _, _, h, w = gen_images.shape

            for start in range(0, num_classes, args.classes_per_sheet):
                cls_ids = list(
                    range(start, min(start + args.classes_per_sheet, num_classes))
                )
                tiles = make_label_tiles(cls_ids, h, w)
                rows = []
                for i, c in enumerate(cls_ids):
                    rows.append(tiles[i : i + 1])
                    rows.append(gen_images[c * k : (c + 1) * k])
                sheet = torch.cat(rows, dim=0)
                save_tensors_to_images(
                    sheet,
                    path=osp.join(
                        pick_dir, f"sheet_{cls_ids[0]:05d}-{cls_ids[-1]:05d}.png"
                    ),
                    nrow=k + 1,
                    max_nimgs=len(cls_ids) * (k + 1),
                )
            logger.info(f"class picking sheets saved to {pick_dir}")


if __name__ == "__main__":
    main(cli_args)
