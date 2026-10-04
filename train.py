import datetime
import math
import json
import argparse
import os
import os.path as osp
import time
import logging
from contextlib import nullcontext
from functools import partial
from cli_utils import (
    str2bool,
    none_or_str,
    get_device_type,
    get_dist_backend,
    set_device,
)

import torch
import torch.distributed as dist
import wandb

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from sphere.loader import create_loader, cycle, get_dataset_cls
from sphere.logger import append_log, setup_logging
from sphere.builder import build_model
from sphere.utils import (
    cosine_scheduler,
    encoder_lr_scaler_scheduler,
    load_ckpt,
    save_ckpt,
    is_ckpt_valid,
    find_latest_valid_ckpt,
    visualize,
    apply_dotted_overrides,
    _BF16ErrorFeedbackState,
    bf16_ef_comm_hook,
    ParamAverager,
)

from sphere.ema import ModuleEMA, get_ema_decay
from sphere.encoder import get_token_channels
from sphere.loss import G1Loss
from sphere.score.config import config as score_config

torch_version = [int(x) for x in torch.__version__.split(".")[:2]]
is_torch2 = torch_version >= [2, 0]

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# fmt: off
parser = argparse.ArgumentParser(description="G1 Training")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="experiments")
parser.add_argument("--data_dir", type=str, default="datasets")
# --- logging
parser.add_argument("--log_interval", type=int, default=100, help="in iterations")
parser.add_argument("--grad_probe", type=str2bool, default=False, help="on log steps, log the gradient norm of every loss term on the decoded image (gnorm_*)")
parser.add_argument("--vis_interval", type=int, default=2, help="in epochs")
parser.add_argument("--ckpt_save_interval", type=int, default=10, help="in epochs")
parser.add_argument("--cleanup_ckpt", type=str2bool, default=True)
parser.add_argument("--cleanup_ckpt_interval", type=int, default=5)
# --- visualization
parser.add_argument("--class_of_interest", type=int, nargs="+", default=None)
parser.add_argument("--forward_steps", type=int, default=2)
parser.add_argument("--cfg_position", type=str, default="angle", choices=["angle"])
parser.add_argument("--cfg", type=float, default=0.0)
# --- wandb
parser.add_argument("--run_str", type=str, default="")
parser.add_argument("--use_wandb", type=str2bool, default=False)
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_key", type=str, default=None)
parser.add_argument("--wandb_dir", type=str, default="wandb")
# --- dataset
parser.add_argument("--dataset_name", type=str, default="cifar-10", choices=["flowers-102", "imagenet"])
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--crop_mode", type=str, default="center_adm", choices=["center", "random", "center_adm", "random_adm"])
parser.add_argument("--flip_image", type=str2bool, default=True)
parser.add_argument("--extra_padding", type=str2bool, default=False)
parser.add_argument("--color_jitter", type=str2bool, default=False)
parser.add_argument("--rot_degrees", type=int, default=0)
parser.add_argument("--interp_mode", type=str, default="bicubic", choices=["bicubic", "nearest"])
parser.add_argument("--concat_train_val_splits", type=str2bool, default=False)
parser.add_argument("--load_from_zip", type=str2bool, default=False)
parser.add_argument("--max_samples", type=int, default=-1, help="for using partial data")
# --- optimizer
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--batch_size_per_rank", type=int, default=16)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--optimizer_name", type=str, default="adamw", choices=["adamw"])
parser.add_argument("--grad_accum_steps", type=int, default=1)
parser.add_argument("--rescale_lr_with_batch_size", type=str2bool, default=True)
# --- scheduler
parser.add_argument("--epochs", type=int, default=800)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--encoder_lr_scaler", type=float, default=0.1)
parser.add_argument("--encoder_lr_scaler_min", type=float, default=0.1, help="floor for the encoder lr scaler after decay")
parser.add_argument("--encoder_lr_scaler_decay_epochs", type=int, default=0, help="epochs to cosine-decay the encoder lr scaler over; <=0 disables")
parser.add_argument("--decay_lr", type=str2bool, default=True)
# --- latent
parser.add_argument("--compression_ratio", type=float, default=3.0)
parser.add_argument("--latent_resolution", type=str, default="high", choices=["low", "high", "super_high"])
# --- noise
parser.add_argument("--noise_sigma_max_angle", type=float, default=90.0)
# --- angle
parser.add_argument("--use_angle_condition", type=str2bool, default=False)
parser.add_argument("--angle_condition_mode", type=str, default="adaln", choices=["both", "adaln", "token"])
# --- model
parser.add_argument("--vit_enc_model_size", type=str, default="base")
parser.add_argument("--vit_dec_model_size", type=str, default="base")
parser.add_argument("--drop_residual_path_prob", type=float, default=0.0)
parser.add_argument("--cond_generator", type=str2bool, default=True)
parser.add_argument("--pixel_head_type", type=str, default="linear", choices=["linear", "conv", "linear+conv"])
parser.add_argument("--pixel_head_use_tanh", type=str2bool, default=False)
parser.add_argument("--spherify_model", type=str2bool, default=False)
parser.add_argument("--spherify_mode", type=str, default="global", choices=["global", "local"])
parser.add_argument("--in_context_size", type=int, default=0)
parser.add_argument("--halve_model_size", type=str2bool, default=False)
parser.add_argument("--freeze_encoder", type=str2bool, default=False)
parser.add_argument("--sdpa_mode", type=none_or_str, default=None, choices=["manual", "sdpa", "sdpa_fp32", "flash_attn"])
parser.add_argument("--use_qk_norm", type=str2bool, default=False)
# --- pretrained encoder 
parser.add_argument("--load_pretrained_encoder", type=none_or_str, default=None)
parser.add_argument("--pretrained_encoder_train_last_n_blocks", type=int, default=0)
parser.add_argument("--pretrained_encoder_train_norm_ls_layers", type=str2bool, default=False)
parser.add_argument("--pretrained_encoder_train_final_norm", type=str2bool, default=False)
parser.add_argument("--pretrained_encoder_interpolate_tokens", type=str2bool, default=False)
parser.add_argument("--pretrained_encoder_interpolate_pos_embed", type=str2bool, default=False)
parser.add_argument("--pretrained_encoder_fix_compression_ratio", type=str2bool, default=False)
# --- ema model
parser.add_argument("--use_ema", type=str2bool, default=True)
parser.add_argument("--use_ema_scheduler", type=str2bool, default=False)
parser.add_argument("--ema_model_decay", type=float, default=0.9997)
# --- training
parser.add_argument("--launch_exec", type=str, default='torchrun', choices=['torchrun', 'mpiexec'])
parser.add_argument("--device_type", type=str, default=None, choices=["cuda", "cpu", "xpu"], help="auto-detected (cuda > xpu > cpu) if unset")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--blas_library", type=str, default=None, choices=["default", "cublas", "cublaslt", "ck"], help="override the CUDA/ROCm BLAS backend; cublas selects hipBLAS on ROCm")
parser.add_argument("--compile_model", type=str2bool, default=False)
parser.add_argument("--use_activation_checkpointing", type=str2bool, default=False)
parser.add_argument("--grad_comm_dtype", type=str, default="fp32", choices=["fp32", "bf16", "bf16_ef"], help="dtype for DDP gradient all-reduce; bf16 halves comm volume, bf16_ef adds error feedback")
parser.add_argument("--ddp_bucket_cap_mb", type=int, default=None, help="DDP gradient bucket cap in MiB; omitted uses PyTorch's 25 MiB default")
parser.add_argument("--max_iters_per_epoch", type=int, default=0, help="cap on per-rank iterations per epoch (0 = full epoch); for smoke tests")
# --- profiling (opt-in; traces a few steps on rank0 then keeps training)
parser.add_argument("--profile", type=str2bool, default=False)
parser.add_argument("--profile_wait", type=int, default=3)
parser.add_argument("--profile_warmup", type=int, default=3)
parser.add_argument("--profile_active", type=int, default=5)
# --- loss
parser.add_argument("--distance_loss_type", type=str, default="l1", choices=["l1", "l2", "l1+l2", "l2+l1"])
parser.add_argument("--pix_recon_dist_loss_weight", type=float, default=0.0)
parser.add_argument("--pix_recon_dist_apply_regime", type=str, default="lo", choices=["hi", "lo", "all"])
parser.add_argument("--pix_recon_perc_loss_weight", type=float, default=0.0)
parser.add_argument("--lat_con_loss_weight", type=float, default=0.0)
parser.add_argument("--lat_con_freeze_encoder", type=str2bool, default=False)
parser.add_argument("--lat_con_use_diffaug", type=str2bool, default=False)
parser.add_argument("--lat_con_apply_regime", type=str, default="hi", choices=["hi", "lo", "all"])
# --- regime
parser.add_argument("--weight_anchor_loss", type=str2bool, default=False)
parser.add_argument("--weight_anchor_cutoff_params", type=str, default="70-5-soft")
# --- representation Frechet distance loss (FD-loss) in the encoder space
parser.add_argument("--fd_loss_weight", type=float, default=0.0)
parser.add_argument("--fd_loss_start_epoch", type=int, default=0, help="the clean statistics are warmed up before this epoch")
parser.add_argument("--fd_pool", type=str, default="cls", choices=["cls", "mean", "token"], help="cls: the backbone CLS token (paper / FDr-6, needs a DINO encoder); mean: token-mean per image; token: every token is a sample")
parser.add_argument("--fd_apply_regime", type=str, default="hi", choices=["hi", "lo", "all"])
parser.add_argument("--fd_ema_decay", type=float, default=0.999)
parser.add_argument("--fd_use_clean_ema", type=str2bool, default=True)
parser.add_argument("--fd_use_noisy_ema", type=str2bool, default=False, help="EMA on the generated side: the batch enters with weight (1 - decay), so scale --fd_loss_weight accordingly")
parser.add_argument("--fd_norm_eps", type=float, default=0.01)
# --- the paper's FD loss (fdr6/fd_loss.py): decoded pixels through frozen judges
parser.add_argument("--fd_space", type=str, default="encoder", choices=["encoder", "judges", "both"], help="encoder: FDLoss on the model's own encoder latents (weight --fd_loss_weight); judges: MultiJudgeFDLoss on the decoded pixels through the FDr-6 judges (the paper; weight --fd_judge_weight, falling back to --fd_loss_weight); both: the two terms at once, each with its own weight. Independent of the score path's own FD (score loss.fd_weight)")
parser.add_argument("--fd_judge_weight", type=float, default=None, help="weight of the judges-space FD term (default: --fd_loss_weight)")
parser.add_argument("--fd_judge_start_epoch", type=int, default=None, help="start epoch of the judges-space FD term (default: --fd_loss_start_epoch)")
parser.add_argument("--fd_judge_seed_size", type=int, default=50000, help="number of generated samples the judges-space FD seeds its EMA statistics from before the loss opens (50000 in the paper)")
parser.add_argument("--fd_judges", type=str, nargs="+", default=["siglip", "mae", "inception"], help="FDr-6 encoder labels; the default is the paper's FD-SIM combination")
parser.add_argument("--fd_judge_weights", type=float, nargs="+", default=None, help="per-judge loss weights (default 1.0 each)")
parser.add_argument("--fd_judge_pool", type=str, default="cls", choices=["cls", "avg"], help="cls: CLS / attention-pooled feature (the paper, what FDr-6 scores); avg: mean patch token (timm ViT judges only)")
parser.add_argument("--fd_use_eigvalsh", type=str2bool, default=True, help="trace term via eigvalsh of the symmetric product (the paper's --fd_eigvalsh); False: eigvals of sigma @ sigma_ref")
parser.add_argument("--fd_stats_used_from", type=str, default="auto", choices=["auto", "jit", "adm", "extr", "rand-50k", "full"], help="source tag of the FDr-6 stats files; auto follows fdr6/eval_fdr6.py (extr for the small datasets, full for imagenet when extracted, else rand-50k)")
parser.add_argument("--fid_stats_dir", type=str, default="fid_stats")
parser.add_argument("--inception_weight_path", type=str, default="workspace/pretrained/fid_pretrained_models/weights-inception-2015-12-05-6726825d.pth")
# --- score matching loss
parser.add_argument("--use_score", type=str2bool, default=False)
parser.add_argument("--score_set", type=str, nargs="+", default=[], metavar="KEY=VALUE", help="e.g. --score_set score.optimizer.lr=1e-4 loss.weight=0.5 loss.space=hidden")
# --- debugging
parser.add_argument("--detect_anomaly", type=str2bool, default=False)
parser.add_argument("--skip_nonfinite_grad", type=str2bool, default=True)
parser.add_argument("--max_consecutive_skips", type=int, default=50, help="after this many consecutive skips with nan gradient, abort training")
# --- resume
parser.add_argument("--load_from", type=str, default=None, help="initialize model/EMA weights from a checkpoint and train from epoch 0; ignored when resuming")
parser.add_argument("--resume_from", type=str, default=None, help="resume training from a checkpoint (weights, EMA, epoch); takes precedence over --load_from")
parser.add_argument("--init_from", type=str, default="scratch", choices=["scratch", "resume"])
parser.add_argument("--auto_resume", type=str2bool, default=True)
parser.add_argument("--override_model_with_ema", type=str2bool, default=False)
parser.add_argument("--override_ema_with_model", type=str2bool, default=False)
# --- lpips
parser.add_argument("--perceptual_ckpt_path", type=str, default="pretrained/lpips")
parser.add_argument("--perceptual_loss_chns_range", type=int, nargs=2, default=None, metavar=("LIDX", "RIDX"))
parser.add_argument("--perceptual_image_size", type=int, default=None)
parser.add_argument("--perceptual_convnext_aug", type=str, default=None, metavar="FLIP-TRANS-SCALE_MIN-SCALE_MAX")
parser.add_argument("--perceptual_convnext_loss_type", type=str, default="mse", choices=["mse", "cosine"])
parser.add_argument("--perceptual_convnext_loss_reduction", type=str, default="mean", choices=["mean", "none"])
# --- wrap up
cli_args = parser.parse_args()
if cli_args.device_type is None:
    cli_args.device_type = get_device_type()
# fmt: on
# -----------------------------------------------------------------------------


def set_exp_name(args):
    encoder = args.load_pretrained_encoder or args.vit_enc_model_size
    parts = [
        "sphere2",
        encoder,
        args.vit_dec_model_size,
        args.dataset_name,
        f"{args.image_size}px",
    ]
    if args.run_str:
        parts.append(args.run_str)
    return "-".join(parts)


def main(args):
    if torch.cuda.is_available():
        if args.blas_library is not None:
            torch.backends.cuda.preferred_blas_library(args.blas_library)
        selected_blas = torch.backends.cuda.preferred_blas_library()
        print(
            f"BLAS backend: override={args.blas_library} selected={selected_blas}",
            flush=True,
        )

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # The cuDNN SDPA backend has a backward bug that returns NaN gradients
    # (ScaledDotProductCudnnAttentionBackward0) on some shapes/cuDNN versions,
    # which here showed up in the frozen-encoder re-encode and corrupted the
    # whole step. Drop it from the SDPA dispatch pool so F.scaled_dot_product_
    # attention falls back to flash / mem-efficient / math (correct backward).
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    # setup dirs
    job_dir = set_exp_name(args)
    exp_dir = osp.join(args.dev_dir, args.out_dir, job_dir)

    # various inits, derived attributes, I/O setup
    if args.launch_exec == "torchrun":
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

    elif args.launch_exec == "mpiexec":
        from mpi4py import MPI

        ddp_rank = MPI.COMM_WORLD.Get_rank()
        ddp_local_rank = int(os.environ.get("PALS_LOCAL_RANKID", 0))
        ddp_world_size = MPI.COMM_WORLD.Get_size()

        os.environ["RANK"] = str(ddp_rank)
        os.environ["WORLD_SIZE"] = str(ddp_world_size)

    device = set_device(args.device_type, ddp_local_rank)
    dist.init_process_group(
        backend=get_dist_backend(args.device_type),
        # a device index is only meaningful for an accelerator backend
        device_id=ddp_local_rank if args.device_type != "cpu" else None,
        timeout=datetime.timedelta(hours=2),
    )
    ddp_rank0 = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed

    # seed
    seed = 99  # alita
    torch.manual_seed(seed + seed_offset)
    if args.device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type=args.device_type, dtype=ptdtype)
        if args.device_type in ("cuda", "xpu")
        else nullcontext()
    )

    # directories
    if ddp_rank0:
        os.makedirs(exp_dir, exist_ok=True)
    args.vis_dir = os.path.join(exp_dir, "vis")  # visualizations
    args.ckpt_dir = os.path.join(exp_dir, "ckpt")  # checkpoints
    for d in [args.vis_dir, args.ckpt_dir]:
        if not ddp_rank0:
            continue
        os.makedirs(d, exist_ok=True)

    # save the scheduler job info (job id + name) for later reference: SLURM
    # first (SLURM_JOB_ID_CAPTURED is for launchers that unset SLURM_JOB_ID),
    # then PBS
    slurm_job_id = os.environ.get("SLURM_JOB_ID_CAPTURED") or os.environ.get(
        "SLURM_JOB_ID"
    )
    pbs_job_id = os.environ.get("PBS_JOBID")
    if ddp_rank0 and (slurm_job_id or pbs_job_id):
        if slurm_job_id:
            job_info = {
                "scheduler": "slurm",
                "job_id": slurm_job_id,
                "job_name": os.environ.get("SLURM_JOB_NAME", ""),
            }
        else:
            job_info = {
                "scheduler": "pbs",
                "job_id": pbs_job_id,
                "job_name": os.environ.get("PBS_JOBNAME", ""),
            }
        with open(os.path.join(exp_dir, "job.info"), "w") as f:
            json.dump(job_info, f, indent=4)

    dist.barrier()

    # logger
    if args.use_wandb and ddp_rank0:
        wandb.login(key=args.wandb_key, host="https://api.wandb.ai", relogin=True)
        wandb.init(
            name=job_dir,
            project=args.wandb_project,
            config=vars(args),
            entity=args.wandb_entity,
            dir=args.wandb_dir,
        )

    setup_logging(output_path=exp_dir, rank=ddp_rank)
    logger.info(
        f"local rank {ddp_local_rank} initialized with global rank {ddp_rank} / {ddp_world_size}"
    )
    logger.info(f"logging to {exp_dir}")

    # for detailed logging
    log_training_path = os.path.join(exp_dir, "log.jsonl")

    # prepare various components before training starts
    args.num_classes = {
        "food-101": 101,
        "flowers-102": 102,
        "animal-faces": 3,
        "cifar-10": 10,
        "cifar-100": 100,
        "imagenet": 1000,
    }[args.dataset_name]

    if args.dataset_name == "imagenet":
        # ids follow val.json / folder_to_id_to_label.json, not the
        # standard sorted-wnid ImageNet ordering

        # fmt: off
        args.class_of_interest = [
            994,  # macaw
            561,  # jay
            140,  # wreck
            849,  # persian cat
            464,  # jack-o-lantern
             19,  # cheeseburger
            167,  # killer whale
            535,  # golden retriever
            729,  # lesser panda
            682,  # valley
            335,  # balloon
             41,  # arctic fox
            456,  # ice cream
            423,  # pineapple
             80,  # head cabbage
            920,  # tractor
            134,  # jellyfish
            675,  # golf ball
              3,  # anemone fish
            942,  # timber wolf
            822,  # spider web
            364,  # orange
            572,  # matchstick
            186,  # tennis ball
            792,  # flamingo
        ]
        # fmt: on

    dataset_cls = get_dataset_cls(args.dataset_name)

    if args.image_size <= 64:
        args.interp_mode = "nearest"

    interp_mode = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "nearest": transforms.InterpolationMode.NEAREST,
    }[args.interp_mode]

    # default patch size (for the case of latent_resolution = "low")
    if args.image_size in [32, 64]:
        args.patch_size = 4
    elif args.image_size in [128]:
        args.patch_size = 8
    elif args.image_size in [256]:
        args.patch_size = 16
    elif args.image_size in [512]:
        args.patch_size = 32

    if args.latent_resolution == "low":
        args.patch_size = args.patch_size // 1
    elif args.latent_resolution == "high":
        args.patch_size = args.patch_size // 2
    elif args.latent_resolution == "super_high":
        args.patch_size = args.patch_size // 4
    else:
        raise ValueError(f"unknown latent resolution: {args.latent_resolution}")

    # calc token channels
    args.latent_resolution = args.image_size // args.patch_size
    args.token_channels = int(
        3 * args.image_size**2 / args.latent_resolution**2 / args.compression_ratio
    )

    # adjust vars if using pretrained encoders
    if (
        args.load_pretrained_encoder is not None
        and not args.pretrained_encoder_fix_compression_ratio
    ):
        args.token_channels = get_token_channels(args.load_pretrained_encoder)
        logger.info(
            f"token_channels set to {args.token_channels} "
            f"for pretrained encoder {args.load_pretrained_encoder}"
        )

    L = args.latent_resolution**2 * args.token_channels
    logger.info(
        f"latent resolution L: {L} = {args.latent_resolution} x {args.latent_resolution} x {args.token_channels}"
        f" (patch size: {args.patch_size}, input size: {args.image_size})"
    )

    # dump config
    config_path = os.path.join(exp_dir, "cfg.json")
    if ddp_rank0:
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    # build data loader and sampler
    train_loader, test_loader, vis_loader, train_sampler = create_loader(
        dataset_cls,
        osp.join(args.dev_dir, args.data_dir, args.dataset_name),
        args.image_size,
        args.patch_size,
        max_samples=args.max_samples,
        interp_mode=interp_mode,
        rot_degrees=args.rot_degrees,
        crop_mode=args.crop_mode,
        flip_image=args.flip_image,
        color_jitter=args.color_jitter,
        extra_padding=args.extra_padding,
        concat_train_val_splits=args.concat_train_val_splits,
        ddp_world_size=ddp_world_size,
        ddp_rank=ddp_rank,
        batch_size_per_rank=args.batch_size_per_rank,
        num_workers=args.num_workers,
        load_from_zip=args.load_from_zip,
    )
    logger.info(f"training dataset: {train_loader.dataset}")

    total_samples = len(train_loader.dataset)
    logger.info(f"total training samples: {total_samples}")

    # learning rate decay scheduler (cosine with warmup)
    effective_batch_size = (
        args.batch_size_per_rank * ddp_world_size * args.grad_accum_steps
    )

    # rescale lr if needed
    if effective_batch_size != args.batch_size and args.rescale_lr_with_batch_size:
        # sqrt scaling for adamw
        rescaler = math.sqrt(effective_batch_size / args.batch_size)
        args.learning_rate *= rescaler
        args.min_lr *= rescaler
        logger.info(
            f"effective batch size: {effective_batch_size}, "
            f"learning rate is rescaled to {args.learning_rate} (min_lr = {args.min_lr})"
        )

    # dataset-centric step calculations
    # : one step = one optimizer update
    steps_per_epoch = math.floor(total_samples / effective_batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs / args.epochs * total_steps)

    # init these up here, can override later if we want to resume from a checkpoint
    cur_epoch = 0

    # build loss

    # apply CLI overrides to the score-matching config before it is consumed.
    # --lat_con_use_diffaug also reads the score augment sub-config (it reuses
    # the same DiffAug knobs for the lat-con re-encode), so apply the overrides
    # for that path too even when the score loss itself is off.
    if args.use_score or args.lat_con_use_diffaug:
        apply_dotted_overrides(score_config, args.score_set, log_mark="score_config")

    # the FD loss judges in the encoder space: the CLS token keeps the
    # backbone width even when the latent is projected to token_channels
    fd_dim = args.token_channels
    # --fd_space picks which of the two top-level FD terms are on; the score
    # path's own FD (score loss.fd_weight, on its convnext / dino featurizer)
    # is separate and can be on alongside either
    fd_enc_weight = args.fd_loss_weight if args.fd_space in ("encoder", "both") else 0.0
    fd_judge_weight = 0.0
    if args.fd_space in ("judges", "both"):
        fd_judge_weight = (
            args.fd_loss_weight
            if args.fd_judge_weight is None
            else args.fd_judge_weight
        )
    if fd_enc_weight > 0 and args.fd_pool == "cls":
        assert args.load_pretrained_encoder is not None and (
            "dinov" in args.load_pretrained_encoder
        ), "--fd_pool cls needs a pretrained DINO encoder (--load_pretrained_encoder)"
        fd_dim = get_token_channels(args.load_pretrained_encoder)

    # the paper's FD loss: frozen FDr-6 judges on the decoded pixels against
    # the fixed reference statistics under workspace/fid_stats (see
    # fdr6/fd_loss.py). Built here because it needs the dataset / stats paths.
    fd_judge_loss = None
    if fd_judge_weight > 0:
        from fdr6.fd_loss import MultiJudgeFDLoss

        fd_stats_source = args.fd_stats_used_from
        if fd_stats_source == "auto":
            # same rule as fdr6/eval_fdr6.py: extr for the small datasets,
            # the full 1.28M train split for imagenet when extracted, else rand-50k
            from fdr6.fdr6_utils import get_encoder_specs, stats_file_path

            if args.dataset_name in [
                "cifar-10",
                "cifar-100",
                "animal-faces",
                "flowers-102",
            ]:
                fd_stats_source = "extr"
            elif all(
                osp.exists(
                    stats_file_path(
                        osp.join(args.dev_dir, args.fid_stats_dir),
                        "full",
                        args.dataset_name,
                        args.image_size,
                        s,
                    )
                )
                for s in get_encoder_specs(args.image_size)
            ):
                fd_stats_source = "full"
            else:
                fd_stats_source = "rand-50k"
        fd_judge_loss = MultiJudgeFDLoss(
            judges=args.fd_judges,
            image_size=args.image_size,
            dataset_name=args.dataset_name,
            stats_dir=osp.join(args.dev_dir, args.fid_stats_dir),
            stats_source=fd_stats_source,
            device=device,
            inception_weight_path=args.inception_weight_path,
            judge_weights=args.fd_judge_weights,
            pool=args.fd_judge_pool,
            seed_size=args.fd_judge_seed_size,
            ema_beta=args.fd_ema_decay,
            norm_eps=args.fd_norm_eps,
            use_eigvalsh=args.fd_use_eigvalsh,
        )

    loss_fn = G1Loss(
        latent_spherify_mode=args.spherify_mode,
        perceptual_ckpt_path=osp.join(args.dev_dir, args.perceptual_ckpt_path),
        perceptual_loss_chns_range=(
            tuple(args.perceptual_loss_chns_range)
            if args.perceptual_loss_chns_range is not None
            else None
        ),
        perceptual_image_size=args.perceptual_image_size,
        perceptual_convnext_aug=args.perceptual_convnext_aug,
        perceptual_convnext_loss_type=args.perceptual_convnext_loss_type,
        perceptual_convnext_loss_reduction=args.perceptual_convnext_loss_reduction,
        distance_loss_type=args.distance_loss_type,
        distance_weight=args.pix_recon_dist_loss_weight,
        distance_apply_regime=args.pix_recon_dist_apply_regime,
        perceptual_weight=args.pix_recon_perc_loss_weight,
        latent_consistency_weight=args.lat_con_loss_weight,
        latent_consistency_apply_regime=args.lat_con_apply_regime,
        weight_anchor_loss=args.weight_anchor_loss,
        weight_anchor_cutoff_params=args.weight_anchor_cutoff_params,
        use_score=args.use_score,
        score_config=score_config if args.use_score else None,
        score_device=device,
        score_num_classes=args.num_classes if args.cond_generator else 0,
        fd_weight=fd_enc_weight,
        fd_start_epoch=args.fd_loss_start_epoch,
        fd_dim=fd_dim,
        fd_pool=args.fd_pool,
        fd_apply_regime=args.fd_apply_regime,
        fd_ema_decay=args.fd_ema_decay,
        fd_use_clean_ema=args.fd_use_clean_ema,
        fd_use_noisy_ema=args.fd_use_noisy_ema,
        fd_norm_eps=args.fd_norm_eps,
        fd_judge_loss=fd_judge_loss,
        fd_judge_weight=fd_judge_weight,
        fd_judge_start_epoch=args.fd_judge_start_epoch,
        grad_probe_interval=args.log_interval if args.grad_probe else 0,
        ptdtype=ptdtype,
    )
    loss_fn.to(device=device, memory_format=torch.channels_last)
    logger.info(loss_fn)

    # build model
    model = build_model(args, for_train=True)
    model.to(device=device, memory_format=torch.channels_last)

    # optional differentiable augmentation for the re-encode.
    #
    # model.score_augs is a model-side hook that augments the SHARED re-encode
    # inside G1.forward (the decoded images x, before enc(x)). Only the lat-con
    # loss wants it now, via --lat_con_use_diffaug, to decohere the
    # position-locked enc-dec echo -- the score loss reads pixels through its
    # own frozen featurizer in both of its spaces, so it never consumes the
    # re-encode and takes its aug on the score loss itself
    # (loss_fn.score_match.pixel_augs), handled separately below.
    score_aug_cfg = score_config["score"].get("augment", {})

    # the lat-con branch opts in explicitly, regardless of the score loss
    _lat_con_wants_hook = args.lat_con_loss_weight > 0 and args.lat_con_use_diffaug
    # the pixel score spaces (dino/convnext) need their own pixel aug (built
    # below)
    _dino_wants_aug = (
        args.use_score
        and score_aug_cfg.get("enabled", False)
        and loss_fn.score_match is not None
        and loss_fn.score_match.space in ("dino", "convnext")
    )

    score_augs = None
    if _lat_con_wants_hook or _dino_wants_aug:
        from sphere.score.diffaug import DiffAug

        aug_type = score_aug_cfg.get("type", "diffaug")
        if aug_type == "flip":
            # shorthand: diffaug with only the flip module active. choosing
            # type="flip" implies flip on, so default to full flip when
            # flip_prob is left at its 0 default.
            _fp = float(score_aug_cfg.get("flip_prob", 0.0))
            score_augs = DiffAug(flip_prob=_fp if _fp > 0 else 1.0)
        else:
            score_augs = DiffAug(
                brightness_prob=float(score_aug_cfg.get("brightness_prob", 0.0)),
                saturation_prob=float(score_aug_cfg.get("saturation_prob", 0.0)),
                contrast_prob=float(score_aug_cfg.get("contrast_prob", 0.0)),
                translation_prob=float(score_aug_cfg.get("translation_prob", 0.0)),
                flip_prob=float(score_aug_cfg.get("flip_prob", 0.0)),
                cutout_prob=float(score_aug_cfg.get("cutout_prob", 0.0)),
                brightness_strength=float(
                    score_aug_cfg.get("brightness_strength", 0.5)
                ),
                saturation_strength=float(
                    score_aug_cfg.get("saturation_strength", 1.0)
                ),
                contrast_strength=float(score_aug_cfg.get("contrast_strength", 0.5)),
                translation_ratio=float(score_aug_cfg.get("translation_ratio", 0.125)),
                translation_padding=str(
                    score_aug_cfg.get("translation_padding", "zeros")
                ),
                cutout_ratio=float(score_aug_cfg.get("cutout_ratio", 0.5)),
            )

    # ---- attach to the model-side re-encode hook (the lat-con branch) ----
    # The hook augments v = enc(aug(x_fake)); the pointwise lat-con target
    # (z_clean) is un-augmented, and with --lat_con_use_diffaug that decohering
    # of the enc-dec echo is INTENDED (see G1.forward).
    if _lat_con_wants_hook:
        model.score_augs = score_augs
        logger.info(f"re-encode augmentation (model.score_augs): {score_augs}")

    # ---- pixel score spaces (dino/convnext): attach the aug to the score
    # loss itself ----
    # These spaces extract fake features from pixels inside prepare_feats (not
    # via the model's re-encode), so they need the aug attached to the score
    # loss itself. Fake-only (asymmetric) aug is valid for flip alone (the
    # loader's flip_image closes the real distribution under it); DiffAug
    # (color/cutout/translation) needs symmetric=True so BOTH sides pass
    # through the aug, GAN style, or the generator pre-compensates the aug
    # (leak).
    if _dino_wants_aug:
        score_aug_symmetric = bool(score_aug_cfg.get("symmetric", False))
        # flip is loader-closed on the real side, so a flip-only aug is safe
        # fake-only (asymmetric); any other active module needs symmetric=True
        # in the dino space to avoid a fake-only leak.
        _nonflip_active = any(
            float(score_aug_cfg.get(f"{m}_prob", 0.0)) > 0
            for m in (
                "brightness",
                "saturation",
                "contrast",
                "translation",
                "cutout",
            )
        )
        if not _nonflip_active or score_aug_symmetric:
            loss_fn.score_match.pixel_augs = score_augs
            loss_fn.score_match.pixel_augs_symmetric = score_aug_symmetric
            logger.info(
                f"score-matching dino-space pixel aug: {score_augs} "
                f"(symmetric={score_aug_symmetric})"
            )
        else:
            logger.warning(
                f"score aug {score_augs} NOT attached to the dino "
                "space: fake-only non-flip DiffAug leaks; set "
                "score.augment.symmetric=True to enable it there"
            )

    logger.info(model)

    ema_model = None
    if args.use_ema:
        ema_model_start_decay = 0.95**args.grad_accum_steps
        ema_model_end_decay = args.ema_model_decay**args.grad_accum_steps

        if not args.use_ema_scheduler:
            ema_model_start_decay = ema_model_end_decay

        logger.info(
            f"EMA model enabled with decay {ema_model_start_decay} to {ema_model_end_decay} (adjusted for grad accumulation)"
        )
        ema_model = ModuleEMA(model, decay=ema_model_start_decay)
        ema_model.eval().requires_grad_(False)

    if args.auto_resume:
        # newest checkpoint that is actually complete: a save cut short by the
        # slurm time limit is skipped in favour of the one before it
        latest = find_latest_valid_ckpt(args.ckpt_dir)
        if latest is not None:
            args.init_from = "resume"
            args.resume_from = latest
            logger.info(f"auto resume from {args.resume_from}")

    if args.resume_from is not None:
        args.init_from = "resume"
        if not is_ckpt_valid(args.resume_from):
            raise RuntimeError(
                f"resume checkpoint {args.resume_from} is broken or incomplete"
            )

    ckpt_path = args.load_from
    if args.init_from == "resume":
        ckpt_path = args.resume_from

    checkpoint = load_ckpt(
        model,
        ckpt_path=ckpt_path,
        ema_model=ema_model,
        strict=False,
        override_model_with_ema=args.override_model_with_ema,
        verbose=True,
        return_ckpt=True,
    )
    if args.init_from == "resume":
        cur_epoch = checkpoint["epoch"] + 1
        if args.use_ema and args.override_ema_with_model:
            ema_model = ModuleEMA(model, decay=ema_model_start_decay)
            ema_model.eval().requires_grad_(False)

    # build optimizer
    params_enc_w_decay = []
    params_dec_w_decay = []
    params_enc = []  # w/o decay
    params_dec = []  # w/o decay

    exclude = lambda name, p: (
        p.ndim < 2
        or any(
            keyword in name
            for keyword in [
                "ln",
                "bias",
                "embedding",
                "norm",
                "embed",
                "token",
            ]
        )
    )

    max_n_len = max([len(n) for n, _ in model.named_parameters()])
    for n, p in model.named_parameters():
        if not p.requires_grad:
            logger.info(
                f"p.requires_grad: {str(p.requires_grad):<5}, "
                f"param: {n:<{max_n_len}}, "
                f"{p.shape}"
            )
            continue
        else:
            p.requires_grad = True

        if not p.is_contiguous():
            p.data = p.data.contiguous()

        if exclude(n, p):
            with_decay = False
            if "encoder" in n:
                params_enc.append(p)
            else:
                params_dec.append(p)
        else:
            with_decay = True
            if "encoder" in n:
                params_enc_w_decay.append(p)
            else:
                params_dec_w_decay.append(p)

        log_str = (
            f"p.requires_grad: {str(p.requires_grad):<5}, "
            f"param: {n:<{max_n_len}}, "
            f"decay: {str(with_decay):<5}, "
            f"{str(p.shape):<30}"
        )
        logger.info(log_str)

    # fmt: off
    optim_params = [
        {"params": params_enc, "weight_decay": 0.0, "pg_name": "encoder"},
        {"params": params_dec, "weight_decay": 0.0, "pg_name": "decoder"},
        {"params": params_enc_w_decay, "weight_decay": args.weight_decay, "pg_name": "encoder"},
        {"params": params_dec_w_decay, "weight_decay": args.weight_decay, "pg_name": "decoder"},
    ]
    # fmt: on

    optimizer_cls = {
        "adamw": torch.optim.AdamW,
    }[args.optimizer_name]

    optimizer = optimizer_cls(
        optim_params,
        lr=args.learning_rate,
        betas=(0.90, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    logger.info(optimizer)

    # stash the score-matching net + optimizer state so it survives the
    # checkpoint free below; consumed after score_ddp / score_optimizer are built.
    score_resume_state = None
    if (
        checkpoint is not None
        and loss_fn.score_match is not None
        and args.init_from == "resume"
    ):
        score_resume_state = {
            "score_match": checkpoint.get("score_match"),
        }
        if not score_config["loss"].get("skip_load_optim_state", False):
            score_resume_state["score_optimizer"] = checkpoint.get("score_optimizer")
        else:
            logger.warning(
                "loss.skip_load_optim_state=True: discarding the score-matching "
                "optimizer state that IS present in the checkpoint; the critic "
                "restarts with zeroed Adam moments"
            )

    # ---- resume the FD loss running statistics (moments) ----
    # two slots: "fd_loss" (encoder space) and "fd_judge_loss" (judges space).
    # Checkpoints written before the split stored a judges-only run under
    # "fd_loss"; fall back to that key when there is no encoder-space term.
    if checkpoint is not None and args.init_from == "resume":
        _fd_slots = []
        if getattr(loss_fn, "fd_loss", None) is not None:
            _fd_slots.append(("fd_loss", loss_fn.fd_loss, ["fd_loss"]))
        if getattr(loss_fn, "fd_judge_loss", None) is not None:
            _keys = ["fd_judge_loss"]
            if getattr(loss_fn, "fd_loss", None) is None:
                _keys.append("fd_loss")
            _fd_slots.append(("fd_judge_loss", loss_fn.fd_judge_loss, _keys))
        for _name, _mod, _keys in _fd_slots:
            fd_sd = next(
                (checkpoint[k] for k in _keys if checkpoint.get(k) is not None), None
            )
            if fd_sd is not None:
                try:
                    _mod.load_state_dict(fd_sd)
                    logger.info(f"resumed {_name} statistics from the checkpoint")
                except RuntimeError as e:
                    logger.warning(
                        f"{_name} state in the checkpoint is incompatible, restarting its statistics: {e}"
                    )
            else:
                logger.warning(
                    f"no {_name} state in the checkpoint: its statistics restart from zero"
                )

    # free up memory
    if checkpoint is not None:
        checkpoint = None

    # count params
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"parameters of model: {total_params/1e6:.2f}M")
    logger.info(f"parameters to train: {train_params/1e6:.2f}M")

    # compile before ddp wrapping
    if args.compile_model:
        logger.info("compiling encoder/decoder submodules...")
        torch._dynamo.config.optimize_ddp = True
        torch._dynamo.config.capture_scalar_outputs = True
        # The outer G.forward keeps cross-step cache state for consistency losses
        # and performs python-side logging, which is brittle under full-module
        # compilation. Compile the transformer submodules instead.
        model.encoder = torch.compile(model.encoder, dynamic=True)
        model.decoder = torch.compile(model.decoder, dynamic=True)

    # wrap the model in DDP
    # a device index is only meaningful for an accelerator backend
    ddp_device_ids = [ddp_local_rank] if args.device_type != "cpu" else None
    model = DDP(
        model,
        device_ids=ddp_device_ids,
        static_graph=True,
        broadcast_buffers=False,
        bucket_cap_mb=args.ddp_bucket_cap_mb,
    )

    # compress grad all-reduce to bf16 to halve comm volume. plain bf16 sums in
    # bf16 (precision degrades with world size); bf16_ef carries the per-rank
    # quantization residual into the next step to recover most of that accuracy.
    def _register_grad_comm_hook(ddp_module, what):
        """Apply --grad_comm_dtype to one DDP wrapper.

        Every DDP wrapper in the job pays its own all-reduce, so each one needs
        its own registration -- the flag is a property of the run, not of a
        module. The error-feedback state keys residuals by BUCKET INDEX, and
        bucket indices restart at 0 in every wrapper, so each wrapper must get a
        fresh state or two different models would read each other's residuals.
        """
        if args.grad_comm_dtype == "fp32":
            return
        if args.grad_comm_dtype == "bf16":
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
                bf16_compress_hook,
            )

            ddp_module.register_comm_hook(state=None, hook=bf16_compress_hook)
        else:  # bf16_ef
            ddp_module.register_comm_hook(
                state=_BF16ErrorFeedbackState(), hook=bf16_ef_comm_hook
            )
        n = sum(p.numel() for p in ddp_module.parameters() if p.requires_grad)
        logger.info(
            f"DDP gradient all-reduce compression [{what}]: "
            f"{args.grad_comm_dtype}, {n / 1e6:.2f}M params "
            f"({n * 4 / 1e6:.0f} MB -> {n * 2 / 1e6:.0f} MB per step)"
        )

    _register_grad_comm_hook(model, "model")

    # apply activation checkpointing to transformer blocks for memory efficiency
    if args.use_activation_checkpointing:
        from sphere.layers import Block

        check_fn = lambda submodule: isinstance(submodule, Block)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        logger.info("activation checkpointing enabled for transformer layers")

    # training loop
    model_without_ddp = model.module
    model_without_ddp.ddp_rank = ddp_rank
    model_without_ddp.ddp_world_size = ddp_world_size

    # ---- score-matching nets: own DDP wrapper + optimizer ----
    # The score net(s) live inside loss_fn (so they move to device with it)
    # but are trained by a separate optimizer via DSM.
    # The generator-side surrogate is computed in loss_fn.forward (score nets
    # frozen there); here we only train the score nets on cached features.
    use_score = loss_fn.score_match is not None
    score_ddp = None
    score_optimizer = None
    score_averager = None  # set below iff loss.sync_interval > 1
    score_sync_interval = 1
    score_updates_done = 0
    if use_score:
        score_opt_cfg = score_config["score"]["optimizer"]
        # two param groups so the REAL and FAKE critic(s) can decay to
        # different cosine FLOORS (shared max lr + decay_epochs; see the loop).
        # The real target is stationary and converges early, so it decays to a
        # smaller floor; the fake critic tracks the moving generator, so its
        # floor stays live. Each group carries its own "min_lr".
        _sm = loss_fn.score_match
        _base_min_lr = float(score_opt_cfg.get("min_lr", score_opt_cfg["lr"]))

        def _grp_min_lr(key):
            v = score_opt_cfg.get(key, None)
            return float(v) if v is not None else _base_min_lr

        _real_min_lr = _grp_min_lr("real_min_lr")
        _fake_min_lr = _grp_min_lr("fake_min_lr")

        _score_groups = [
            {
                "params": [
                    p for p in _sm.real_score_nets.parameters() if p.requires_grad
                ],
                "min_lr": _real_min_lr,
                "anchor": float(_sm.real_start_epoch),
                "name": "real",
            },
            {
                "params": [
                    p for p in _sm.fake_score_nets.parameters() if p.requires_grad
                ],
                "min_lr": _fake_min_lr,
                "anchor": float(_sm.fake_start_epoch),
                "name": "fake",
            },
        ]
        score_optimizer = torch.optim.AdamW(
            _score_groups,
            lr=float(score_opt_cfg["lr"]),
            betas=tuple(score_opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=float(score_opt_cfg.get("weight_decay", 0.0)),
            fused=True,
        )
        logger.info(
            f"score-net lr floors: real={_real_min_lr:g}, fake={_fake_min_lr:g}"
        )
        # trainable = the denoisers only. The pixel featurizer is held in a
        # proxy tuple (see DinoFeaturizer / _ConvNextFeaturizerBase), so it is
        # deliberately absent from .parameters() -- counted separately here
        # because it still costs memory and a forward per step.
        score_match_total_params = sum(p.numel() for p in _sm.parameters())
        _feat = _sm.featurizer
        _feat_params = (
            sum(p.numel() for p in _feat.proxy[0].parameters())
            if _feat is not None
            else 0
        )
        logger.info(f"score matching enabled ({_sm})")
        logger.info(
            f"score model total params: {score_match_total_params/1e6:.2f}M trainable"
            + (
                f" (+ {_feat_params/1e6:.2f}M frozen featurizer)"
                if _feat_params
                else ""
            )
        )
        logger.info(
            f"score module active from epoch {_sm.start_epoch}; "
            f"DSM updates: real from epoch {_sm.real_start_epoch}, "
            f"fake from epoch {_sm.fake_start_epoch}"
        )
        logger.info(
            f"generator surrogate from epoch {_sm.gen_start_epoch} "
            f"(warmup {_sm.gen_warmup_shape} from epoch "
            f"{_sm.gen_warmup_start_epoch})"
        )
        logger.info(score_optimizer)

        # cosine lr decay for the score net (epoch units). Decay begins as soon
        # as a group starts training: every group carries its own anchor (its
        # own start epoch), and score_decay_anchor below is only the fallback.
        score_lr_max = float(score_opt_cfg["lr"])
        score_lr_min = float(score_opt_cfg.get("min_lr", score_lr_max))
        score_decay_epochs = int(score_opt_cfg.get("decay_epochs", 0))
        score_warmup_epochs = float(score_opt_cfg.get("warmup_epochs", 0))
        # warmup (if any) must fit inside the decay window, or the cosine
        # decay_ratio denominator (decay_steps - warmup_steps) goes <= 0.
        if score_decay_epochs > 0 and score_warmup_epochs > 0:
            assert score_warmup_epochs < score_decay_epochs, (
                f"score warmup_epochs ({score_warmup_epochs}) must be < "
                f"decay_epochs ({score_decay_epochs})"
            )
        # Grad-norm clip each denoiser independently. A single joint clip lets
        # one unstable (side, feature-layer) critic consume the entire norm
        # budget and suppress otherwise healthy critics. Kept as
        # (threshold, params) groups so each returned pre-clip norm also serves
        # as that group's non-finite guard signal. Together the groups cover
        # every trainable param in score_ddp (the featurizer is held in a proxy
        # tuple, see DinoFeaturizer).
        score_grad_clip = float(score_opt_cfg.get("grad_clip", 0.0))
        score_clip_groups = []
        for _nets in (_sm.real_score_nets, _sm.fake_score_nets):
            for _net in _nets:
                score_clip_groups.append(
                    (
                        score_grad_clip,
                        [p for p in _net.parameters() if p.requires_grad],
                    )
                )
        if score_grad_clip > 0:
            logger.info(
                f"score-net per-denoiser grad-norm clip: {score_grad_clip:g} "
                f"across {len(score_clip_groups)} groups"
            )
        score_decay_anchor = _sm.start_epoch
        if score_warmup_epochs > 0:
            logger.info(
                f"score-net lr warmup: 0 -> {score_lr_max:g} over "
                f"{score_warmup_epochs:g} epochs from epoch {score_decay_anchor}"
            )
        if score_decay_epochs > 0 and score_lr_min != score_lr_max:
            logger.info(
                f"score-net lr cosine decay: {score_lr_max:g} -> {score_lr_min:g} "
                f"over {score_decay_epochs} epochs from epoch {score_decay_anchor}"
            )

        # ---- resume score net weights + optimizer ----
        if score_resume_state is not None:
            score_sd = score_resume_state["score_match"]
            unused_keys = []
            missing_keys = []
            score_weights_loaded = False
            if score_sd is not None:
                # the score backbone can change across a resume (e.g. dino ->
                # convnext), which makes the checkpoint tensors incompatible in
                # shape/name; warn and keep the freshly-built net rather than
                # crash.
                try:
                    msgs = _sm.load_state_dict(score_sd, strict=False)
                except RuntimeError as e:
                    logger.warning(
                        "score-matching weights in checkpoint do not match the "
                        f"current score model ({e}); starting score net fresh"
                    )
                else:
                    logger.info(f"resumed score-matching weights: {msgs}")
                    unused_keys = msgs.unexpected_keys
                    missing_keys = msgs.missing_keys
                    score_weights_loaded = True
            else:
                logger.warning("no score-matching weights in checkpoint to resume")

            score_opt_sd = None
            if "score_optimizer" in score_resume_state:
                score_opt_sd = score_resume_state["score_optimizer"]
            if (
                score_opt_sd is not None
                and score_weights_loaded
                and len(unused_keys) == 0
                and len(missing_keys) == 0
            ):
                # group-count mismatch happens when the set of score param
                # groups changes across a resume; start the optimizer fresh
                # rather than crash.
                if len(score_opt_sd["param_groups"]) != len(
                    score_optimizer.param_groups
                ):
                    logger.warning(
                        "score-optimizer param-group count changed "
                        f"({len(score_opt_sd['param_groups'])} -> "
                        f"{len(score_optimizer.param_groups)}); "
                        "starting score optimizer fresh"
                    )
                else:
                    # load_state_dict takes the SAVED param-group metadata
                    # wholesale (only "params" is kept from the live groups), so
                    # it would silently reinstate the checkpoint's lr schedule
                    # and ignore this run's config. Snapshot the config-derived
                    # keys and put them back; only the optimizer STATE (moments,
                    # step counts) is meant to come from the checkpoint.
                    _sched_keys = (
                        "lr_max",
                        "min_lr",
                        "warmup_epochs",
                        "decay_epochs",
                        "anchor",
                        "name",
                    )
                    _sched = [
                        {k: pg[k] for k in _sched_keys if k in pg}
                        for pg in score_optimizer.param_groups
                    ]
                    score_optimizer.load_state_dict(score_opt_sd)
                    for pg, s in zip(score_optimizer.param_groups, _sched):
                        for k in _sched_keys:
                            pg.pop(k, None)
                        pg.update(s)
                    logger.info("resumed score-matching optimizer state")
            elif "score_optimizer" not in score_resume_state:
                pass  # already reported at stash time (skip_load_optim_state)
            else:
                logger.warning("no score-matching optimizer state in checkpoint")

            score_resume_state = None  # free up memory

        if bool(score_config["loss"].get("compile", False)):
            logger.info("compiling score-matching nets...")
            torch._dynamo.config.optimize_ddp = True
            torch._dynamo.config.capture_scalar_outputs = True
            _sm.real_score_nets = torch.nn.ModuleList(
                torch.compile(
                    net,
                    dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )
                for net in _sm.real_score_nets
            )
            _sm.fake_score_nets = torch.nn.ModuleList(
                torch.compile(
                    net,
                    dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )
                for net in _sm.fake_score_nets
            )

        score_ddp = DDP(_sm, device_ids=ddp_device_ids, static_graph=True)
        _register_grad_comm_hook(score_ddp, "score")

        # -- how often the critics synchronize (loss.sync_interval) ----------
        # Above 1 the DSM forward/backward runs under no_sync(): no gradient
        # all-reduce, each rank steps its own critic, and the ranks are
        # re-coupled by averaging the weights every N updates. See the config
        # for the trade and the measured numbers.
        #
        # no_sync() also suppresses DDP's per-forward buffer broadcast, which is
        # safe here: every score buffer is derived from an all-reduced statistic
        # (see _update_scale / _transport_weights), so the ranks agree on them
        # without rank 0 having to broadcast.
        score_sync_interval = max(1, int(score_config["loss"].get("sync_interval", 1)))
        if score_sync_interval > 1:
            score_averager = ParamAverager(_sm.parameters())
            logger.info(
                f"score-net rank sync every {score_sync_interval} DSM updates "
                f"by parameter averaging (no gradient all-reduce): "
                f"{score_averager}"
            )

    global_step = 0

    if args.init_from == "resume":
        global_step = steps_per_epoch * cur_epoch

    for _use_ema in [False, True]:

        if not args.use_ema:
            _use_ema = False

        visualize(
            vis_loader,
            model_without_ddp,
            ddp_rank,
            epoch=cur_epoch if args.init_from == "resume" else 0,
            cfg=args.cfg,
            cfg_position=args.cfg_position,
            class_of_interest=args.class_of_interest,
            forward_steps=args.forward_steps,
            use_ema_model=_use_ema,
            ema_model=ema_model,
            save_dir=args.vis_dir,
            device=device,
            ctx=autocast_ctx,
        )

    # scheduler for lr
    get_lr = partial(
        cosine_scheduler,
        warmup_steps=warmup_steps,
        decay=args.decay_lr,
        decay_steps=total_steps,
    )

    train_iterator = cycle(train_loader)

    # optional profiler: trace a few steps on rank0, print a breakdown, then
    # keep training normally (overhead is limited to the first few iterations).
    prof = None
    profile_total = 0
    if args.profile and ddp_rank0:
        from torch.profiler import (
            profile,
            schedule,
            ProfilerActivity,
            tensorboard_trace_handler,
        )

        profile_total = args.profile_wait + args.profile_warmup + args.profile_active
        profile_activities = [ProfilerActivity.CPU]
        if args.device_type == "cuda":
            profile_activities.append(ProfilerActivity.CUDA)
        elif args.device_type == "xpu":
            profile_activities.append(ProfilerActivity.XPU)
        prof = profile(
            activities=profile_activities,
            schedule=schedule(
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(osp.join(exp_dir, "profiler")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        prof.start()
        logger.info(
            f"profiler enabled: wait={args.profile_wait} warmup={args.profile_warmup} "
            f"active={args.profile_active}; trace -> {osp.join(exp_dir, 'profiler')}"
        )

    logger.info(
        f"training starts at epoch {cur_epoch} and global step {global_step} 🚀"
    )
    checked_dead_params = False  # one-shot guard, see the check after backward
    # DDP + static_graph=True arms its reducer on the FIRST backward of the
    # process (a delayed all-reduce queued by _DDPSink); running that backward
    # inside no_sync() trips `expect_autograd_hooks_ INTERNAL ASSERT FAILED`
    # in reducer.cpp, which is where every grad_accum_steps > 1 DDP run used
    # to die at local_step 0. The first micro-step therefore always syncs (the
    # same guard the score critic has via score_updates_done); the averaged
    # partial grad is identical on every rank, so summing it with the later
    # micro-steps gives the same window gradient.
    did_first_backward = False
    consecutive_skips = 0  # non-finite-grad tripwire, see the guard after backward
    epoch = cur_epoch - 1  # ensure `epoch` is bound even if the loop below is empty
    for epoch in range(cur_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        iters_per_rank = total_samples // ddp_world_size // args.batch_size_per_rank
        if args.max_iters_per_epoch > 0:
            iters_per_rank = min(iters_per_rank, args.max_iters_per_epoch)
        # a whole number of accumulation windows per epoch: a trailing partial
        # window would either step with fewer micro-batches at the full 1/N
        # scale or be discarded by the zero_grad at the next epoch start, and
        # its global_step would spill into the next epoch's first step.
        iters_per_rank -= iters_per_rank % args.grad_accum_steps

        for local_step in range(iters_per_rank):
            t0 = time.perf_counter()
            data = next(train_iterator)
            dd = time.perf_counter() - t0

            global_step = epoch * steps_per_epoch + local_step // args.grad_accum_steps
            global_step = min(global_step, total_steps - 1)

            if local_step % args.grad_accum_steps == 0:
                lr = get_lr(args.learning_rate, args.min_lr, global_step)
                encoder_lr_scaler = encoder_lr_scaler_scheduler(
                    args.encoder_lr_scaler,
                    args.encoder_lr_scaler_min,
                    epoch,
                    args.encoder_lr_scaler_decay_epochs,
                )
                for pg in optimizer.param_groups:
                    pg_lr = lr
                    if "encoder" in pg["pg_name"]:
                        pg_lr *= encoder_lr_scaler
                    pg["lr"] = pg_lr

            imgs, clss = data[:2]  # FIXME: ignore the rest
            imgs = imgs.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )  # [-1, 1]
            clss = clss.to(device, non_blocking=True)

            # iters_per_rank is a multiple of grad_accum_steps (see above), so
            # the last micro-step of the epoch always closes a window
            is_accumulating = (local_step + 1) % args.grad_accum_steps != 0
            sync_ctx = (
                model.no_sync()
                if is_accumulating and did_first_backward
                else nullcontext()
            )

            # micro-step units on both sides: steps_per_epoch counts optimizer
            # steps, so with accumulation the old `epoch * steps_per_epoch +
            # local_step` repeated seeds (and the angle / noise draws) across
            # epochs
            model_without_ddp.step_seed = seed + epoch * iters_per_rank + local_step

            # forward
            with sync_ctx:

                with autocast_ctx:
                    pixels, hidden_states, latents = model(
                        imgs,
                        clss,
                        ema_model=ema_model if args.use_ema else None,
                    )

                loss = loss_fn(
                    input=pixels,
                    target=imgs,
                    epoch=epoch,
                    step=local_step,
                    hidden_states=hidden_states,
                    latent_list=latents,
                    autocast_ctx=autocast_ctx,
                    class_labels=clss,
                )
                loss = loss / args.grad_accum_steps
                loss.backward()
            did_first_backward = True

            if not is_accumulating:
                # dead-parameter guard, once. DDP is built with static_graph=True,
                # which (unlike the default) silently tolerates params that never
                # participate in the backward: it records the used/unused set on
                # the first iteration and never expects a grad from the rest. The
                # optimizers then skip them (`if g is None: continue`), so a
                # misconnected submodule trains at its init values forever without
                # a single warning -- exactly how the encoder adapter head stayed
                # dead. static_graph guarantees the set is fixed after iteration
                # one, so checking once is complete; `p.grad is None` touches no
                # data and forces no device sync.
                if not checked_dead_params:
                    checked_dead_params = True
                    dead = [
                        n
                        for n, p in model_without_ddp.named_parameters()
                        if p.requires_grad and p.grad is None
                    ]
                    if dead:
                        raise RuntimeError(
                            f"{len(dead)} param(s) have requires_grad=True but "
                            f"received no gradient; they are registered but never "
                            f"used in the forward pass and will never train: "
                            f"{dead[:16]}"
                            + (
                                f" ... (+{len(dead) - 16} more)"
                                if len(dead) > 16
                                else ""
                            )
                        )

                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )

                # guard against a non-finite main-model gradient. clip_grad_norm_
                # propagates a nan/inf norm into EVERY grad (grad *= nan), so a
                # single bad term would otherwise write garbage into the weights
                # and the divergence only surfaces later as the LPIPS range
                # assert. Skip the step and report which params carry it.
                skip_step = False
                if args.skip_nonfinite_grad:
                    if args.grad_clip > 0:
                        # clip_grad_norm_ already reduced over every grad, so a
                        # non-finite total norm == some grad is nan/inf (cheap).
                        skip_step = not torch.isfinite(grad_norm)
                    else:
                        skip_step = any(
                            p.grad is not None and not torch.isfinite(p.grad).all()
                            for p in model.parameters()
                        )
                    if skip_step:
                        consecutive_skips += 1
                        if ddp_rank0:
                            bad = [
                                n
                                for n, p in model_without_ddp.named_parameters()
                                if p.grad is not None
                                and not torch.isfinite(p.grad).all()
                            ]
                            logger.warning(
                                f"non-finite grad at epoch {epoch} iter "
                                f"{local_step} (grad_norm="
                                f"{locals().get('grad_norm', float('nan'))}); "
                                f"skipping optimizer step "
                                f"({consecutive_skips} in a row). {len(bad)} "
                                f"param(s) affected, first few: {bad[:8]}"
                            )
                        optimizer.zero_grad(set_to_none=True)

                        if (
                            args.max_consecutive_skips > 0
                            and consecutive_skips >= args.max_consecutive_skips
                        ):
                            raise RuntimeError(
                                f"{consecutive_skips} consecutive non-finite "
                                f"grads at epoch {epoch} iter {local_step}; "
                                "training state is unrecoverable (persistent "
                                "nan in weights or a running buffer). Aborting "
                                "-- resume from the last good checkpoint."
                            )
                    else:
                        consecutive_skips = 0

                # NOTE: do NOT `continue` here. The score-net update and the
                # metric logging below are what tell you WHY the grad went
                # non-finite (which loss term, which critic); skipping straight
                # to the next iteration hides the one step that mattered. Only
                # the main optimizer step and the EMA update are withheld.
                if not skip_step:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # update ema model
                if args.use_ema and not skip_step:
                    if args.use_ema_scheduler:
                        ema_model_decay = get_ema_decay(
                            current_step=global_step,
                            total_steps=total_steps,
                            start_decay=ema_model_start_decay,
                            end_decay=ema_model_end_decay,
                        )
                        ema_model.decay = ema_model_decay
                    ema_model.step(model)

            # ---- score-net update (denoising score matching) ----
            # Runs on EVERY micro-step, outside the accumulation window: each
            # loss call overwrites the cache, so stepping only at window close
            # would train the critics on the last micro-batch alone while the
            # generator gradient covers all of them.
            # Trains real (+ fake) score nets on the cached detached
            # features from this step's forward. Runs only once the
            # generator surrogate is active and features were cached.
            sm = loss_fn.score_match
            if use_score and epoch >= sm.start_epoch and sm._cache is not None:
                # cosine-decay the score-net lr in fractional epoch units.
                # Each group carries its OWN anchor (real_start_epoch /
                # fake_start_epoch) and may carry its own warmup and decay
                # length. Clamped at 0 so anything before a group's anchor
                # stays flat at its warmup/max value.
                score_epoch_pos = epoch + local_step / iters_per_rank
                group_lrs = {}
                for pg in score_optimizer.param_groups:
                    pg_decay_epochs = pg.get("decay_epochs", score_decay_epochs)
                    pg_lr = cosine_scheduler(
                        pg.get("lr_max", score_lr_max),
                        pg.get("min_lr", score_lr_min),
                        max(
                            0.0,
                            score_epoch_pos - pg.get("anchor", score_decay_anchor),
                        ),
                        warmup_steps=pg.get("warmup_epochs", score_warmup_epochs),
                        decay=pg_decay_epochs > 0,
                        decay_steps=pg_decay_epochs,
                    )
                    pg["lr"] = pg_lr
                    group_lrs[pg.get("name", "score")] = pg_lr
                # log each critic's lr under its own key
                loss_fn.log_dict["score_dsm_real_lr"] = group_lrs["real"]
                loss_fn.log_dict["score_dsm_fake_lr"] = group_lrs["fake"]

                real, real_w, fake, fake_w, score_conditions = sm._cache

                def _side_grad_norm(params):
                    # (pre-clip) L2 grad norm over one critic's params.
                    # With sync_interval == 1 the grads are already
                    # DDP-all-reduced by backward, so this is the true
                    # global norm on every rank; above 1 the critics step
                    # locally and this (like the clip and the non-finite
                    # guard below) is per-rank.
                    gs = [p.grad.detach() for p in params if p.grad is not None]
                    if not gs:
                        return torch.zeros((), device=device)
                    return torch.norm(torch.stack([g.norm() for g in gs]))

                def _clip_group(params, max_norm):
                    # clip ONE group to its own threshold and return its
                    # PRE-clip total norm (clip_grad_norm_ returns exactly
                    # that). With max_norm <= 0 the group is only measured.
                    # Either way the returned norm is nan/inf iff some grad
                    # in the group is, so it doubles as the guard signal.
                    if max_norm > 0:
                        return torch.nn.utils.clip_grad_norm_(params, max_norm)
                    return _side_grad_norm(params)

                for _ in range(int(score_config["loss"]["updates"])):
                    score_optimizer.zero_grad(set_to_none=True)
                    # no_sync() has to cover the FORWARD too -- DDP decides
                    # there whether this backward will reduce.
                    #
                    # The FIRST update of the process always syncs: under
                    # static_graph=True the reducer is armed by _DDPSink on
                    # the first backward, and entering that one inside
                    # no_sync() trips `expect_autograd_hooks_ INTERNAL
                    # ASSERT FAILED` in reducer.cpp. One synced update per
                    # process (including after a resume) costs nothing.
                    _local_step = score_averager is not None and (
                        score_updates_done > 0
                    )
                    _sync_ctx = (
                        score_ddp.no_sync() if _local_step else nullcontext()
                    )
                    with _sync_ctx:
                        with autocast_ctx:
                            dsm_loss = score_ddp(
                                real,
                                real_w,
                                fake,
                                fake_w,
                                score_conditions,
                            )
                        dsm_loss.backward()
                    # per-side (pre-clip) grad norms: real and fake nets
                    # have independent params + independent loss terms, so
                    # their grads don't interact. Logged separately to tell
                    # which critic is destabilizing. The legacy aggregate
                    # norm is reconstructed below from the group norms.
                    loss_fn.log_dict["score_dsm_real_grad_norm"] = _side_grad_norm(
                        sm.real_score_nets.parameters()
                    )
                    loss_fn.log_dict["score_dsm_fake_grad_norm"] = _side_grad_norm(
                        sm.fake_score_nets.parameters()
                    )
                    # optimizer-side grad-norm clip (like the main model's
                    # --grad_clip): caps the DSM update magnitude so a
                    # spiky gradient can't blow up the critic. Each returned
                    # norm is also the group's non-finite signal (it is
                    # nan/inf iff some grad in it is), replacing the
                    # per-param scan.
                    score_bad = False
                    score_group_norms = []
                    for _max_norm, _params in score_clip_groups:
                        _n = _clip_group(_params, _max_norm)
                        score_group_norms.append(_n)
                        score_bad = score_bad or not torch.isfinite(_n)
                    # Preserve the existing aggregate metric for dashboards
                    # and comparisons with older runs. All groups are
                    # disjoint, so their L2 norms combine in quadrature.
                    loss_fn.log_dict["score_dsm_grad_norm"] = (
                        torch.linalg.vector_norm(torch.stack(score_group_norms))
                    )
                    # guard the score-net step the same way the main model
                    # is guarded: a low-sigma DSM draw under autocast can
                    # emit a non-finite grad, and an unguarded step would
                    # write NaN into the real critic weights, after which
                    # every generator transport (pred_real = real_net(...))
                    # is NaN forever and the encoder grad goes NaN with no
                    # recovery.
                    if score_bad:
                        if ddp_rank0:
                            logger.warning(
                                f"non-finite score-net grad at epoch "
                                f"{epoch} iter {local_step}; skipping "
                                f"score step"
                            )
                        score_optimizer.zero_grad(set_to_none=True)
                    else:
                        score_optimizer.step()

                    # re-couple the ranks (loss.sync_interval > 1 only).
                    # Counted in DSM UPDATES rather than training steps, so
                    # `updates` > 1 and gradient accumulation cannot
                    # silently change the sync rate. Incremented
                    # unconditionally: a rank that skipped its step above
                    # still has to enter the collective, and score_bad is a
                    # per-rank decision once the grads are not reduced.
                    score_updates_done += 1
                    if score_averager is not None and (
                        score_updates_done % score_sync_interval == 0
                    ):
                        score_averager.sync()
                loss_fn.log_dict["score_dsm_loss"] = dsm_loss.detach()
                sm._cache = None

            # update logging
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1

            if (
                local_step == 0 or (local_step + 1) % args.log_interval == 0
            ) and ddp_rank0:
                # unscale the loss for logging
                lossf = loss.item() * args.grad_accum_steps
                log_str = (
                    f"epoch {epoch:4d} | "
                    f"iter {local_step:8d} [/{len(train_loader)}] | "
                    f"step {global_step:6d} [/{total_steps}]: "
                    f"lr {lr:.8f}, "
                    f"enc_lr {lr * encoder_lr_scaler:.8f}, "
                    + f"loss {lossf:.6f}, "
                    + f"time {dt*1000:.2f}ms, "
                    + f"data {dd*1000:.2f}ms"
                )
                for k, v in loss_fn.log_dict.items():
                    log_str += f", {k} {v:.10f}"

                if args.grad_clip > 0 and "grad_norm" in locals():
                    log_str += f", grad_norm {grad_norm:.5f}"

                log_dict = {
                    "epoch": epoch,
                    "lr": lr,
                    "enc_lr": lr * encoder_lr_scaler,
                    "loss": lossf,
                    "step": global_step,
                    **loss_fn.log_dict,
                    **model_without_ddp.log_dict,
                }
                if args.grad_clip > 0 and "grad_norm" in locals():
                    log_dict["grad_norm"] = grad_norm

                if args.use_ema:
                    log_dict["ema_decay"] = ema_model.decay

                logger.info(log_str)

                if args.use_wandb:
                    wandb.log(log_dict)

                append_log(file_path=log_training_path, entry=log_dict)

            # advance the profiler; once the active window closes, dump a
            # CUDA-time-sorted op breakdown and tear the profiler down so the
            # rest of training runs at full speed.
            if prof is not None:
                prof.step()
                if local_step + 1 >= profile_total:
                    prof.stop()
                    logger.info(
                        "\n"
                        + prof.key_averages().table(
                            sort_by=f"{args.device_type}_time_total", row_limit=30
                        )
                    )
                    logger.info(
                        f"profiler trace written to {osp.join(exp_dir, 'profiler')} "
                        f"(open with: tensorboard --logdir {exp_dir})"
                    )
                    prof = None

        if epoch == 0 or (epoch + 1) % args.vis_interval == 0:
            visualize(
                vis_loader,
                model_without_ddp,
                ddp_rank,
                epoch,
                class_of_interest=args.class_of_interest,
                forward_steps=args.forward_steps,
                use_ema_model=False,
                ema_model=ema_model,
                save_dir=args.vis_dir,
                device=device,
                ctx=autocast_ctx,
            )

        if (epoch + 1) % args.ckpt_save_interval == 0:
            # the critics may be mid-interval and hold per-rank weights; average
            # them so the checkpoint is the same model the next sync would
            # produce, not whatever rank 0 happened to land on.
            if score_averager is not None:
                score_averager.sync()
            save_ckpt(
                model_without_ddp,
                epoch=epoch,
                ema_model=ema_model,
                score_match=score_ddp.module if use_score else None,
                score_optimizer=score_optimizer if use_score else None,
                fd_loss=getattr(loss_fn, "fd_loss", None),
                fd_judge_loss=getattr(loss_fn, "fd_judge_loss", None),
                ckpt_dir=args.ckpt_dir,
                ddp_rank0=ddp_rank0,
                cleanup_ckpt=args.cleanup_ckpt,
                cleanup_ckpt_interval=args.cleanup_ckpt_interval,
            )

    # save the final checkpoint (skip if no epoch ran, e.g. resuming at args.epochs)
    if epoch >= cur_epoch:
        if score_averager is not None:
            score_averager.sync()
        save_ckpt(
            model_without_ddp,
            epoch=epoch,
            ema_model=ema_model,
            score_match=score_ddp.module if use_score else None,
            score_optimizer=score_optimizer if use_score else None,
            fd_loss=getattr(loss_fn, "fd_loss", None),
            fd_judge_loss=getattr(loss_fn, "fd_judge_loss", None),
            ckpt_dir=args.ckpt_dir,
            ddp_rank0=ddp_rank0,
        )

    # end of training
    dist.destroy_process_group()


if __name__ == "__main__":
    main(cli_args)
