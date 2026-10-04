import ast
import glob
import logging
import math
import json
import os
from contextlib import nullcontext
from typing import List

import numpy as np
import PIL as pil
import torch
import torch.distributed as dist
import torchvision
from sphere.loader import resize_arr
from cli_utils import get_device_type

logger = logging.getLogger(__name__)

"""
helpers: schedulers
"""


def cosine_scheduler(
    max_value, min_value, current_step, warmup_steps=0, decay=True, decay_steps=0
):
    # 1) linear warmup
    if current_step < warmup_steps:
        return max_value * (current_step + 1) / (warmup_steps + 1)
    # 2) constant if no decay
    if not decay:
        return max_value
    # 3) floor at min_value if past decay_steps
    if current_step >= decay_steps:
        return min_value
    # 4) cosine decay
    decay_ratio = (current_step - warmup_steps) / (decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_value + coeff * (max_value - min_value)


def encoder_lr_scaler_scheduler(init_scaler, min_scaler, current_epoch, decay_epochs):
    # 1) no decay requested -> constant
    if decay_epochs <= 0:
        return init_scaler
    # 2) floor at min_scaler once decay is done
    if current_epoch >= decay_epochs:
        return min_scaler
    # 3) cosine decay
    decay_ratio = current_epoch / decay_epochs
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_scaler + coeff * (init_scaler - min_scaler)


"""
helpers: configurations
"""


def apply_dotted_overrides(cfg, overrides, log_mark="dot_config"):

    def _coerce(value):
        # parse a CLI string into bool / int / float, falling back to str
        low = value.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        # list / tuple literal, e.g. "[0]" or "[0,2]" (for list-valued keys)
        if value and value[0] in "[(":
            try:
                return list(ast.literal_eval(value))
            except (ValueError, SyntaxError):
                pass
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value

    # overrides: list of "a.b.c=value" strings; walks into nested dicts and sets
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--{log_mark} entry must be KEY=VALUE, got: {item!r}")
        key, raw = item.split("=", 1)
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"unknown {log_mark} key path: {key}")
            node = node[k]
        leaf = keys[-1]
        if leaf not in node:
            raise KeyError(f"unknown {log_mark} key path: {key}")
        node[leaf] = _coerce(raw)
        logger.info(f"{log_mark} override: {key} = {node[leaf]!r}")
    logger.info(f"{log_mark} after overrides: {json.dumps(cfg, indent=4)}")
    return cfg


"""
helpers: visualization
"""


def make_label_tiles(
    labels: List[int], height: int, width: int, dtype=torch.float32, font_size=None
) -> torch.Tensor:
    """
    render one white tile per label with the label text in black, sized to
    match the generated images so it can sit as a column in a make_grid row.
    returns a (N, 3, height, width) tensor in [0, 1] on cpu.
    """
    from PIL import Image, ImageDraw, ImageFont

    if font_size is None:
        font_size = max(10, height // 5)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)

    tiles = []
    for lbl in labels:
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        text = str(lbl)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        xy = ((width - (r - l)) / 2 - l, (height - (b - t)) / 2 - t)
        draw.text(xy, text, fill=(0, 0, 0), font=font)
        arr = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        tiles.append(arr.to(dtype) / 255.0)
    return torch.stack(tiles, dim=0)


def save_tensors_to_images(
    tensors: torch.Tensor | List[torch.Tensor],
    path: str = None,
    nrow_mult: int = 2,
    max_nimgs: int = 48,
    nrow: int = 0,
    gather_all_tensors: bool = False,
):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    assert path is not None

    tensors = [t for t in tensors]
    g = len(tensors)
    x = torch.cat(tensors, dim=-1)
    if gather_all_tensors:
        dist.barrier()
        x = concat_all_gather(x)
    x = x[:max_nimgs].cpu()
    b, c, h, w = x.shape
    x = (
        x.clamp(min=0.0, max=1.0)
        .reshape(b, c, h, g, w // g)
        .permute(0, 3, 1, 2, 4)
        .reshape(b * g, c, h, w // g)
    )
    if x.shape[0] // g <= 16:
        nrow_mult = 1
    grid = torchvision.utils.make_grid(
        x,
        nrow=nrow if nrow else nrow_mult * g,
        padding=max(0, (int(h / 32) - 2) // 2),
        pad_value=1,
    )
    if dist.is_initialized() and dist.get_rank() == 0:
        torchvision.utils.save_image(grid, path)
    logger.info(f"grid saved to the image: {path}")


@torch.inference_mode()
def save_image(x, batch_idx, ddp_rank, save_dir, force_image_size=-1):
    assert isinstance(x, torch.Tensor)
    x = torch.round(x * 255.0).to(torch.uint8)
    x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
    x = x.cpu().numpy()

    for i, img in enumerate(x):
        image_name = f"rank={ddp_rank:05d}_ord={batch_idx:05d}_idx={i:05d}.png"
        image_path = os.path.join(save_dir, image_name)
        image = pil.Image.fromarray(img)

        if force_image_size > 0:
            image = resize_arr(image, image_size=force_image_size)

        image.save(image_path, format="PNG", compress_level=0)


@torch.no_grad()
def visualize(
    vis_loader,
    model_without_ddp,
    ddp_rank,
    epoch,
    cfg=0.0,
    cfg_position="angle",
    forward_steps=1,
    class_of_interest=None,
    ema_model=None,
    use_ema_model=False,
    save_dir=None,
    gather_all_tensors=False,
    device=get_device_type(),
    ctx=nullcontext(),
):
    assert save_dir is not None

    if ema_model is not None:
        ema_model = ema_model.module  # ModuleEMA
        ema_model.eval()

    if use_ema_model and ema_model is not None:
        model = ema_model
    else:
        logger.info("no ema_model to use")
        use_ema_model = False
        model = model_without_ddp

    model.eval()

    skip = int(torch.randint(0, 32, (1,)).item())
    for _ in range(skip):
        next(vis_loader)
    imgs, clss = next(vis_loader)[:2]

    N = 1 if gather_all_tensors else len(imgs)

    imgs = imgs[:N].to(device, non_blocking=True)
    clss = clss[:N].to(device, non_blocking=True)

    if class_of_interest is not None:
        coi = torch.tensor(class_of_interest, dtype=torch.long, device=device)
        if gather_all_tensors:
            gen_clss = coi[ddp_rank % len(coi)].repeat(N)
        else:
            gen_clss = coi[torch.arange(N, device=device) % len(coi)]
    else:
        gen_clss = clss

    with ctx:
        # reconstruction
        rec_imgs = model.reconstruct(imgs, clss, sampling=False)
        rec_imgs_with_noise_small = model.reconstruct(
            imgs, clss, noise_scaler=float(45 / 90), sampling=True
        )
        rec_imgs_with_noise_large = model.reconstruct(
            imgs, clss, noise_scaler=float(89 / 90), sampling=True
        )

        # generation
        gen_rand_1_step, gen_rand_n_step = model.generate(
            batch_size=N,
            y=clss if model.use_modulation else None,
            cfg=cfg,
            cfg_position=cfg_position,
            forward_steps=forward_steps,
            device=device,
        )
        gen_clss_1_step, gen_clss_n_step = model.generate(
            batch_size=N,
            y=gen_clss if model.use_modulation else None,
            cfg=cfg,
            cfg_position=cfg_position,
            forward_steps=forward_steps,
            device=device,
        )

    ori_imgs = imgs * 0.5 + 0.5

    to_zip = [
        ori_imgs,
        rec_imgs,
        rec_imgs_with_noise_small,
        rec_imgs_with_noise_large,
        gen_rand_1_step,
        gen_clss_1_step,
    ]
    if forward_steps > 1:
        to_zip.append(gen_rand_n_step)
        to_zip[-1], to_zip[-2] = to_zip[-2], to_zip[-1]
        to_zip.append(gen_clss_n_step)

    if ddp_rank == 0:
        img_path = (
            f"imgs_ep{epoch:04d}"
            f"_ema={use_ema_model}"
            f"_cfg={cfg}-{cfg_position}"
            f"_steps={forward_steps}"
            f".png"
        )
        save_tensors_to_images(
            to_zip,
            path=os.path.join(save_dir, img_path),
            nrow_mult=2,
        )

    dist.barrier()


"""
helper: checkpointing
"""


def save_ckpt(
    model_without_ddp,
    optimizer=None,
    loss_scaler=None,
    epoch=0,
    ema_model=None,
    score_match=None,
    score_optimizer=None,
    fd_loss=None,
    fd_judge_loss=None,
    ckpt_dir=None,
    ddp_rank0=False,
    cleanup_ckpt=False,
    cleanup_ckpt_interval=5,
):
    assert ckpt_dir is not None

    dist.barrier()

    if ddp_rank0:
        ckpt = {
            "model": model_without_ddp.state_dict(),
            "epoch": epoch,
        }
        if ema_model is not None:
            ckpt["ema_model"] = ema_model.module.state_dict()  # ModuleEMA
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if loss_scaler is not None:
            ckpt["loss_scaler"] = loss_scaler.state_dict()
        if score_match is not None:
            # only the score net(s) are trainable submodules of ScoreMatchingLoss
            ckpt["score_match"] = score_match.state_dict()
        if score_optimizer is not None:
            ckpt["score_optimizer"] = score_optimizer.state_dict()
        if fd_loss is not None:
            # running (mu, E[xx^T]) moments of the FD loss
            ckpt["fd_loss"] = fd_loss.state_dict()
        if fd_judge_loss is not None:
            # per-judge history (JudgeEMA buffers) of the judges-space FD
            ckpt["fd_judge_loss"] = fd_judge_loss.state_dict()
        ckpt_path = os.path.join(ckpt_dir, f"ep{epoch:04d}.pth")
        # write to a temp name and rename atomically: if slurm kills the job
        # mid-write, the truncated file never carries the .pth name, so the
        # auto-resume glob cannot pick it up.
        tmp_path = ckpt_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)
        logger.info(f"checkpoint saved to {ckpt_path}")

        organize_ckpt(
            ckpt_dir,
            milestone_interval=cleanup_ckpt_interval,
            cleanup_checkpoints=cleanup_ckpt,
        )

    dist.barrier()


def is_ckpt_valid(ckpt_path: str, ref_size: int = None) -> bool:
    """
    Check that a checkpoint file is complete without loading its tensors.

    torch.save writes a zip archive whose central directory is at the END of
    the file, so a save that was cut short (slurm time limit, node failure)
    leaves a file that cannot be opened as a zip or fails the CRC scan. This
    reads only the directory table and is cheap even for multi-GB files.

    `ref_size` (bytes) is an optional size of a sibling checkpoint from the
    same run; a file much smaller than it is logged as suspicious but the zip
    check is what decides.
    """
    import zipfile

    if not os.path.isfile(ckpt_path):
        return False
    size = os.path.getsize(ckpt_path)
    if size == 0:
        logger.warning(f"checkpoint {ckpt_path} is empty")
        return False
    if ref_size is not None and size < 0.9 * ref_size:
        logger.warning(
            f"checkpoint {ckpt_path} is {size / 2**20:.1f} MiB, "
            f"much smaller than its sibling ({ref_size / 2**20:.1f} MiB)"
        )
    try:
        with zipfile.ZipFile(ckpt_path) as zf:
            bad = zf.testzip()  # CRC check of every member; None if all good
        if bad is not None:
            logger.warning(f"checkpoint {ckpt_path} has a corrupt member: {bad}")
            return False
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        logger.warning(f"checkpoint {ckpt_path} is not a complete zip file: {e}")
        return False
    return True


def find_latest_valid_ckpt(ckpt_dir: str):
    """
    Return the newest complete `ep*.pth` checkpoint in `ckpt_dir`, or None.

    Checkpoints are sorted by epoch and tried newest first; broken ones (e.g. a
    save cut short by the slurm time limit) are skipped with a warning so
    training resumes from the last good epoch instead of crashing.
    """
    ckpts = glob.glob(os.path.join(ckpt_dir, "ep*.pth"))
    if not ckpts:
        return None

    def epoch_of(path):
        try:
            return int(os.path.basename(path).rsplit("ep", 1)[-1].split(".")[0])
        except ValueError:
            return -1

    ckpts = sorted(ckpts, key=epoch_of, reverse=True)  # newest first
    largest = max(os.path.getsize(c) for c in ckpts)
    for ckpt in ckpts:
        if is_ckpt_valid(ckpt, ref_size=largest):
            return ckpt
        logger.warning(f"skipping broken checkpoint {ckpt}")
    logger.warning(f"no valid checkpoint found in {ckpt_dir}")
    return None


def load_ckpt(
    model_without_ddp,
    ckpt_path=None,
    ema_model=None,
    strict=False,
    override_model_with_ema=False,
    verbose=False,
    return_ckpt=False,
):
    if ckpt_path is None:
        return

    dist.barrier()

    logger.info(f"loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    def _normalize_name(name: str) -> str:
        return (
            name.replace("module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("._orig_mod", "")
        )

    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        k = _normalize_name(k)

        new_state_dict[k] = v
    state_dict = new_state_dict

    msgs = model_without_ddp.load_state_dict(state_dict, strict=strict)
    if strict:
        assert len(msgs.missing_keys) == 0, f"{msgs.missing_keys}"
    else:
        logger.warning(f"missing keys: {msgs.missing_keys}")

    if verbose:
        logger.info(msgs)

    if ema_model is not None:
        if "ema_model" in ckpt:

            ema_state_dict = ckpt["ema_model"]
            new_state_dict = {}
            for k, v in ema_state_dict.items():
                k = _normalize_name(k)
                new_state_dict[k] = v
            ema_state_dict = new_state_dict

            ema_model.load_state_dict(ema_state_dict, strict=strict)

            if override_model_with_ema:
                ema_model.copy_to(model_without_ddp)
                logger.info("copy ema model to model")

        else:
            ema_model.load_state_dict(state_dict, strict=False)
            logger.info("no ema_model state_dict, load ema model from model")

            if override_model_with_ema:
                logger.warning(
                    "override_model_with_ema is True, but no ema_model to override"
                )

    dist.barrier()

    if return_ckpt:
        return ckpt

    ckpt = None  # free up memory


def organize_ckpt(
    ckpt_dir: str,
    milestone_interval: int = 5,
    cleanup_checkpoints: bool = False,
):
    """
    Clean up older checkpoint files in `ckpt_dir` while keeping milestone checkpoints and the newest checkpoint.

    Parameters
    ----------
    ckpt_dir : str
        The directory where checkpoint .pth files are stored.
    milestone_interval : int, optional
        The interval used to decide if a checkpoint is a "milestone."
        If (epoch_num + 1) % milestone_interval == 0, it is kept (default=50).
    cleanup_checkpoints : bool, optional
        Whether to delete the checkpoints that are not kept (default=False).
    """

    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]

    def get_ckpt_num(path):
        """
        Extract the epoch number from a checkpoint filename.
        """
        filename = os.path.basename(path)
        # expecting something like 'epoch_049.pth'
        # we'll parse out the part after the last underscore and before '.pth'
        try:
            return int(filename.rsplit("ep", 1)[-1].split(".")[0])
        except ValueError:
            return None

    # sort checkpoints by epoch number
    ckpts.sort(key=lambda x: (get_ckpt_num(x) is None, get_ckpt_num(x)))

    # filter out any that failed to parse an integer epoch (get_ckpt_num == None)
    ckpts = [ckpt for ckpt in ckpts if get_ckpt_num(ckpt) is not None]

    if not ckpts:
        # if no checkpoints remain, nothing to do
        return

    # determine which checkpoints to keep:
    # 1. the newest checkpoint for resume.
    # 2. any milestone checkpoints.
    #    (epoch_num + 1) % milestone_interval == 0
    newest_ckpt = ckpts[-1]
    milestone_keep = set(
        ckpt for ckpt in ckpts if ((get_ckpt_num(ckpt) + 1) % milestone_interval == 0)
    )

    # union of both sets
    keep_set = milestone_keep.union({newest_ckpt})

    # remove anything not in keep_set
    for ckpt in ckpts:
        if ckpt not in keep_set and cleanup_checkpoints:
            os.remove(ckpt)
            logger.info(f"Removed checkpoint: {ckpt}")


"""
helpers: tensor ops
"""


class _AllGatherWithGrad(torch.autograd.Function):
    """
    all_gather along dim 0 with the same gradient semantics as the (now
    deprecated) torch.distributed.nn.functional.all_gather: the backward is
    a SUM reduce-scatter, so every rank's contribution to the gathered
    tensor's gradient reaches the rank that owns that chunk.
    """

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        tensor = tensor.contiguous()
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.world_size = world_size
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        chunks = list(grad_output.chunk(ctx.world_size, dim=0))
        grad_input = torch.empty_like(chunks[0])
        dist.reduce_scatter(grad_input, chunks, op=dist.ReduceOp.SUM)
        return grad_input


def nn_concat_all_gather(tensor, gather_dim=0):
    """with gradient support"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    if gather_dim == 0:
        return _AllGatherWithGrad.apply(tensor)
    # gather along another dim: move it to the front, gather, move it back
    out = _AllGatherWithGrad.apply(tensor.movedim(gather_dim, 0))
    return out.movedim(0, gather_dim)


def concat_all_gather(tensor):
    """without gradient support"""
    if dist.get_world_size() == 1:
        return tensor
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def vector_compute_magnitude(x):
    assert x.ndim >= 2
    reduce_dims = tuple(range(1, x.ndim))
    mag = x.float().square().sum(dim=reduce_dims, keepdim=True).sqrt()
    return mag


"""
helpers: comm
"""


class _BF16ErrorFeedbackState:
    """
    Per-bucket fp32 residual store for the error-feedback bf16 all-reduce.
    """

    def __init__(self, process_group=None):
        self.process_group = process_group
        self.residuals = {}


def bf16_ef_comm_hook(state, bucket):
    """
    bf16-compressed gradient all-reduce with error feedback.

    Standard EF: correct this step's local grads with the residual carried
    from last step, quantize to bf16, stash the new quantization error as the
    next residual, then all-reduce the bf16 payload (half the wire volume).
    Each rank keeps its own residual, so the error is never dropped -- it is
    folded into the following step instead, which recovers most of the accuracy
    that a plain bf16 all-reduce loses when summing across many ranks.
    """
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = group.size()

    grad = bucket.buffer()  # local (pre-all-reduce) fp32 grads
    idx = bucket.index()

    residual = state.residuals.get(idx)
    if residual is not None and residual.shape != grad.shape:
        residual = None
        state.residuals.pop(idx, None)
    if residual is not None:
        grad.add_(residual)  # p_t = grad_t + e_{t-1}

    compressed = grad.to(torch.bfloat16)
    # e_t = p_t - Q(p_t); stash before the buffer is overwritten by the result
    state.residuals[idx] = grad - compressed.to(grad.dtype)

    fut = dist.all_reduce(compressed, group=group, async_op=True).get_future()

    def _decompress(fut):
        out = fut.value()[0].to(grad.dtype)
        out.div_(world_size)
        return out

    return fut.then(_decompress)


class ParamAverager:
    """
    Periodic parameter averaging for a DDP module that is stepped LOCALLY.

    The alternative to synchronizing gradients: every rank runs its own
    forward/backward/step under `no_sync()` -- no gradient collective at all --
    and the ranks are re-coupled here by averaging the WEIGHTS every N updates.
    Comm per update goes from

        one gradient all-reduce  every step   (bf16, via the comm hook)
        one parameter all-reduce every N steps (fp32)

    i.e. down by a factor of N/2. The parameters are deliberately NOT quantized
    to bf16 the way the gradients are: a rounding error in a gradient is
    re-estimated on the next step, whereas one written into a weight is
    permanent and would accumulate over every sync.

    Optimizer state is left local: syncing Adam's two moments would triple the
    payload, for second-order state that re-estimates itself within a few steps
    of a weight sync.

    Between syncs the ranks hold slightly different models. For an auxiliary
    network (a critic, a discriminator) that is a mild approximation; for the
    model actually being trained it would not be.
    """

    def __init__(self, params):
        self.params = [p for p in params if p.requires_grad]
        assert self.params, "ParamAverager got no trainable parameters"
        assert all(
            p.dtype == torch.float32 for p in self.params
        ), "ParamAverager assumes fp32 parameters (it stages them in one fp32 buffer)"
        self.numel = sum(p.numel() for p in self.params)
        # ONE contiguous staging buffer, so a sync is one collective rather than
        # one per tensor: at a few hundred tensors the per-collective latency
        # would otherwise dwarf the transfer itself.
        self.flat = torch.empty(
            self.numel, device=self.params[0].device, dtype=torch.float32
        )
        views, ofs = [], 0
        for p in self.params:
            views.append(self.flat[ofs : ofs + p.numel()].view_as(p))
            ofs += p.numel()
        self.views = views

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(tensors={len(self.params)}, "
            f"params={self.numel / 1e6:.2f}M, "
            f"{self.numel * 4 / 1e6:.0f} MB per sync)"
        )

    @torch.no_grad()
    def sync(self):
        """Average the parameters across ranks.

        A COLLECTIVE: every rank must call it, the same number of times, in the
        same order. No-op outside distributed / on a single rank.
        """
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return
        torch._foreach_copy_(self.views, [p.data for p in self.params])
        dist.all_reduce(self.flat, op=dist.ReduceOp.SUM)
        self.flat.div_(dist.get_world_size())
        torch._foreach_copy_([p.data for p in self.params], self.views)

