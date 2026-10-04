import argparse
import torch


def get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def get_dist_backend(device_type):
    return {"cuda": "nccl", "xpu": "xccl"}.get(device_type, "gloo")


def device_count(device_type):
    if device_type == "cuda":
        return torch.cuda.device_count()
    elif device_type == "xpu":
        return torch.xpu.device_count()
    else:
        return 1


def set_device(device_type, index):
    device = torch.device(device_type if device_type == "cpu" else f"{device_type}:{index}")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    elif device_type == "xpu":
        torch.xpu.set_device(device)
    return device


def empty_cache(device_type):
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "xpu":
        torch.xpu.empty_cache()


def str2bool(v):
    """
    usage:
        --flag_name true or --flag_name false
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def none_or_str(v):
    return None if v.lower() == "none" else v


def cfg_sweep(cfg_min, cfg_max, cfg_gap):
    """guidance values to sweep: `cfg_min` first, then every multiple of
    `cfg_gap` above it up to and including `cfg_max` (the max is always the
    last value). e.g. (1, 25, 5) -> [1, 5, 10, 15, 20, 25] and
    (1.0, 2.0, 0.2) -> [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]."""
    cfg_min, cfg_max = float(cfg_min), float(cfg_max)
    vals = [cfg_min]
    if cfg_gap > 0 and cfg_max > cfg_min:
        # first multiple of the gap strictly above cfg_min
        k = int(cfg_min / cfg_gap) + 1
        while True:
            v = round(k * cfg_gap, 6)
            if v >= cfg_max - 1e-9:
                break
            if v > cfg_min + 1e-9:
                vals.append(v)
            k += 1
    if cfg_max > cfg_min:
        vals.append(cfg_max)
    return [float(round(v, 2)) for v in vals]
