
import argparse
import importlib
import json
from pathlib import Path

import torch


def depth_to_target(raw):

    raw = raw.float()
    if raw.ndim != 2 or not torch.isfinite(raw).all():
        raise ValueError("Expected a finite two-dimensional depth prediction")
    low, high = torch.quantile(raw.flatten(), torch.tensor([0.01, 0.99], device=raw.device))
    if high - low <= 1e-8:
        raise ValueError("Degenerate depth prediction")
    target = ((raw - low) / (high - low)).clamp(0, 1)
    return target.unsqueeze(0).expand(3, -1, -1), {"low": low.item(), "high": high.item()}


def similarity(prediction, target):
    """Display-space diagnostics, without alignment, clipping, or refitting."""
    gray = prediction.mean(0)
    truth = target.mean(0)
    a = gray.flatten().double() - gray.double().mean()
    b = truth.flatten().double() - truth.double().mean()
    denominator = a.norm() * b.norm()
    correlation = (a @ b / denominator).item() if denominator > 1e-12 else None
    return {
        "rgb_mse": (prediction - target).square().mean().item(),
        "gray_mse": (gray - truth).square().mean().item(),
        "gray_pearson": correlation,
        "channel_spread_mean": (prediction.max(0).values - prediction.min(0).values).mean().item(),
        "output_min_max": [prediction.min().item(), prediction.max().item()],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list of image path strings")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be new")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--loader-module", default="experiments.intrinsics.traverse")
    parser.add_argument("--seed", type=int, default=673)
    args = parser.parse_args()
    paths = [Path(p) for p in json.loads(args.manifest.read_text())]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Manifest must contain existing image files")
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU node")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    import torch.distributed as dist
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    loader = importlib.import_module(args.loader_module)
    dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        config_path = args.checkpoint.parent.parent / "cfg.json" if args.checkpoint.parent.name == "ckpt" else args.checkpoint.parent / "cfg.json"
        config = json.loads(config_path.read_text())
        model = loader.build_checkpoint_model(config, args.checkpoint, "ema", True).cuda()
        print("Frozen Sphere Encoder checkpoint loaded", flush=True)
        processor = AutoImageProcessor.from_pretrained(args.depth_model, cache_dir=str(args.cache_dir))
        depth_model = AutoModelForDepthEstimation.from_pretrained(args.depth_model, cache_dir=str(args.cache_dir)).eval().requires_grad_(False).cuda()
        print("Reference depth estimator loaded", flush=True)
        from sphere.loader import center_crop_arr

        metadata = {
            "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
            "depth_model": args.depth_model, "depth_model_revision": getattr(depth_model.config, "_commit_hash", None),
            "seed": args.seed, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "conditioning": "null class, decoder angle fixed at zero", "weights": "ema",
            "depth_normalization": "per-image percentile 1 to 99, clipped to [0,1], estimator value orientation retained",
            "target_status": "pseudo relative-depth targets, not measured ground truth",
            "metrics": "display-space reconstruction diagnostics; not metric-depth accuracy",
            "images": [],
        }
        overview = []
        overview_labels = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                cropped = center_crop_arr(image.convert("RGB"), model.decoder.input_size)
            original = torch.from_numpy(np.array(cropped).copy()).permute(2, 0, 1).float().cuda() / 255
            depth_inputs = processor(images=cropped, return_tensors="pt").to("cuda")
            prediction = depth_model(**depth_inputs)
            raw = processor.post_process_depth_estimation(prediction, target_sizes=[(cropped.height, cropped.width)])[0]["predicted_depth"]
            target, normalization = depth_to_target(raw)
            rgb_latent = model.f(model.encode(original.unsqueeze(0) * 2 - 1)).float()
            depth_latent = model.f(model.encode(target.unsqueeze(0) * 2 - 1)).float()
            rgb_reconstruction = model.decode(rgb_latent, alpha_deg=0)[0].float()
            depth_reconstruction = model.decode(depth_latent, alpha_deg=0)[0].float()
            for tensor in [rgb_reconstruction, depth_reconstruction]:
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("Nonfinite Sphere Encoder output")
            gray_rgb = original.mean(0, keepdim=True).expand_as(original)
            constant = torch.full_like(target, target.mean().item())
            metrics = {
                "depth_autoencoder": similarity(depth_reconstruction, target),
                "rgb_decoder_vs_depth": similarity(rgb_reconstruction, target),
                "grayscale_rgb_vs_depth": similarity(gray_rgb, target),
                "constant_vs_depth": similarity(constant, target),
                "rgb_reconstruction_mse": (rgb_reconstruction - original).square().mean().item(),
                "rgb_to_depth_latent": loader.latent_stats(rgb_latent, depth_latent, model.spherify_mode),
            }
            stem = f"{index:02d}_{path.stem}"
            np.save(args.output_dir / f"{stem}_raw_depth.npy", raw.cpu().numpy())
            np.save(args.output_dir / f"{stem}_target.npy", target[0].cpu().numpy())
            for name, tensor in [("rgb", original), ("target", target), ("reconstructed_depth", depth_reconstruction)]:
                array = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(array).save(args.output_dir / f"{stem}_{name}.png")
            row = [original, rgb_reconstruction, target, depth_reconstruction, gray_rgb]
            labels = ["Input RGB", "RGB reconstruction", "Pseudo-depth target", "Depth reconstruction", "Grayscale RGB baseline"]
            loader.save_grid([row], [labels], args.output_dir / f"{stem}_feasibility.png")
            overview.append(row)
            overview_labels.append(labels)
            record = {"image": str(path), "stem": stem, "normalization": normalization, "metrics": metrics}
            metadata["images"].append(record)
            # Incremental progress survives an interrupted job.
            (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"{index+1}/{len(paths)} {path.name}: depth MSE={metrics['depth_autoencoder']['rgb_mse']:.6f}, constant MSE={metrics['constant_vs_depth']['rgb_mse']:.6f}", flush=True)
        loader.save_grid(overview, overview_labels, args.output_dir / "overview.png")
        summary = {}
        for method in ["depth_autoencoder", "rgb_decoder_vs_depth", "grayscale_rgb_vs_depth", "constant_vs_depth"]:
            summary[method] = {}
            for metric in ["rgb_mse", "gray_mse", "gray_pearson", "channel_spread_mean"]:
                values = [r["metrics"][method][metric] for r in metadata["images"] if r["metrics"][method][metric] is not None]
                summary[method][metric] = sum(values) / len(values) if values else None
        metadata["summary_mean"] = summary
        metadata["depth_autoencoder_beats_constant_count"] = sum(r["metrics"]["depth_autoencoder"]["rgb_mse"] < r["metrics"]["constant_vs_depth"]["rgb_mse"] for r in metadata["images"])
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Pilot completed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
