"""Independent relative-depth evaluation on measured DIODE validation data.

No adaptation or checkpoint selection occurs here. Metrics are relative inverse
depth, not metric-depth benchmark scores. Alignment uses evaluation ground truth
and is explicitly reported as an oracle affine alignment diagnostic.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .compare_adaptive import make_candidate
from experiments.intrinsics.geometry import normalize
from experiments.intrinsics.train_shared_depth import decode_with_grad


def aligned_sensor_crop(depth, mask, size=256):
    """Use nearest samples with the exact sequence of RGB resize/crop coordinates."""
    import numpy as np
    a = torch.from_numpy(np.asarray(depth, dtype=np.float32).squeeze().copy())[None, None]
    b = torch.from_numpy(np.asarray(mask, dtype=np.float32).squeeze().copy())[None, None]
    while min(a.shape[-2:]) >= 2 * size:
        shape = tuple(x // 2 for x in a.shape[-2:])
        a, b = F.interpolate(a, size=shape, mode="nearest-exact"), F.interpolate(b, size=shape, mode="nearest-exact")
    scale = size / min(a.shape[-2:])
    shape = tuple(round(x * scale) for x in a.shape[-2:])
    a, b = F.interpolate(a, size=shape, mode="nearest-exact"), F.interpolate(b, size=shape, mode="nearest-exact")
    y, x = (shape[0] - size) // 2, (shape[1] - size) // 2
    return a[0, 0, y:y+size, x:x+size], b[0, 0, y:y+size, x:x+size].bool()


def measured_metrics(prediction, depth, mask):
    pred = prediction.mean(0).double()
    valid = mask & torch.isfinite(depth) & (depth > 0)
    if valid.sum() < 100:
        raise ValueError("Insufficient valid measured depth pixels")
    truth = depth[valid].double().reciprocal()
    values = pred[valid]
    low, high = torch.quantile(truth, torch.tensor([.01, .99], device=truth.device, dtype=truth.dtype))
    if high <= low:
        raise ValueError("Degenerate measured inverse depth")
    target = ((truth - low) / (high - low)).clamp(0, 1)
    centered, tc = values - values.mean(), truth - truth.mean()
    denominator = centered.norm() * tc.norm()
    correlation = (centered @ tc / denominator).item() if denominator > 1e-12 else None
    slope = (centered @ tc) / centered.square().sum().clamp_min(1e-12)
    aligned = slope * centered + truth.mean()
    return {"valid_pixels": valid.sum().item(),
            "normalized_inverse_mse": (values - target).square().mean().item(),
            "inverse_pearson": correlation,
            "oracle_affine_inverse_rmse": (aligned - truth).square().mean().sqrt().item(),
            "oracle_affine_slope": slope.item()}


def select_images(root, per_scan):
    groups = defaultdict(list)
    for p in sorted(root.rglob("*.png")):
        if p.with_name(p.stem + "_depth.npy").is_file():
            groups[p.parent].append(p)
    selected = []
    for folder in sorted(groups):
        files = groups[folder]
        if per_scan and len(files) > per_scan:
            ids = torch.linspace(0, len(files) - 1, per_scan).round().long().tolist()
            files = [files[i] for i in ids]
        selected.extend(files)
    if not selected:
        raise ValueError("No DIODE RGB/depth pairs found")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--training-pairs", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-scan", type=int, default=4, help="0 evaluates every image; default is a fixed pilot subset")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    import numpy as np
    from PIL import Image
    import torch.distributed as dist
    from sphere.loader import center_crop_arr
    from experiments.intrinsics.traverse import build_checkpoint_model, save_grid
    paths = select_images(args.dataset_root, args.per_scan)
    # Commit membership to disk before any model predictions.
    manifest = [{"image": str(path), "scan": str(path.parent.relative_to(args.dataset_root)),
                 "scene": str(path.parent.parent.relative_to(args.dataset_root))} for path in paths]
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        model = build_checkpoint_model(json.loads((args.checkpoint.parent.parent / "cfg.json").read_text()), args.checkpoint, "ema", True).to(device)
        if model.spherify_mode != "global":
            raise ValueError("Expected global-sphere checkpoint")
        train = torch.load(args.training_pairs, map_location="cpu", weights_only=True)
        template = train["target"][[i for i, row in enumerate(train["manifest"]) if row["split"] == "train"]].mean(0).to(device)
        rgb, zs, depths, masks = [], [], [], []
        with torch.no_grad():
            for path in paths:
                with Image.open(path) as image:
                    crop = center_crop_arr(image.convert("RGB"), 256)
                original = torch.from_numpy(np.array(crop).copy()).permute(2, 0, 1).float().to(device) / 255
                depth, mask = aligned_sensor_crop(np.load(path.with_name(path.stem + "_depth.npy")),
                                                  np.load(path.with_name(path.stem + "_depth_mask.npy")))
                rgb.append(original.cpu()); depths.append(depth); masks.append(mask)
                zs.append(normalize(model.f(model.encode(original[None] * 2 - 1)).float())[0].cpu())
        result = {"dataset": "DIODE official validation", "subset_per_scan": args.per_scan,
                  "software": {"torch": str(torch.__version__), "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
                  "protocol": "zero adaptation; all supplied validation-selected models reported; relative inverse depth",
                  "alignment": "per-image 1st/99th inverse-depth percentiles for MSE; oracle affine fit for aligned inverse RMSE",
                  "source": "https://diode-dataset.org/", "manifest": manifest, "runs": []}

        @torch.no_grad()
        def evaluate(candidate=None, variant=None, tag="template"):
            records, rows, captions = [], [], []
            for i, z in enumerate(zs):
                latent = z[None].to(device)
                if candidate is None:
                    prediction = template
                elif "probe" in variant:
                    prediction = candidate(latent, (256, 256))[0]
                else:
                    prediction = decode_with_grad(model, candidate(latent))[0]
                record = measured_metrics(prediction, depths[i].to(device), masks[i].to(device))
                record.update(index=i, scene=manifest[i]["scene"], scan=manifest[i]["scan"])
                records.append(record)
                if i % max(1, len(zs) // 8) == 0 and len(rows) < 8:
                    mask = masks[i].to(device) & torch.isfinite(depths[i].to(device)) & (depths[i].to(device) > 0)
                    inv = depths[i].to(device).clamp_min(1e-6).reciprocal()
                    low, high = torch.quantile(inv[mask], torch.tensor([.01, .99], device=device))
                    display = ((inv - low) / (high - low)).clamp(0, 1).masked_fill(~mask, 0)
                    rows.append([rgb[i].to(device), display[None].expand(3, -1, -1), prediction])
                    captions.append(["RGB", "Measured inverse depth (masked)", tag])
            summary = {}
            for key in ("normalized_inverse_mse", "inverse_pearson", "oracle_affine_inverse_rmse"):
                values = [r[key] for r in records if r[key] is not None]
                summary[key] = sum(values) / len(values) if values else None
            save_grid(rows, captions, args.output_dir / f"{tag}.png")
            return {"summary": summary, "per_image": records}

        result["template"] = evaluate()
        for folder in args.run_dirs:
            checkpoints = sorted(folder.glob("*.pt"))
            if not (folder / "results.json").is_file():
                raise ValueError(f"Incomplete comparison directory: {folder}")
            for checkpoint in checkpoints:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                variant = state["variant"]
                if state["args"]["task"] != "depth":
                    raise ValueError("Only depth models can be evaluated against DIODE depth")
                candidate = make_candidate(variant, torch.zeros(state["initial_shape"]), state["initial_shape"][-1], state["args"]["method"], "depth").to(device)
                candidate.load_state_dict(state["state_dict"]); candidate.eval()
                with torch.no_grad():
                    sample = zs[0][None].to(device)
                    def predict_one():
                        return candidate(sample, (256, 256)) if "probe" in variant else decode_with_grad(model, candidate(sample))
                    for _ in range(3):
                        predict_one()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(10):
                        predict_one()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    inference_ms = (time.perf_counter() - started) * 100
                tag = f"{folder.name}_{checkpoint.stem}"
                run = {"checkpoint": str(checkpoint), "variant": variant, "method": state["args"]["method"],
                       "inference_ms_per_image_excluding_encoder": inference_ms,
                       "best_step": state["best_step"], "evaluation": evaluate(candidate, variant, tag)}
                result["runs"].append(run)
                (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
                print(tag, run["evaluation"]["summary"], flush=True)
        result["frozen_verified"] = not any(p.requires_grad or p.grad is not None for p in model.parameters())
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
