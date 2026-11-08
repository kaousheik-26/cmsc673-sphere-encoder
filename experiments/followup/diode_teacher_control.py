"""Check the pinned pseudo-depth teacher against the same measured DIODE subset."""
import argparse
import json
from pathlib import Path

import torch

from experiments.intrinsics.depth_feasibility import depth_to_target
from .evaluate_diode import aligned_sensor_crop, measured_metrics, select_images


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--per-scan", type=int, default=4)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    import numpy as np
    from PIL import Image
    from sphere.loader import center_crop_arr
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from experiments.intrinsics.traverse import save_grid
    device = torch.device(args.device)
    revision = "5426e4f0f36572d16453bbda7a8389317b1bef99"
    name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(name, revision=revision, cache_dir=str(args.cache_dir), local_files_only=True)
    teacher = AutoModelForDepthEstimation.from_pretrained(name, revision=revision, cache_dir=str(args.cache_dir), local_files_only=True).eval().requires_grad_(False).to(device)
    paths = select_images(args.dataset_root, args.per_scan)
    result = {"teacher": name, "revision": revision, "images": [str(p) for p in paths], "per_image": []}
    rows, captions = [], []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            crop = center_crop_arr(image.convert("RGB"), 256)
        output = teacher(**processor(images=crop, return_tensors="pt").to(device))
        raw = processor.post_process_depth_estimation(output, target_sizes=[(256, 256)])[0]["predicted_depth"]
        prediction, _ = depth_to_target(raw)
        depth, mask = aligned_sensor_crop(np.load(path.with_name(path.stem + "_depth.npy")), np.load(path.with_name(path.stem + "_depth_mask.npy")))
        record = measured_metrics(prediction, depth.to(device), mask.to(device))
        record.update(image=str(path), scene=str(path.parent.parent.relative_to(args.dataset_root)))
        result["per_image"].append(record)
        if index % max(1, len(paths) // 8) == 0 and len(rows) < 8:
            original = torch.from_numpy(np.array(crop).copy()).permute(2, 0, 1).float().to(device) / 255
            valid = mask.to(device) & torch.isfinite(depth.to(device)) & (depth.to(device) > 0)
            inverse = depth.to(device).clamp_min(1e-6).reciprocal()
            low, high = torch.quantile(inverse[valid], torch.tensor([.01, .99], device=device))
            display = ((inverse - low) / (high - low)).clamp(0, 1).masked_fill(~valid, 0)
            rows.append([original, display[None].expand(3, -1, -1), prediction])
            captions.append(["RGB", "Measured inverse depth", "Pinned depth teacher"])
        if (index + 1) % 10 == 0:
            print(f"Teacher control {index+1}/{len(paths)}", flush=True)
    result["summary"] = {}
    for key in ("normalized_inverse_mse", "inverse_pearson", "oracle_affine_inverse_rmse"):
        values = [r[key] for r in result["per_image"] if r[key] is not None]
        result["summary"][key] = sum(values) / len(values) if values else None
    save_grid(rows, captions, args.output_dir / "teacher-reference.png")
    (args.output_dir / "teacher-results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
