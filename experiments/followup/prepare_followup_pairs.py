"""Keep original train/validation pairs and add reproducibly sampled fresh evaluation images."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from experiments.intrinsics.geometry import normalize
from experiments.intrinsics.depth_feasibility import depth_to_target


def sample_manifest(old, image_root, seed=20261027, per_category=8):
    rng = random.Random(seed)
    retained = [dict(r) for r in old if r["split"] in ("train", "validation")]
    excluded_paths = {r["image"] for r in old}
    excluded_categories = {r["category"] for r in old}
    seen_categories = sorted({r["category"] for r in retained})
    candidates = sorted(p for p in image_root.iterdir() if p.is_dir() and p.name not in excluded_categories)
    rng.shuffle(candidates)
    new_categories = []
    for folder in candidates:
        if len([p for p in folder.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")]) >= per_category:
            new_categories.append(folder.name)
        if len(new_categories) == len(seen_categories):
            break
    if len(new_categories) != len(seen_categories):
        raise ValueError("Insufficient fresh categories")
    for split, categories in (("test", seen_categories), ("unseen", sorted(new_categories))):
        for category in categories:
            paths = sorted(p for p in (image_root / category).iterdir()
                           if p.suffix.lower() in (".jpeg", ".jpg", ".png") and str(p) not in excluded_paths)
            if len(paths) < per_category:
                raise ValueError(f"Insufficient fresh images in {category}")
            for path in rng.sample(paths, per_category):
                retained.append({"image": str(path), "category": category, "split": split})
    return retained


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-pairs", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-category", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261027)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    old = torch.load(args.previous_pairs, map_location="cpu", weights_only=True)
    manifest = sample_manifest(old["manifest"], args.image_root, args.seed, args.per_category)
    # Write the evaluation membership before generating any predictions.
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    import numpy as np
    from PIL import Image
    import torch.distributed as dist
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from sphere.loader import center_crop_arr
    from experiments.intrinsics.traverse import build_checkpoint_model
    dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        model = build_checkpoint_model(json.loads((args.checkpoint.parent.parent / "cfg.json").read_text()), args.checkpoint, "ema", True).cuda()
        name, revision = "depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99"
        processor = AutoImageProcessor.from_pretrained(name, revision=revision, cache_dir=str(args.cache_dir))
        teacher = AutoModelForDepthEstimation.from_pretrained(name, revision=revision, cache_dir=str(args.cache_dir)).eval().requires_grad_(False).cuda()
        index = {r["image"]: i for i, r in enumerate(old["manifest"])}
        old_hashes = {r.get("cropped_rgb_sha256") for r in old["manifest"]} - {None}
        hashes, rgb, targets, zs, tzs = set(), [], [], [], []
        old_tz = old.get("target_z", old.get("depth_z"))
        with torch.no_grad():
            for row in manifest:
                if row["split"] in ("train", "validation"):
                    i = index[row["image"]]
                    original, target, z, tz = old["rgb"][i], old["target"][i], old["z"][i], old_tz[i]
                    digest = row.get("cropped_rgb_sha256")
                    if digest is None:
                        raise ValueError("Original cache must include cropped-image hashes")
                else:
                    with Image.open(row["image"]) as image:
                        crop = center_crop_arr(image.convert("RGB"), 256)
                    array = np.array(crop)
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    if digest in old_hashes:
                        raise ValueError("Fresh image duplicates a previously evaluated crop")
                    original = torch.from_numpy(array.copy()).permute(2, 0, 1).float().cuda() / 255
                    raw = processor.post_process_depth_estimation(teacher(**processor(images=crop, return_tensors="pt").to("cuda")), target_sizes=[(256, 256)])[0]["predicted_depth"]
                    target, row["depth_normalization"] = depth_to_target(raw)
                    z = normalize(model.f(model.encode(original[None] * 2 - 1)).float())[0]
                    tz = normalize(model.f(model.encode(target[None] * 2 - 1)).float())[0]
                if digest in hashes:
                    raise ValueError("Duplicate exact crop in follow-up manifest")
                hashes.add(digest)
                row["cropped_rgb_sha256"] = digest
                for collection, value in ((rgb, original), (targets, target), (zs, z), (tzs, tz)):
                    collection.append(value.cpu())
                if len(rgb) % 24 == 0:
                    print(f"Prepared {len(rgb)}/{len(manifest)}", flush=True)
        torch.save({"rgb": torch.stack(rgb), "target": torch.stack(targets), "z": torch.stack(zs),
                    "target_z": torch.stack(tzs), "manifest": manifest}, args.output_dir / "pairs.pt")
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (args.output_dir / "protocol.json").write_text(json.dumps({"seed": args.seed, "teacher_revision": revision,
            "retained": "original train/validation", "evaluation": "new image paths and exact crops; no near-duplicate audit",
            "excluded_prior_images": len(index), "per_category": args.per_category}, indent=2))
        print("Fresh depth evaluation cache complete", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
