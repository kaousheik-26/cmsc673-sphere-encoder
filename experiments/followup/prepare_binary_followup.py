"""Label the already fixed fresh RGB split with the pinned segmentation teacher."""
import argparse
import json
from pathlib import Path

import torch

from experiments.intrinsics.property_metrics import segmentation_palette


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fresh-depth-pairs", type=Path, required=True)
    p.add_argument("--previous-segmentation-pairs", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    fresh = torch.load(args.fresh_depth_pairs, map_location="cpu", weights_only=True)
    previous = torch.load(args.previous_segmentation_pairs, map_location="cpu", weights_only=True)
    index = {r["image"]: i for i, r in enumerate(previous["manifest"])}
    revision = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(name, revision=revision, cache_dir=str(args.cache_dir), local_files_only=True)
    teacher = AutoModelForSemanticSegmentation.from_pretrained(name, revision=revision, cache_dir=str(args.cache_dir), local_files_only=True).eval().requires_grad_(False).cuda()
    if teacher.config.id2label[2] != "sky":
        raise ValueError("Unexpected label mapping")
    palette, labels, targets = segmentation_palette(), [], []
    for i, row in enumerate(fresh["manifest"]):
        if row["split"] in ("train", "validation"):
            old_id = index[row["image"]]
            if row["cropped_rgb_sha256"] != previous["manifest"][old_id]["cropped_rgb_sha256"]:
                raise ValueError("Retained RGB crop changed")
            label = previous["labels"][old_id]
        else:
            array = (fresh["rgb"][i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            image = Image.fromarray(array)
            output = teacher(**processor(images=image, return_tensors="pt").to("cuda"))
            label = processor.post_process_semantic_segmentation(output, target_sizes=[(256, 256)])[0].cpu()
        labels.append(label)
        targets.append(palette[label].permute(2, 0, 1))
        if (i + 1) % 24 == 0:
            print(f"Fresh segmentation labels {i+1}/{len(fresh['manifest'])}", flush=True)
    torch.save({"rgb": fresh["rgb"], "z": fresh["z"], "labels": torch.stack(labels),
                "target": torch.stack(targets), "manifest": fresh["manifest"]}, args.output_dir / "pairs.pt")
    (args.output_dir / "protocol.json").write_text(json.dumps({"teacher_revision": revision, "positive_id": 2,
        "membership": "fresh depth manifest, unchanged", "training": "none; evaluate already trained sky-mask edits"}, indent=2))


if __name__ == "__main__":
    main()
