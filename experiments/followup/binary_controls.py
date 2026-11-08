"""Audit sky-mask edits against zero edits, templates, random edits, and shuffled targets."""
import argparse
import json
from pathlib import Path

import torch

from .compare_adaptive import make_candidate, task_target, validate_pairs
from experiments.intrinsics.property_metrics import segmentation_metrics
from experiments.intrinsics.train_shared_depth import decode_with_grad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--positive-labels", type=int, nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    import torch.distributed as dist
    from experiments.intrinsics.traverse import build_checkpoint_model, save_grid
    dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        data = torch.load(args.pairs, map_location="cpu", weights_only=True)
        splits = validate_pairs(data)
        target = task_target(data, "binary", args.positive_labels)
        template = target[splits["train"]].mean(0).to(device)
        model = build_checkpoint_model(json.loads((args.checkpoint.parent.parent / "cfg.json").read_text()), args.checkpoint, "ema", True).to(device)

        @torch.no_grad()
        def evaluate(edit, indices, name, fixed=None, images=False, probe=False):
            cm = torch.zeros(2, 2, dtype=torch.long, device=device)
            shuffled_cm = torch.zeros_like(cm)
            rows, labels = [], []
            display_positions = set(torch.linspace(0, len(indices) - 1, min(8, len(indices))).round().long().tolist())
            for position, index in enumerate(indices):
                latent = data["z"][index:index+1].to(device)
                if fixed is not None:
                    prediction = fixed
                elif probe:
                    prediction = edit(latent, target.shape[-2:])[0]
                else:
                    prediction = decode_with_grad(model, latent if edit is None else edit(latent))[0]
                binary = (prediction.mean(0) >= .5).long()
                truth = target[index, 0].to(device).long()
                other = target[indices[(position + 1) % len(indices)], 0].to(device).long()
                cm += torch.bincount((truth * 2 + binary).flatten(), minlength=4).reshape(2, 2)
                shuffled_cm += torch.bincount((other * 2 + binary).flatten(), minlength=4).reshape(2, 2)
                if images and position in display_positions:
                    zero = decode_with_grad(model, latent)[0]
                    mask_image = binary.float()[None].expand(3, -1, -1)
                    rows.append([data["rgb"][index].to(device), target[index].to(device), zero, prediction, mask_image])
                    labels.append(["RGB", "Teacher sky mask", "Zero edit", name, "Thresholded prediction"])
            def score(matrix):
                result = segmentation_metrics(matrix)
                result["positive_iou"] = (matrix[1, 1] / (matrix[1].sum() + matrix[:, 1].sum() - matrix[1, 1]).clamp_min(1)).item()
                result["confusion"] = matrix.cpu().tolist()
                return result
            if rows:
                save_grid(rows, labels, args.output_dir / f"{name}.png")
            return {"own_target": score(cm), "cyclic_shuffled_target": score(shuffled_cm)}

        result = {"positive_label_ids": args.positive_labels, "source_pairs": str(args.pairs), "controls": {}, "runs": []}
        for split in ("test", "unseen"):
            ids = splits[split]
            result["controls"][split] = {
                "zero_edit": evaluate(None, ids, "zero"),
                "training_template": evaluate(None, ids, "template", fixed=template),
                "all_negative": evaluate(None, ids, "negative", fixed=torch.zeros_like(template))}
        for folder in args.run_dirs:
            for checkpoint in sorted(folder.glob("*.pt")):
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if state["args"]["task"] != "binary" or state["args"]["positive_labels"] != args.positive_labels:
                    raise ValueError("Incompatible binary checkpoint")
                initial = torch.ones(state["initial_shape"])
                edit = make_candidate(state["variant"], initial, initial.shape[-1], state["args"]["method"], "binary").to(device)
                edit.load_state_dict(state["state_dict"]); edit.eval()
                is_probe = "probe" in state["variant"]
                if state["variant"] != "shared" and not is_probe:
                    raise ValueError("Random-size control currently supports shared edits only")
                seed = int(checkpoint.stem.rsplit("_", 1)[1])
                if not is_probe:
                    torch.manual_seed(seed + 1000)
                    random_edit = make_candidate("shared", torch.randn_like(initial), initial.shape[-1], state["args"]["method"], "binary").to(device)
                    random_edit.raw_size.data.copy_(edit.raw_size.data)
                record = {"checkpoint": str(checkpoint), "seed": seed, "variant": state["variant"],
                          "method": state["args"]["method"] if not is_probe else state["variant"]}
                for split in ("test", "unseen"):
                    record[split] = {"learned": evaluate(edit, splits[split], f"{folder.name}_{state['variant']}_{split}", images=seed == 673, probe=is_probe)}
                    if not is_probe:
                        record[split]["random_same_size"] = evaluate(random_edit, splits[split], "random")
                result["runs"].append(record)
                (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
                print(folder.name, record["unseen"], flush=True)
        result["frozen_verified"] = not any(p.requires_grad or p.grad is not None for p in model.parameters())
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
