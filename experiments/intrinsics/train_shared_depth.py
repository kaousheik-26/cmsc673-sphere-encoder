"""Train and evaluate shared RGB-to-pseudo-depth edits with frozen weights."""
import argparse
import copy
import hashlib
import importlib
import json
import time
from pathlib import Path

import torch

from .depth_feasibility import depth_to_target, similarity
from .geometry import geodesic_move, normalize
from .shared_transform import SharedTransform


def decode_with_grad(model, z):
    """Bypass no_grad inference wrapper while keeping decoder weights frozen."""
    y = torch.full((z.shape[0],), model.num_classes, device=z.device, dtype=torch.long) if model.num_classes else None
    alpha = model._dec_angle(z.shape[0], 0., z.device)
    return model.decoder(z, y, alpha=alpha) * 0.5 + 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--loader-module", default="experiments.intrinsics.traverse")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[673, 674, 675])
    parser.add_argument("--per-image-steps", type=int, default=150)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU required")
    torch.set_num_threads(4)
    torch.manual_seed(673)
    import torch.distributed as dist
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    loader = importlib.import_module(args.loader_module)
    dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    started = time.time()
    try:
        config = json.loads((args.checkpoint.parent.parent / "cfg.json").read_text())
        model = loader.build_checkpoint_model(config, args.checkpoint, "ema", True).cuda()
        assert not any(p.requires_grad for p in model.parameters())
        manifest = json.loads(args.manifest.read_text())
        paths = [Path(row["image"]) for row in manifest]
        split = {name: [i for i, row in enumerate(manifest) if row["split"] == name] for name in ("train", "validation", "test")}
        if not all(split.values()) or len(set(paths)) != len(paths):
            raise ValueError("All splits must be nonempty and filenames disjoint")
        processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", cache_dir=str(args.cache_dir), revision="5426e4f0f36572d16453bbda7a8389317b1bef99")
        teacher = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", cache_dir=str(args.cache_dir), revision="5426e4f0f36572d16453bbda7a8389317b1bef99").eval().requires_grad_(False).cuda()
        from sphere.loader import center_crop_arr
        originals, targets, latents, depth_latents, hashes = [], [], [], [], []
        with torch.no_grad():
            for i, path in enumerate(paths):
                with Image.open(path) as image:
                    cropped = center_crop_arr(image.convert("RGB"), model.decoder.input_size)
                array = np.array(cropped)
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f"Duplicate cropped image: {path}; revise manifest")
                hashes.append(digest)
                rgb = torch.from_numpy(array.copy()).permute(2, 0, 1).float().cuda() / 255
                raw = processor.post_process_depth_estimation(teacher(**processor(images=cropped, return_tensors="pt").to("cuda")), target_sizes=[(256, 256)])[0]["predicted_depth"]
                target, bounds = depth_to_target(raw)
                manifest[i]["depth_normalization"] = bounds
                manifest[i]["cropped_rgb_sha256"] = digest
                originals.append(rgb.cpu())
                targets.append(target.cpu())
                latents.append(normalize(model.f(model.encode(rgb[None] * 2 - 1)).float(), model.spherify_mode)[0].cpu())
                depth_latents.append(normalize(model.f(model.encode(target[None] * 2 - 1)).float(), model.spherify_mode)[0].cpu())
                if (i + 1) % 12 == 0:
                    print(f"Prepared {i+1}/{len(paths)} pairs", flush=True)
        del teacher
        torch.cuda.empty_cache()
        rgb = torch.stack(originals)
        target = torch.stack(targets)
        z = torch.stack(latents)
        zd = torch.stack(depth_latents)
        torch.save({"rgb": rgb, "target": target, "z": z, "depth_z": zd, "manifest": manifest}, args.output_dir / "pairs.pt")
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        template = target[split["train"]].mean(0).cuda()

        @torch.no_grad()
        def evaluate(edit, indices, save=False, tag=""):
            records, rows, labels = [], [], []
            for index in indices:
                latent = z[index:index+1].cuda()
                moved = edit(latent) if edit is not None else latent
                prediction = decode_with_grad(model, moved)[0]
                truth = target[index].cuda()
                record = similarity(prediction, truth)
                record.update(loader.latent_stats(latent, moved, model.spherify_mode))
                # Test whether the output matches its own geometry better
                # than the next test image's geometry (fixed cyclic shuffle).
                other = indices[(indices.index(index) + 1) % len(indices)]
                record["shuffled_target_mse"] = (prediction - target[other].cuda()).square().mean().item()
                record["image_index"] = index
                records.append(record)
                if save:
                    zero = decode_with_grad(model, latent)[0]
                    depth_rec = decode_with_grad(model, zd[index:index+1].cuda())[0]
                    row = [rgb[index].cuda(), truth, zero, prediction, template, depth_rec]
                    label = ["Input RGB", "Pseudo-depth target", "Zero edit", tag, "Training mean target", "Target reconstruction"]
                    loader.save_grid([row], [label], args.output_dir / f"{tag}_{index:03d}.png")
                    rows.append(row)
                    labels.append(label)
            summary = {}
            for key in ["rgb_mse", "gray_mse", "gray_pearson", "channel_spread_mean", "shuffled_target_mse", "max_relative_radius_error"]:
                values = [r[key] for r in records if r[key] is not None]
                summary[key] = sum(values) / len(values) if values else None
            if save:
                loader.save_grid(rows, labels, args.output_dir / f"{tag}_overview.png")
            return {"summary": summary, "per_image": records}

        result = {"checkpoint": str(args.checkpoint), "split_counts": {k: len(v) for k, v in split.items()},
                  "conditioning": "null class, decoder angle 0", "steps": args.steps, "seeds": args.seeds,
                  "depth_model_revision": "5426e4f0f36572d16453bbda7a8389317b1bef99", "runs": []}
        result["zero_edit"] = evaluate(None, split["test"])
        result["training_mean_template"] = {"rgb_mse": sum(similarity(template, target[i].cuda())["rgb_mse"] for i in split["test"]) / len(split["test"])}
        result["constant_oracle"] = {"rgb_mse": sum((target[i] - target[i].mean()).square().mean().item() for i in split["test"]) / len(split["test"])}
        direction_mean = (zd[split["train"]] - z[split["train"]]).mean(0, keepdim=True).cuda()
        for method in ["geodesic", "addition"]:
            for seed in args.seeds:
                torch.manual_seed(seed)
                edit = SharedTransform(direction_mean + 0.05 * torch.randn_like(direction_mean), method, model.spherify_mode).cuda()
                optimizer = torch.optim.Adam([{"params": [edit.direction], "lr": 0.03}, {"params": [edit.raw_size], "lr": 0.01}])
                curve, best, best_state, best_step = [], float("inf"), None, 0
                rng = torch.Generator().manual_seed(seed)
                for step in range(args.steps + 1):
                    if step % 50 == 0 or step == args.steps:
                        validation = evaluate(edit, split["validation"])["summary"]["rgb_mse"]
                        curve.append({"step": step, "validation_mse": validation, "size": edit.size().item()})
                        if validation < best:
                            best, best_state, best_step = validation, copy.deepcopy(edit.state_dict()), step
                        print(f"{method} seed={seed} step={step}: validation MSE={validation:.6f}, size={edit.size().item():.3f}", flush=True)
                    if step == args.steps:
                        break
                    batch = torch.tensor(split["train"])[torch.randint(len(split["train"]), (4,), generator=rng)]
                    optimizer.zero_grad(set_to_none=True)
                    output = decode_with_grad(model, edit(z[batch].cuda()))
                    loss = (output - target[batch].cuda()).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in edit.parameters()):
                        raise RuntimeError("Editing gradients missing or nonfinite")
                    torch.nn.utils.clip_grad_norm_(edit.parameters(), 1.)
                    optimizer.step()
                edit.load_state_dict(best_state)
                assert not any(p.grad is not None for p in model.parameters())
                tag = f"{method}_seed{seed}"
                torch.save({"state_dict": best_state, "method": method, "seed": seed, "best_step": best_step}, args.output_dir / f"{tag}.pt")
                test = evaluate(edit, split["test"], save=seed == args.seeds[0], tag=tag)
                train = evaluate(edit, split["train"])
                # Same shared movement size with an untrained random direction.
                random_edit = SharedTransform(torch.randn_like(direction_mean), method, model.spherify_mode).cuda()
                random_edit.raw_size.data.copy_(edit.raw_size.data)
                random_result = evaluate(random_edit, split["test"])
                result["runs"].append({"method": method, "seed": seed, "best_validation_mse": best, "best_step": best_step,
                                       "size": edit.size().item(), "curve": curve, "train": train, "test": test, "random": random_result})
                (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        # Per-image optimized direction is a finite-budget reference, not a
        # guaranteed upper bound. Only first four held-out images, no sharing.
        references = []
        for index in split["test"][::max(1, len(split["test"]) // 4)][:4]:
            edit = SharedTransform((zd[index:index+1] - z[index:index+1]).cuda(), "geodesic", model.spherify_mode).cuda()
            optimizer = torch.optim.Adam([{"params": [edit.direction], "lr": 0.03}, {"params": [edit.raw_size], "lr": 0.01}])
            latent, truth = z[index:index+1].cuda(), target[index:index+1].cuda()
            for step in range(args.per_image_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = (decode_with_grad(model, edit(latent)) - truth).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(edit.parameters(), 1.)
                optimizer.step()
            references.append(evaluate(edit, [index], save=True, tag="per_image"))
            print(f"Per-image reference {index} completed", flush=True)
        result["per_image_optimized_references"] = references
        result["model_weights_frozen_verified"] = not any(p.requires_grad or p.grad is not None for p in model.parameters())
        result["elapsed_seconds"] = time.time() - started
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        print("Shared-direction experiment completed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
