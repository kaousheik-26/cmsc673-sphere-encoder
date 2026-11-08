"""Compare frozen-decoder edits and direct probes on cached property pairs."""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .adaptive_transform import AdaptiveTransform, PropertyProbe
from experiments.intrinsics.depth_feasibility import similarity
from experiments.intrinsics.geometry import geodesic_move, normalize, normalized_addition, project_tangent
from experiments.intrinsics.property_metrics import normal_metrics, segmentation_metrics
from experiments.intrinsics.train_shared_depth import decode_with_grad


VARIANTS = ("shared", "adaptive-size", "subspace4", "subspace8",
            "budget4", "budget8", "linear-probe", "nonlinear-probe")


def validate_pairs(data):
    """Reject missing splits, nonfinite data, and exact crop/path overlap."""
    n = len(data["manifest"])
    for key in ("z", "target", "rgb"):
        if len(data[key]) != n or not torch.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} cache")
    paths = [r["image"] for r in data["manifest"]]
    hashes = [r.get("cropped_rgb_sha256") for r in data["manifest"]]
    if any(h is None for h in hashes):
        hashes = [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in data["rgb"]]
    if len(set(paths)) != n or len(set(hashes)) != n:
        raise ValueError("Duplicate filenames or exact crops")
    splits = {s: [i for i, r in enumerate(data["manifest"]) if r["split"] == s]
              for s in ("train", "validation", "test", "unseen")}
    if not all(splits[s] for s in ("train", "validation", "test")):
        raise ValueError("Train, validation, and test must be nonempty")
    train_categories = {data["manifest"][i].get("category") for i in splits["train"]} - {None}
    unseen_categories = {data["manifest"][i].get("category") for i in splits["unseen"]} - {None}
    if train_categories & unseen_categories:
        raise ValueError("Unseen categories overlap training categories")
    return {k: v for k, v in splits.items() if v}


def task_target(data, task, positive_labels):
    if task != "binary":
        return data["target"].float()
    if not positive_labels or data.get("labels") is None:
        raise ValueError("Binary task needs label cache and explicit --positive-labels")
    mask = torch.zeros_like(data["labels"], dtype=torch.bool)
    for label in positive_labels:
        mask |= data["labels"] == label
    return mask[:, None].float().expand(-1, 3, -1, -1).contiguous()


def loss_for(pred, truth, task):
    if task == "normals":
        return (1 - (F.normalize(pred * 2 - 1, dim=1) *
                     F.normalize(truth * 2 - 1, dim=1)).sum(1)).mean()
    if task == "binary":
        # Soft nearest-color logits for black/white masks, through the frozen RGB decoder.
        gray = pred.mean(1)
        logits = torch.stack((-gray.square(), -(gray - 1).square()), 1) / .1
        return F.cross_entropy(logits, truth[:, 0].long())
    return (pred - truth).square().mean()


def make_candidate(variant, initial, channels, method, task):
    if "probe" in variant:
        return PropertyProbe(channels, 3, 0 if variant == "linear-probe" else 128)
    rank = int(variant[-1]) if variant[-1].isdigit() else 1
    return AdaptiveTransform(initial, rank, method,
                             conditional_size=variant != "shared",
                             conditional_direction=rank > 1,
                             factorized=variant.startswith("budget"))


@torch.no_grad()
def geometry_audit(z, direction):
    """Compare matched tangent addition with geodesics at several fixed angles."""
    tangent = normalize(project_tangent(z, direction))
    errors = {}
    for angle in (0., 5., 20., 40., 60., 75.):
        a = geodesic_move(z, direction, angle)
        b = normalized_addition(z, tangent, torch.tan(torch.deg2rad(torch.tensor(angle, device=z.device))))
        errors[str(angle)] = (a - b).abs().max().item()
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", choices=("depth", "normals", "binary"), default="depth")
    p.add_argument("--positive-labels", type=int, nargs="+")
    p.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    p.add_argument("--method", choices=("geodesic", "addition", "tangent"), default="geodesic")
    p.add_argument("--seeds", type=int, nargs="+", default=[673, 674, 675])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=1, help="Microbatches per optimizer step")
    p.add_argument("--device", default="cuda")
    p.add_argument("--evaluation-status", default="exploratory: previously inspected cached splits")
    args = p.parse_args()
    if args.steps < 0 or args.eval_every < 1 or args.batch_size < 1 or args.accumulation < 1:
        p.error("Invalid training budget")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(673)
    data = torch.load(args.pairs, map_location="cpu", weights_only=True)
    splits = validate_pairs(data)
    target = task_target(data, args.task, args.positive_labels)
    if args.task == "binary":
        for split in ("train", "validation"):
            fraction = target[splits[split], 0].mean().item()
            if not 0 < fraction < 1:
                raise ValueError(f"Binary {split} split must contain both labels")
    z = data["z"].float()
    device = torch.device(args.device)
    model, loader, dist = None, None, None
    started = time.time()
    needs_decoder = any("probe" not in v for v in args.variants)
    if needs_decoder:
        if args.checkpoint is None:
            p.error("--checkpoint is required for decoder edits")
        import torch.distributed as dist
        from experiments.intrinsics import traverse as loader
        dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        if needs_decoder:
            config = json.loads((args.checkpoint.parent.parent / "cfg.json").read_text())
            model = loader.build_checkpoint_model(config, args.checkpoint, "ema", True).to(device)
            if model.spherify_mode != "global":
                raise ValueError("Adaptive study currently requires global sphere geometry")
        train_ids = splits["train"]
        if needs_decoder:
            if args.task == "binary":
                with torch.no_grad():
                    tz = torch.cat([normalize(model.f(model.encode(target[i:i+1].to(device) * 2 - 1)).float()).cpu()
                                    for i in train_ids])
                initial = (tz - z[train_ids]).mean(0, keepdim=True)
            else:
                target_z = data.get("target_z", data.get("depth_z"))
                if target_z is None:
                    raise ValueError("Cache lacks target latents")
                initial = (target_z[train_ids] - z[train_ids]).mean(0, keepdim=True)
        else:
            initial = torch.ones_like(z[:1])

        digest = hashlib.sha256()
        with args.pairs.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "pairs_sha256": digest.hexdigest(),
                  "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ("compare_adaptive.py", "adaptive_transform.py", "geometry.py")},
                  "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
                  "selection": "validation task loss, no test selection",
                  "split_counts": {k: len(v) for k, v in splits.items()},
                  "manifest": data["manifest"], "runs": []}
        result["geometry_max_abs_errors"] = geometry_audit(z[:4].to(device), initial.to(device))

        def predict(candidate, latent, variant):
            if "probe" in variant:
                return candidate(latent, target.shape[-2:])
            return decode_with_grad(model, candidate(latent))

        @torch.no_grad()
        def evaluate(candidate, variant, ids, tag=None, fixed=None, reconstruct=False):
            records, rows, captions = [], [], []
            cm = torch.zeros(2, 2, dtype=torch.long, device=device)
            for pos, i in enumerate(ids):
                latent = z[i:i+1].to(device)
                truth = target[i:i+1].to(device)
                if fixed is not None:
                    pred = fixed[None]
                elif reconstruct:
                    encoded = normalize(model.f(model.encode(truth * 2 - 1)).float())
                    pred = decode_with_grad(model, encoded)
                else:
                    pred = predict(candidate, latent, variant)
                record = similarity(pred[0], truth[0])
                record.update(index=i, category=data["manifest"][i].get("category"),
                              loss=loss_for(pred, truth, args.task).item())
                if args.task == "normals":
                    record.update(normal_metrics(pred[0], truth[0]))
                elif args.task == "binary":
                    labels = (pred[0].mean(0) >= .5).long()
                    confusion = torch.bincount((truth[0, 0].long() * 2 + labels).flatten(), minlength=4).reshape(2, 2)
                    cm += confusion
                    record.update(segmentation_metrics(confusion))
                    record["positive_iou"] = (confusion[1, 1] / (confusion[1].sum() + confusion[:, 1].sum() - confusion[1, 1]).clamp_min(1)).item()
                if "probe" not in variant and fixed is None and not reconstruct:
                    moved = candidate(latent)
                    record["angle_deg"] = torch.rad2deg(torch.acos(F.cosine_similarity(latent.flatten(1), moved.flatten(1)).clamp(-1, 1))).item()
                    record["radius_error"] = ((moved.norm() - latent.norm()).abs() / latent.norm()).item()
                records.append(record)
                if tag and loader is not None and pos < 8:
                    rows.append([data["rgb"][i].to(device), truth[0], pred[0]])
                    captions.append(["RGB", "Teacher target", tag])
            keys = [k for k, v in records[0].items() if isinstance(v, (int, float)) and k != "index"]
            summary = {k: sum(r[k] for r in records if r.get(k) is not None) / sum(r.get(k) is not None for r in records) for k in keys}
            if args.task == "binary":
                summary.update(segmentation_metrics(cm))
                summary["positive_iou"] = (cm[1, 1] / (cm[1].sum() + cm[:, 1].sum() - cm[1, 1]).clamp_min(1)).item()
            if rows:
                loader.save_grid(rows, captions, args.output_dir / f"{tag}.png")
            return {"summary": summary, "per_image": records}

        template = target[train_ids].mean(0).to(device)
        result["template"] = {s: evaluate(None, "fixed", ids, fixed=template) for s, ids in splits.items() if s != "train"}
        if model is not None and args.task == "binary":
            result["target_reconstruction"] = {s: evaluate(None, "fixed", ids, reconstruct=True) for s, ids in splits.items() if s != "train"}
        for variant in args.variants:
            for seed in args.seeds:
                torch.manual_seed(seed)
                candidate = make_candidate(variant, initial, z.shape[-1], args.method, args.task).to(device)
                if "probe" in variant:
                    optimizer = torch.optim.Adam(candidate.parameters(), lr=.003)
                else:
                    groups = [{"params": [candidate.raw_size], "lr": .01}]
                    if candidate.factorized:
                        groups.append({"params": [candidate.left, candidate.right], "lr": .01})
                    else:
                        groups.append({"params": [candidate.dictionary], "lr": .03})
                    if candidate.conditioner is not None:
                        groups.append({"params": candidate.conditioner.parameters(), "lr": .003})
                    optimizer = torch.optim.Adam(groups)
                rng = torch.Generator().manual_seed(seed)
                best, best_state, best_step = float("inf"), None, 0
                curve = []
                train_tensor = torch.tensor(train_ids)
                run_started = time.time()
                for step in range(args.steps + 1):
                    if step % args.eval_every == 0 or step == args.steps:
                        candidate.eval()
                        score = evaluate(candidate, variant, splits["validation"])["summary"]["loss"]
                        if score < best:
                            best, best_state, best_step = score, copy.deepcopy(candidate.state_dict()), step
                        curve.append({"step": step, "validation_loss": score})
                        print(f"{args.task}/{args.method}/{variant} seed={seed} step={step} val={score:.6f}", flush=True)
                    if step == args.steps:
                        break
                    candidate.train()
                    optimizer.zero_grad(set_to_none=True)
                    for _ in range(args.accumulation):
                        ids = train_tensor[torch.randint(len(train_ids), (args.batch_size,), generator=rng)]
                        prediction = predict(candidate, z[ids].to(device), variant)
                        loss = loss_for(prediction, target[ids].to(device), args.task) / args.accumulation
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite loss")
                        loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in candidate.parameters()):
                        raise RuntimeError("Missing or nonfinite candidate gradients")
                    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
                    optimizer.step()
                candidate.load_state_dict(best_state)
                candidate.eval()
                with torch.no_grad():
                    sample = z[splits["validation"][:1]].to(device)
                    for _ in range(3):
                        predict(candidate, sample, variant)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    inference_started = time.perf_counter()
                    for _ in range(10):
                        predict(candidate, sample, variant)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    inference_ms = (time.perf_counter() - inference_started) * 100
                run = {"variant": variant, "seed": seed, "method": args.method,
                       "parameter_count": sum(p.numel() for p in candidate.parameters()),
                       "inference_ms_per_image": inference_ms,
                       "best_step": best_step, "validation_loss": best, "curve": curve,
                       "training_seconds": time.time() - run_started}
                for split, ids in splits.items():
                    tag = f"{variant}_{seed}_{split}" if seed == args.seeds[0] and split in ("test", "unseen") else None
                    run[split] = evaluate(candidate, variant, ids, tag=tag)
                if model is not None and any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise RuntimeError("Frozen model acquired gradients")
                torch.save({"state_dict": best_state, "variant": variant, "initial_shape": list(initial.shape),
                            "args": result["args"], "best_step": best_step}, args.output_dir / f"{variant}_{seed}.pt")
                result["runs"].append(run)
                result["frozen_verified"] = model is None or not any(p.requires_grad or p.grad is not None for p in model.parameters())
                result["elapsed_seconds"] = time.time() - started
                (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print("Adaptive comparison complete", flush=True)
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
