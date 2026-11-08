"""Summarize completed runs and paired, category-resampled depth differences."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def paired_category_interval(candidate, baseline, seed=673, samples=2000):
    """Average seeds per image, then resample categories, retaining image pairing."""
    def collect(runs):
        values = defaultdict(list)
        categories = {}
        for run in runs:
            for row in run["per_image"]:
                values[row["index"]].append(row["rgb_mse"])
                categories[row["index"]] = row.get("category") or str(row["index"])
        return {i: sum(v) / len(v) for i, v in values.items()}, categories
    a, categories = collect(candidate)
    b, _ = collect(baseline)
    if a.keys() != b.keys():
        raise ValueError("Paired comparisons require identical evaluation images")
    groups = defaultdict(list)
    for i in a:
        groups[categories[i]].append(a[i] - b[i])
    group_names = sorted(groups)
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        values = [x for name in rng.choices(group_names, k=len(group_names)) for x in groups[name]]
        boot.append(sum(values) / len(values))
    boot.sort()
    return {"mean_mse_difference": sum(a[i] - b[i] for i in a) / len(a),
            "category_bootstrap_95_percent_interval": [boot[int(.025 * samples)], boot[int(.975 * samples)]],
            "categories": len(groups), "images": len(a), "bootstrap_seed": seed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    documents = []
    for path in sorted(args.root.glob("*/results.json")):
        if path.parent.name.startswith("smoke"):
            continue
        data = json.loads(path.read_text())
        if "args" in data and "runs" in data:
            documents.append((path, data))
    lines = ["# Follow-up results", "", "Only completed run entries are summarized. Missing jobs are not successes.", "",
             "| Task | Method | Variant | Seed | Parameters | Best step | Train loss | Validation loss | Test loss | Unseen loss |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    depth = defaultdict(list)
    endpoints = defaultdict(list)
    for path, data in documents:
        for run in data["runs"]:
            task = data["args"]["task"]
            method_label = "n/a" if "probe" in run["variant"] else run["method"]
            def score(split):
                return f'{run[split]["summary"]["loss"]:.6f}' if split in run else "—"
            lines.append(f'| {task} | {method_label} | {run["variant"]} | {run["seed"]} | {run["parameter_count"]} | {run["best_step"]} | {score("train")} | {score("validation")} | {score("test")} | {score("unseen")} |')
            metric = {"depth": "rgb_mse", "normals": "mean_angle_deg", "binary": "positive_iou"}[task]
            for split in ("test", "unseen"):
                if split in run:
                    endpoints[(task, method_label, run["variant"], split, metric)].append(run[split]["summary"][metric])
            if task == "depth":
                depth[(data["pairs_sha256"], run["method"], run["variant"])].append(run)
    comparisons = []
    for (digest, method, variant), runs in depth.items():
        baseline = depth.get((digest, method, "shared"))
        if variant == "shared" or not baseline:
            continue
        # Restrict each comparison to seeds completed for both methods.
        seeds = {r["seed"] for r in runs} & {r["seed"] for r in baseline}
        for split in ("test", "unseen"):
            selected = [r[split] for r in runs if r["seed"] in seeds and split in r]
            shared = [r[split] for r in baseline if r["seed"] in seeds and split in r]
            if selected and shared:
                comp = paired_category_interval(selected, shared)
                comp.update(method=method, variant=variant, split=split, seeds=sorted(seeds), pairs_sha256=digest)
                comparisons.append(comp)
    lines += ["", "## Task metrics", "", "Mean ± sample standard deviation across available seeds. One seed has no standard deviation. Probe methods do not use a movement rule.", "",
              "| Task | Method | Variant | Split | Metric | Mean | Seed SD | Seeds |",
              "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for (task, method, variant, split, metric), values in sorted(endpoints.items()):
        sd = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "—"
        lines.append(f"| {task} | {method} | {variant} | {split} | {metric} | {statistics.mean(values):.6f} | {sd} | {len(values)} |")
    lines += ["", "## Paired depth differences", "", "Negative MSE differences favor the candidate. Intervals resample categories after averaging matched seeds per image; they do not establish broad population generalization.", ""]
    for row in comparisons:
        low, high = row["category_bootstrap_95_percent_interval"]
        lines.append(f'- {row["method"]}/{row["variant"]}, {row["split"]}, seeds {row["seeds"]}: ΔMSE {row["mean_mse_difference"]:.6f}, 95% interval [{low:.6f}, {high:.6f}].')
    for path in sorted(args.root.glob("diode*/results.json")):
        data = json.loads(path.read_text())
        lines += ["", f"## Measured depth: {path.parent.name}", "", f'{len(data["manifest"])} images; relative inverse-depth evaluation, no adaptation.', "",
                  "| Model | Normalized inverse MSE | Inverse Pearson |",
                  "| --- | ---: | ---: |"]
        for name, metrics in [("Training template", data["template"]["summary"])] + [(r["checkpoint"], r["evaluation"]["summary"]) for r in data["runs"]]:
            corr = f'{metrics["inverse_pearson"]:.6f}' if metrics["inverse_pearson"] is not None else "undefined"
            lines.append(f'| {name} | {metrics["normalized_inverse_mse"]:.6f} | {corr} |')
    for path in sorted(args.root.glob("diode*/teacher-results.json")):
        data = json.loads(path.read_text())
        metrics = data["summary"]
        lines += ["", "## Pinned teacher reference on measured depth", "",
                  f'{len(data["images"])} images; normalized inverse MSE {metrics["normalized_inverse_mse"]:.6f}; inverse Pearson {metrics["inverse_pearson"]:.6f}.']
    for path in sorted(args.root.glob("*controls/results.json")):
        data = json.loads(path.read_text())
        if "controls" not in data:
            continue
        lines += ["", f"## Binary mask controls: {path.parent.name}", "",
                  "| Split | Control/model | Own-target sky IoU | Shuffled-target sky IoU |",
                  "| --- | --- | ---: | ---: |"]
        for split, controls in data["controls"].items():
            for name, record in controls.items():
                lines.append(f'| {split} | {name} | {record["own_target"]["positive_iou"]:.6f} | {record["cyclic_shuffled_target"]["positive_iou"]:.6f} |')
        for run in data["runs"]:
            for split in ("test", "unseen"):
                for name, record in run[split].items():
                    lines.append(f'| {split} | {run["method"]}/{run["seed"]}/{name} | {record["own_target"]["positive_iou"]:.6f} | {record["cyclic_shuffled_target"]["positive_iou"]:.6f} |')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps({"sources": [str(p) for p, _ in documents], "comparisons": comparisons}, indent=2))
    print(f"Summarized {len(documents)} result files into {args.output}")


if __name__ == "__main__":
    main()
