"""
collect the analysis runs into one results table: prints the markdown tables
for the readme and writes the data blocks of the results website (index.html).

reads, under <dev_dir>/inverse/<job_dir>/:
    analysis/<task>/summary.csv   32 val images, every strength (the curves)
    <figure run>/summary.csv      the 10 val images shown in each figure
    latent/direct_summary.json    the one-step latent fix (shift_direct.py)

usage:
    python inverse/summarize.py
"""

import argparse
import csv
import json
import os.path as osp
import re
from collections import defaultdict

HERE = osp.dirname(osp.abspath(__file__))

# task -> how it is shown. "mu" is the anchor weight used in the figure; the
# analysis compares it against no anchor (mu=0) at every strength
TASKS = [
    dict(id="deblur", name="Blur", what="Gaussian blur std", unit="px",
         mu=1000, figure="figures/deblur_12.jpg", figure_run="strong10/deblur_12",
         figure_level=12, figure_passes=[1, 3, 5]),
    dict(id="denoise", name="Noise", what="noise std", unit="",
         mu=0.05, figure="figures/denoise_0.5.jpg", figure_run="strong10/denoise_0.5",
         figure_level=0.5, figure_passes=[1, 2, 3]),
    dict(id="dehaze", name="Haze", what="transmission in the densest patch",
         unit="", mu=25, figure="figures/dehaze_0.1.jpg", figure_run="haze/val10_undo",
         figure_level=0.1, figure_passes=[1, 3, 5]),
    dict(id="overexpose", name="Too bright", what="exposure factor", unit="x",
         mu=10, figure="figures/overexpose_4.jpg", figure_run="exposure/over_val10",
         figure_level=4, figure_passes=[1, 3, 5]),
]


def read_summary(path):
    """{level: {config: {step: (psnr, ssim, lpips, n)}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for row in csv.DictReader(f):
            out[float(row["level"])][row["config"]][int(row["step"])] = (
                float(row["psnr"]), float(row["ssim"]), float(row["lpips"]),
                int(row["n"]),
            )
    return out


def point(v, step=None):
    return {"step": step, "psnr": round(v[0], 2), "ssim": round(v[1], 3),
            "lpips": round(v[2], 3)}


def mu_of(config):
    """'10x5|mu=25' -> 25.0; references ('degraded', ...) -> None"""
    m = re.search(r"\|mu=(.+)$", config)
    return float(m.group(1)) if m else None


def level_block(configs):
    """one strength: references plus the pass curves of each anchor weight"""
    block = {"degraded": point(configs["degraded"][0]), "n": configs["degraded"][0][3]}
    if "undo-no-model" in configs:
        block["undo"] = point(configs["undo-no-model"][1])
    block["curves"] = {}
    for config, steps in configs.items():
        mu = mu_of(config)
        if mu is not None:
            block["curves"][f"{mu:g}"] = [point(steps[k], k) for k in sorted(steps)]
    return block


def best(curve):
    return min(curve, key=lambda p: p["lpips"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dev_dir", default=osp.join(osp.dirname(HERE), "workspace"))
    p.add_argument("--job_dir", default="sphere2-large-flowers-512px")
    p.add_argument("--html", default=osp.join(HERE, "index.html"))
    args = p.parse_args()
    root = osp.join(args.dev_dir, "inverse", args.job_dir)

    data = []
    for task in TASKS:
        analysis = read_summary(osp.join(root, "analysis", task["id"], "summary.csv"))
        shown = read_summary(osp.join(root, task["figure_run"], "summary.csv"))
        levels = sorted(analysis, reverse=task["id"] == "dehaze")  # mild -> heavy
        data.append({
            **{k: task[k] for k in ("id", "name", "what", "unit", "figure",
                                    "figure_level", "figure_passes")},
            "mu": task["mu"],
            "levels": [{"level": lv, **level_block(analysis[lv])} for lv in levels],
            "figure_metrics": level_block(shown[float(task["figure_level"])]),
        })

    # markdown tables: per strength, the input vs the best no-anchor and the
    # best anchored result (lowest lpips over passes)
    for t in data:
        mu = f"{t['mu']:g}"
        undo = "undo" in t["levels"][0]
        print(f"\n### {t['name']} ({t['what']}), 32 val images\n")
        head = "| Strength | Degraded |" + (" Undo, no model |" if undo else "")
        head += " No anchor: best (pass) | Anchored: best (pass) |"
        print(head)
        print("|" + " --- |" * (head.count("|") - 1))
        for lv in t["levels"]:
            loop, anch = best(lv["curves"]["0"]), best(lv["curves"][mu])
            fmt = lambda q: f"{q['psnr']:.1f} / {q['lpips']:.3f}"  # noqa: E731
            row = f"| {lv['level']:g}{t['unit']} | {fmt(lv['degraded'])} |"
            row += f" {fmt(lv['undo'])} |" if undo else ""
            row += f" {fmt(loop)} ({loop['step']}) | {fmt(anch)} ({anch['step']}) |"
            print(row)

    # the one-step latent fix, next to the best restoration loop at the same
    # (heaviest) strength
    with open(osp.join(root, "latent", "direct_summary.json")) as f:
        direct = json.load(f)
    fmt = lambda q: f"{q['psnr']:.1f} / {q['lpips']:.3f}"  # noqa: E731
    print("\n### One-step latent fix, heaviest strength, 32 val images\n")
    print("| Damage | Damaged | Encode, decode | Encode, subtract, decode | Best restoration loop |")
    print("| --- | --- | --- | --- | --- |")
    for t in data:
        r, heavy = direct[t["id"]], t["levels"][-1]
        assert r["level"] == heavy["level"], (t["id"], r["level"], heavy["level"])
        helps = all(best(lv["curves"][f"{t['mu']:g}"])["lpips"] < best(lv["curves"]["0"])["lpips"]
                    for lv in t["levels"])
        loop = best(heavy["curves"][f"{t['mu']:g}" if helps else "0"])
        r["best_loop"] = {**loop, "anchored": helps}
        print(f"| {t['name']} {heavy['level']:g}{t['unit']} | {fmt(r['damaged'])} | "
              f"{fmt(r['decode_without_subtraction'])} | {fmt(r['decode_with_subtraction'])} "
              f"(alpha {r['alpha']:g}) | {fmt(loop)} |")

    # write the data blocks of the website
    with open(args.html) as f:
        html = f.read()
    for block_id, payload in [("results", data), ("direct", direct)]:
        block = json.dumps(payload, separators=(",", ":"))
        html, n = re.subn(
            rf'(<script id="{block_id}" type="application/json">).*?(</script>)',
            lambda m: m.group(1) + block + m.group(2), html, flags=re.S,
        )
        if n != 1:
            raise SystemExit(f'no "{block_id}" block found in {args.html}')
    with open(args.html, "w") as f:
        f.write(html)
    print(f"\nwrote the data blocks of {args.html}")


if __name__ == "__main__":
    main()
