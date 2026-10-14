import argparse
import csv
import json
import math
import os.path as osp
from collections import defaultdict

from halfsphere.records import read_rows

parser = argparse.ArgumentParser(description="summarise a Repaint run")
parser.add_argument("run_dir", type=str)
parser.add_argument("--top", type=int, default=15)
args = parser.parse_args()

KEYS = ["lpips_B", "psnr_B", "psnr_A_prepin", "seam", "lat_angle_true", "lpips_full", "final_lat_move_deg", "final_resid_A"]


def mean_sem(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def pareto(rows, a="lpips_B", b="kid"):
    front = []
    for r in rows:
        if r.get(b) is None:
            continue
        dominated = any(
            o is not r and o.get(b) is not None and o[a] <= r[a] and o[b] <= r[b] and (o[a] < r[a] or o[b] < r[b]) for o in rows
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[a])


def main():
    rows = read_rows(args.run_dir)
    summary = json.load(open(osp.join(args.run_dir, "summary.json")))
    by = defaultdict(list)
    cfg_of = {}
    for r in rows:
        key = (r["method"], r["config_hash"], r.get("draw", 0))
        by[key].append(r["metrics"])
        cfg_of[r["config_hash"]] = r["config"]
    table = []
    for (method, ch, draw), ms in by.items():
        if draw != 0:
            continue
        out = {"method": method, "config_hash": ch, "n": len(ms)}
        for k in KEYS:
            xs = [m[k] for m in ms if k in m]
            if xs:
                out[k], out[k + "_sem"] = mean_sem(xs)
        s = summary.get(ch, {})
        out["kid"] = s.get("kid", [None])[0] if isinstance(s.get("kid"), list) else None
        out["kid_sd"] = s.get("kid", [None, None])[1] if isinstance(s.get("kid"), list) else None
        out["per_image_dec_fwd"] = s.get("per_image_dec_fwd")
        if "diversity_lpips_B" in s:
            out["diversity_lpips_B"] = sum(s["diversity_lpips_B"]) / len(s["diversity_lpips_B"])
        out.update({f"cfg_{k}": v for k, v in cfg_of[ch].items() if k in ("K", "alpha_start", "alpha_end", "schedule", "share_noise", "fill", "dec_alpha_offset", "lam_end", "final_clean", "alphaA", "alphaB")})
        table.append(out)
    table.sort(key=lambda r: r.get("lpips_B", 9))
    fields = sorted({k for r in table for k in r}, key=lambda k: (not k.startswith("method"), not k.startswith("cfg_"), k))
    with open(osp.join(args.run_dir, "table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    repaint_rows = [r for r in table if r["method"] == "repaint"]
    front = pareto(repaint_rows)
    json.dump(front, open(osp.join(args.run_dir, "pareto.json"), "w"), indent=2)

    def fmt(r):
        c = f"K={r.get('cfg_K', '-')} a={r.get('cfg_alpha_start', '-')}->{r.get('cfg_alpha_end', '-')} {r.get('cfg_schedule', '')}"
        nondefault = {"share_noise": False, "fill": "blur", "final_clean": False}
        extra = [k for k, v in nondefault.items() if r.get("cfg_" + k) == v]
        if r.get("cfg_dec_alpha_offset"):
            extra.append(f"dec+{r['cfg_dec_alpha_offset']}")
        if r.get("cfg_lam_end"):
            extra.append(f"lam{r['cfg_lam_end']}")
        if r.get("cfg_alphaA") is not None:
            c = f"tok K={r['cfg_K']} A{r['cfg_alphaA']} B{r['cfg_alphaB']}"
        c = (c + " " + ",".join(extra)).strip()
        kid_s = f"{r['kid'] * 1e3:6.2f}" if r.get("kid") is not None else "   n/a"
        return (f"{r['method']:7s} {r['config_hash']:10s} {c:44s} LPIPS_B {r['lpips_B']:.4f}+-{r['lpips_B_sem']:.4f}  "
                f"PSNR_B {r['psnr_B']:5.2f}  PSNR_A(pre) {r['psnr_A_prepin']:5.2f}  seam {r['seam']:.3f}  "
                f"angle {r['lat_angle_true']:5.2f}  KIDx1e3 {kid_s}")

    print("references:")
    for r in table:
        if r["method"] != "repaint":
            print("  " + fmt(r))
    print(f"best {args.top} by LPIPS_B:")
    for r in repaint_rows[: args.top]:
        print("  " + fmt(r))
    print("pareto front (LPIPS_B vs KID):")
    for r in front:
        print("  " + fmt(r))


if __name__ == "__main__":
    main()
