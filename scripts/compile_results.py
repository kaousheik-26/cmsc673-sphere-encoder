import argparse
import json
import math
import os
import os.path as osp
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from halfsphere.records import read_rows

C = {"enc": "#2F6DB5", "dec": "#D9822B", "sph": "#7A4FB0", "loss": "#C0392B", "ok": "#2E8B57", "cmp": "#B8672A", "mute": "#6B6B6B", "q4": "#A23B72"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 300})

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
M = json.load(open(args.manifest))
os.makedirs(args.out, exist_ok=True)
H = {}


def mean_sem(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    m = st.mean(xs)
    return m, (st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


def agg(run_dir, method=None, images=None):
    by = defaultdict(lambda: defaultdict(list))
    cfg = {}
    for r in read_rows(run_dir):
        if method and r["method"] != method:
            continue
        if images is not None and r["image_id"] not in images:
            continue
        if r.get("draw", 0) != 0:
            continue
        for k, v in r["metrics"].items():
            by[(r["method"], r["config_hash"])][k].append(v)
        cfg[(r["method"], r["config_hash"])] = r["config"]
    out = {}
    for key, d in by.items():
        out[key] = {k: mean_sem(v) for k, v in d.items()}
        out[key]["n"] = len(next(iter(d.values())))
        out[key]["config"] = cfg[key]
    return out


def summary(run_dir):
    p = osp.join(run_dir, "summary.json")
    return json.load(open(p)) if osp.isfile(p) else {}


def save(fig, name):
    fig.savefig(osp.join(args.out, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def repaint_coarse():
    d = M["repaint_coarse"]
    a, s = agg(d), summary(d)
    pts = []
    for (meth, ch), m in a.items():
        k = s.get(ch, {}).get("kid", [None])[0] if meth == "repaint" else s.get(meth, {}).get("kid", [None])[0]
        pts.append((meth, ch, m["lpips_B"][0], k, m["seam"][0], m["config"]))
    fig, ax = plt.subplots(figsize=(2.3, 1.7))
    Kc = {4: C["enc"], 8: C["ok"], 12: C["dec"], 20: C["loss"]}
    for meth, ch, lp, k, _, cf in pts:
        if meth == "repaint":
            ax.scatter(lp, k * 1e3, s=6, color=Kc[cf["K"]], alpha=0.75, lw=0, marker="o" if cf["schedule"] != "constant" else "^")
    for meth, ch, lp, k, _, cf in pts:
        if meth in ("mirror", "oracle"):
            ax.scatter(lp, max(k * 1e3, 0.05), s=28, color=C["mute"] if meth == "mirror" else C["sph"], marker="*", zorder=5)
            lab = meth if k > 0 else f"{meth} (KID$\\leq$0, drawn at 0.05)"
            ax.annotate(lab, (lp, max(k * 1e3, 0.05)), xytext=(3, 2), textcoords="offset points", fontsize=6, color=C["mute"] if meth == "mirror" else C["sph"])
    for K, col in Kc.items():
        ax.scatter([], [], s=8, color=col, label=f"K={K}")
    ax.scatter([], [], s=8, color=C["mute"], marker="^", label="constant $\\alpha$")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", handletextpad=0.1, borderpad=0.1, labelspacing=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("LPIPS on missing half (lower = closer to truth)")
    ax.set_ylabel(r"KID $\times 10^3$ (lower = more realistic)")
    save(fig, "fig_repaint_pareto.png")
    repaint = [p for p in pts if p[0] == "repaint"]
    best = min(repaint, key=lambda p: p[2])
    ref = {p[0]: p for p in pts if p[0] in ("mirror", "oracle")}
    H["repaint_coarse"] = {
        "n_configs": len(repaint), "n_images": a[("mirror", "mirror")]["n"],
        "lpips_B_range": [min(p[2] for p in repaint), max(p[2] for p in repaint)], "kid_range": [min(p[3] for p in repaint), max(p[3] for p in repaint)],
        "seam_range": [min(p[4] for p in repaint), max(p[4] for p in repaint)],
        "best_lpips": {"hash": best[1], "lpips_B": best[2], "kid": best[3], "cfg": {k: best[5][k] for k in ("K", "alpha_start", "alpha_end", "schedule")}},
        "mirror": {"lpips_B": ref["mirror"][2], "kid": ref["mirror"][3], "seam": ref["mirror"][4]},
        "oracle": {"lpips_B": ref["oracle"][2], "kid": ref["oracle"][3], "seam": ref["oracle"][4], "psnr_A_prepin": a[("oracle", "oracle")]["psnr_A_prepin"][0]},
        "lpips_B_sem_typical": a[("repaint", best[1])]["lpips_B"][1],
    }
    byK = defaultdict(list)
    for p in repaint:
        byK[p[5]["K"]].append(p)
    H["repaint_coarse"]["by_K"] = {K: {"lpips_B": st.mean(p[2] for p in v), "kid": st.mean(p[3] for p in v), "seam": st.mean(p[4] for p in v)} for K, v in sorted(byK.items())}
    moves = [m["final_lat_move_deg"][0] for (meth, ch), m in a.items() if meth == "repaint"]
    H["repaint_coarse"]["final_move_deg_range"] = [min(moves), max(moves)]


def repaint_table():
    rows = []
    for name in ("repaint_refine_A", "repaint_refine_B", "repaint_null", "repaint_tb", "repaint_second", "repaint_tok"):
        if name not in M:
            continue
        a, s = agg(M[name]), summary(M[name])
        for (meth, ch), m in a.items():
            if meth in ("mirror", "oracle") and name not in ("repaint_tb", "repaint_second", "repaint_null"):
                continue
            kid_ = s.get(ch if meth.startswith("repaint") else meth, {}).get("kid", [None])[0]
            rows.append({"run": name, "method": meth, "hash": ch, "cfg": m["config"], "n": m["n"],
                         **{k: m[k][0] for k in ("lpips_B", "psnr_B", "seam", "lat_angle_true") if k in m},
                         "lpips_B_sem": m["lpips_B"][1], "kid": kid_})
    H["repaint_table"] = rows


def repaint_samples():
    if "repaint_samples" not in M:
        return
    s = summary(M["repaint_samples"])
    a = agg(M["repaint_samples"])
    out = {}
    for (meth, ch), m in a.items():
        if meth == "repaint":
            div = s[ch].get("diversity_lpips_B")
            out[ch] = {"cfg": {k: m["config"][k] for k in ("K", "alpha_start", "alpha_end", "schedule")}, "lpips_B": m["lpips_B"][0],
                       "diversity_lpips_B": st.mean(div) if div else None, "n_draws": 8}
    H["repaint_samples"] = out


def repaint_traj():
    if "repaint_traj" not in M:
        return
    import torch

    d = M["repaint_traj"]
    out = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("traj_") and f.endswith(".pt"):
            steps = torch.load(osp.join(d, f))
            ch = f.split("_")[1]
            out[ch] = {"lat_move_deg": [float(st_["lat_move_deg"].mean()) for st_ in steps], "seam": [float(st_["seam"].mean()) for st_ in steps],
                       "resid_A": [float(st_["resid_A"].mean()) for st_ in steps], "alpha": [st_["alpha"] for st_ in steps]}
    H["repaint_traj"] = out


def h1():
    r = json.load(open(osp.join(M["ridgemap_h1"], "h1_cca.json")))
    k = len(r["real"])
    xs = list(range(1, k + 1))
    fig, ax = plt.subplots(figsize=(2.3, 1.7))
    lo = [m - 3 * sd for m, sd in zip(r["null_mean"], r["null_sd"])]
    hi = [m + 3 * sd for m, sd in zip(r["null_mean"], r["null_sd"])]
    ax.fill_between(xs, lo, hi, color=C["cmp"], alpha=0.35, lw=0, label="shuffled pairs ($\\pm$3 sd)")
    ax.plot(xs, r["null_mean"], color=C["cmp"], lw=0.7)
    ax.plot(xs, r["real"], color=C["sph"], lw=1.2, label="real left/right pairs")
    ax.axhline(r["wachter_edge"], color=C["mute"], lw=0.6, ls="--")
    ax.text(k, r["wachter_edge"] + 0.02, "Wachter edge", fontsize=5.5, color=C["mute"], ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("canonical component (of 512)")
    ax.set_ylabel("canonical correlation")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    save(fig, "fig_h1_cca.png")
    z = r["z_top20"]
    cross = next((i for i, (a_, b_) in enumerate(zip(r["real"], r["null_mean"])) if a_ < b_ + 3 * max(r["null_sd"][min(i, len(r["null_sd"]) - 1)], 1e-9)), None)
    H["h1"] = {"n": r["n"], "k_pca": r["k_pca"], "top_cc": r["real"][:5], "null_top": r["null_mean"][0], "null_sd": r["null_sd"][0],
               "wachter": r["wachter_edge"], "min_z_top20": min(z), "components_above_null": cross}


def h2():
    R = read_rows(M["ridgemap_h2"])
    out = {}
    for e in ("0.01", "0.001"):
        rk = [x["where"]["true"][e]["rank_eps"] for x in R]
        fd = [x["where"]["true"][e]["fiber_dim"] for x in R]
        out[e] = {"rank_mean": st.mean(rk), "rank_min": min(rk), "rank_max": max(rk), "fiber_mean": st.mean(fd), "fiber_min": min(fd),
                  "probe_sd_mean": st.mean(x["where"]["true"][e]["rank_eps_sd"] for x in R)}
    out["n"] = len(R)
    out["wall_s_mean"] = st.mean(x["wall_s"] for x in R)
    fig, ax = plt.subplots(figsize=(2.3, 1.7))
    for j, (e, col) in enumerate((("0.01", C["enc"]), ("0.001", C["dec"]))):
        rk = sorted(x["where"]["true"][e]["rank_eps"] for x in R)
        ax.bar([i + 0.4 * j for i in range(len(rk))], rk, width=0.4, color=col, label=f"$\\varepsilon$={e}")
    ax.set_xticks([])
    ax.set_xlabel(f"{len(R)} images (sorted)")
    ax.set_ylabel(r"numerical rank $r_\varepsilon$ of $J_A$")
    ax.legend(fontsize=5.5, frameon=False)
    save(fig, "fig_h2_rank.png")
    if "ridgemap_check" in M:
        c = json.load(open(osp.join(M["ridgemap_check"], "check.json")))
        out["check"] = {"max_rel_err_vs_arpack": c["max_rel_err"], "slq_k80": c["slq"]["80"], "slq_k200": c["slq"]["200"]}
    H["h2"] = out


def h3():
    R = read_rows(M["ridgemap_h3"])
    by = defaultdict(list)
    step = defaultdict(list)
    for x in R:
        for e in x["h3"]:
            by[e["alpha"]].append(e["rho"])
            step[e["alpha"]].append(e["last_step_rel"])
    al = sorted(by)
    med = [st.median(by[a]) for a in al]
    q1 = [sorted(by[a])[len(by[a]) // 4] for a in al]
    q3 = [sorted(by[a])[(3 * len(by[a])) // 4] for a in al]
    fig, ax = plt.subplots(figsize=(2.3, 1.7))
    ax.fill_between(al, q1, q3, color=C["sph"], alpha=0.2, lw=0)
    ax.plot(al, med, "-o", color=C["sph"], ms=2.5, lw=1)
    ax.axhline(1.0, color=C["loss"], lw=0.7, ls="--")
    ax.text(al[-1], 1.0, " stable below", fontsize=5.5, color=C["loss"], va="top", ha="right")
    ax.set_xlabel(r"constant angle $\alpha$ (deg)")
    ax.set_ylabel(r"contraction $\rho$ (median, IQR)")
    save(fig, "fig_h3_rho.png")
    frac_below = {a: sum(r < 1 for r in by[a]) / len(by[a]) for a in al}
    H["h3"] = {"n": len(R), "alpha": al, "rho_median": med, "rho_q1": q1, "rho_q3": q3, "frac_below_1": frac_below,
               "last_step_median": {a: st.median(step[a]) for a in al}, "any_alpha_median_below_1": any(m < 1 for m in med)}


def modes():
    if "ridgemap_modes" not in M:
        return
    R = read_rows(M["ridgemap_modes"])
    out = {}
    for where in ("repaint", "true"):
        lk = [l for x in R for l in x["modes"][where]["leak"]]
        rs = [r for x in R for r in x["modes"][where]["ritz_resid"]]
        oc = [o for x in R for o in x["modes"][where]["obs_change_mean_abs"]]
        mc = [o for x in R for o in x["modes"][where]["miss_change_mean_abs"]]
        out[where] = {"leak_median": st.median(lk), "leak_max": max(lk), "resid_median": st.median(rs), "frac_leak_below_0.2": sum(l < 0.2 for l in lk) / len(lk),
                      "obs_change": st.mean(oc), "miss_change": st.mean(mc), "eig_top_mean": st.mean(x["modes"][where]["eig"][0] for x in R)}
    out["n"] = len(R)
    H["modes"] = out


def lm():
    if "ridgemap_lm" not in M:
        return
    R = read_rows(M["ridgemap_lm"])
    out = {}

    def col(where, k):
        return [x["lm"][where][k] for x in R]

    for where in ("repaint", "true"):
        out[where] = {k: st.mean(col(where, k)) for k in ("resid_A_before", "resid_A_after", "psnr_A_before", "psnr_A_after", "lpips_B_before", "lpips_B_after", "moved_deg")}
        out[where]["frac_improved"] = sum(a < b for a, b in zip(col(where, "resid_A_after"), col(where, "resid_A_before"))) / len(R)
    out["n"] = len(R)
    H["lm"] = out


def steer():
    rows = []
    imgs32 = None
    for name in ("steer_h5", "steer_variants", "steer_matched", "steer_h4"):
        if name not in M:
            continue
        a, s = agg(M[name]), summary(M[name])
        for (meth, ch), m in a.items():
            if imgs32 is None:
                imgs32 = {r["image_id"] for r in read_rows(M[name])}
            info = s.get(ch, {})
            rows.append({"run": name, "hash": ch, "cfg": m["config"], "n": m["n"],
                         **{k: m[k][0] for k in m if k not in ("n", "config")}, "lpips_B_sem": m["lpips_B"][1],
                         "kid": info.get("kid", [None])[0], "dec_passes": info.get("per_image_dec_fwd"), "dec_bwd": info.get("per_image_dec_bwd"),
                         "net_passes": sum(info.get("passes", {}).values()) / m["n"] if info.get("passes") else None})
    H["steer"] = rows
    if imgs32 and "repaint_refine_A" in M:
        for name in ("repaint_refine_A", "repaint_refine_B"):
            a = agg(M[name], images=imgs32)
            for (meth, ch), m in a.items():
                if meth == "repaint" and m["config"].get("share_noise", True) and m["config"].get("fill") == "mirror" and not m["config"].get("dec_alpha_offset") and not m["config"].get("lam_end") and m["config"].get("final_clean", True):
                    cf = m["config"]
                    H.setdefault("repaint_on_steer_images", {})[name] = {"lpips_B": m["lpips_B"][0], "psnr_A_prepin": m["psnr_A_prepin"][0], "seam": m["seam"][0], "n": m["n"],
                                                                 "net_passes": 2 * (cf["K"] + int(cf.get("final_clean", True)))}
                if meth in ("mirror", "oracle"):
                    H.setdefault("repaint_on_steer_images", {})[meth] = {"lpips_B": m["lpips_B"][0], "seam": m["seam"][0], "n": m["n"]}
    if rows:
        fig, ax = plt.subplots(figsize=(2.3, 1.7))
        h5 = sorted([r for r in rows if r["run"] == "steer_h5"], key=lambda r: r["cfg"]["lam_cyc"])
        xs = list(range(len(h5)))
        ax.bar(xs, [r["lpips_B"] for r in h5], width=0.6, color=C["sph"], yerr=[r["lpips_B_sem"] for r in h5], error_kw={"lw": 0.6})
        ax.set_xticks(xs, [f"{r['cfg']['lam_cyc']:g}" for r in h5])
        ax.set_xlabel(r"$\lambda_{\rm cyc}$ (Steer, 300 steps)")
        ax.set_ylabel("LPIPS on missing half")
        repaint_r = H.get("repaint_on_steer_images", {})
        for nm, col, lab in (("repaint_refine_A", C["cmp"], "Repaint A"), ("mirror", C["mute"], "mirror")):
            if nm in repaint_r:
                ax.axhline(repaint_r[nm]["lpips_B"], color=col, lw=0.7, ls="--")
                ax.text(len(h5) - 0.5, repaint_r[nm]["lpips_B"], lab, fontsize=5.5, color=col, va="bottom", ha="right")
        lo = min([r["lpips_B"] for r in h5] + [v["lpips_B"] for v in repaint_r.values() if v["lpips_B"] > 0.3])
        ax.set_ylim(lo - 0.08, None)
        save(fig, "fig_h5.png")


def method_figs():
    if "repaint_traj" in H:
        fig, axs = plt.subplots(1, 2, figsize=(3.6, 1.45))
        names = {"fd88111b30": ("A: K=4, 50$^\\circ$", C["enc"]), "1bf109b6b8": ("B: K=12, 80$\\to$20$^\\circ$", C["dec"])}
        for ch, tr in H["repaint_traj"].items():
            lab, col = names.get(ch, (ch, C["mute"]))
            ks = list(range(1, len(tr["lat_move_deg"])))
            axs[0].plot(ks, tr["lat_move_deg"][1:], "-o", ms=2, lw=1, color=col, label=lab)
            axs[1].plot(range(len(tr["resid_A"])), tr["resid_A"], "-o", ms=2, lw=1, color=col)
        axs[0].set_xlabel("step $k$")
        axs[0].set_ylabel("latent move (deg)")
        axs[0].set_ylim(0, None)
        axs[0].legend(fontsize=5.5, frameon=False)
        axs[1].set_xlabel("step $k$")
        axs[1].set_ylabel("observed-half RMS error")
        axs[1].set_ylim(0, None)
        fig.tight_layout(w_pad=0.8)
        save(fig, "fig_repaint_traj.png")
    if "steer_h5" in M:
        curves = defaultdict(list)
        for r in read_rows(M["steer_h5"]):
            curves[r["config"]["lam_cyc"]].append(r["loss_curve_best"])
        fig, ax = plt.subplots(figsize=(1.9, 1.45))
        cols = [C["enc"], C["ok"], C["dec"], C["loss"]]
        for (lam, cs), col in zip(sorted(curves.items()), cols):
            n = min(len(c) for c in cs)
            mean = [st.mean(c[t] for c in cs) for t in range(n)]
            ax.plot([10 * t for t in range(n)], mean, color=col, lw=1, label=f"$\\lambda_{{\\rm cyc}}$={lam:g}")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(r"objective $\Phi$ (mean)")
        ax.legend(fontsize=5.5, frameon=False)
        save(fig, "fig_steer_curves.png")
    if H.get("steer") and H.get("repaint_on_steer_images"):
        fig, axs = plt.subplots(1, 2, figsize=(3.6, 1.45))
        pts = [r for r in H["steer"] if r["run"] in ("steer_matched", "steer_h5") and r["cfg"].get("lam_cyc") == 0.0]
        pts.sort(key=lambda r: r["net_passes"])
        xs = [r["net_passes"] for r in pts]
        axs[0].plot(xs, [r["seam"] for r in pts], "-o", ms=2.5, lw=1, color=C["sph"], label="Steer ($\\lambda_{\\rm cyc}$=0)")
        axs[1].plot(xs, [r["psnr_A_prepin"] for r in pts], "-o", ms=2.5, lw=1, color=C["sph"])
        for nm, col, lab in (("repaint_refine_A", C["enc"], "Repaint A"), ("repaint_refine_B", C["dec"], "Repaint B")):
            v = H["repaint_on_steer_images"].get(nm)
            if v:
                axs[0].scatter([v["net_passes"]], [v["seam"]], color=col, s=12, zorder=5, label=lab)
                axs[1].scatter([v["net_passes"]], [v["psnr_A_prepin"]], color=col, s=12, zorder=5)
        for ax in axs:
            ax.set_xscale("log")
            ax.set_xlabel("network passes per image")
        axs[0].set_ylabel("seam score")
        axs[1].set_ylabel("observed-half PSNR (dB)")
        axs[0].legend(fontsize=5.5, frameon=False)
        fig.tight_layout(w_pad=0.8)
        save(fig, "fig_steer_budget.png")
    if "lm" in H:
        fig, axs = plt.subplots(1, 2, figsize=(3.0, 1.45))
        groups = [("from Repaint A", "repaint"), ("from true $v(x)$", "true")]
        for j, (key, lab) in enumerate((("psnr_A", "observed-half PSNR (dB)"), ("lpips_B", "missing-half LPIPS"))):
            for g, (gl, w) in enumerate(groups):
                b, a = H["lm"][w][key + "_before"], H["lm"][w][key + "_after"]
                axs[j].bar([3 * g, 3 * g + 1], [b, a], color=[C["mute"], C["sph"]], width=0.9)
            axs[j].set_xticks([0.5, 3.5], [gl for gl, _ in groups], fontsize=6)
            axs[j].set_ylabel(lab)
        from matplotlib.patches import Patch

        axs[0].legend(handles=[Patch(color=C["mute"], label="before"), Patch(color=C["sph"], label="after 10 LM steps")],
                      fontsize=5.5, frameon=False, loc="lower right")
        fig.tight_layout(w_pad=0.8)
        save(fig, "fig_lm.png")


for fn in (repaint_coarse, repaint_table, repaint_samples, repaint_traj, h1, h2, h3, modes, lm, steer, method_figs):
    try:
        fn()
        print(f"ok   {fn.__name__}")
    except KeyError as e:
        print(f"skip {fn.__name__}: missing {e}")
json.dump(H, open(osp.join(args.out, "headline.json"), "w"), indent=2, default=float)
print("wrote", args.out)
