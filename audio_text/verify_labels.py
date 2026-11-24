"""Check that audio matches its AudioSet-strong label using CLAP.
For each clip: rank of the labelled class among all 456 AudioSet label names (1 = CLAP's top guess),
plus CLAP's top-3 guesses. Compares the clips we trained on with all other clips of the same label.
Usage: python verify_labels.py --meta data/meta.json --clap data/clap_all"""
import argparse, glob, json
import numpy as np, pandas as pd, torch

D = "/fs/gamma-projects/audio/audio_datasets/audioset_strong"


def load_clap(d):
    parts = [torch.load(f) for f in sorted(glob.glob(f"{d}/audio_*.pt"))]
    ids = sum([p["ids"] for p in parts], []); dirs = sum([p["dir"] for p in parts], [])
    emb = torch.cat([p["emb"] for p in parts]); lab = torch.load(f"{d}/labels.pt")
    return {(dr, i): k for k, (dr, i) in enumerate(zip(dirs, ids))}, emb, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--clap", default="data/clap_all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    idx, emb, lab = load_clap(a.clap)
    mids = lab["mids"]; names = lab["names"]
    sim = emb @ lab["emb"].T  # [N, 456]
    rank = (sim.argsort(1, descending=True).argsort(1) + 1)  # rank of each label per clip
    meta = json.load(open(a.meta))
    mid_of = {c: None for c in meta["classes"]}
    import prep_data_classes as P  # noqa  (class -> mid table)
    rows = []
    for m in meta["clips"]:
        mid = P.MID[m["cls"]]
        k = idx.get((m.get("dir", "audio_clips_train"), m["id"]))
        j = mids.index(mid)
        top = sim[k].topk(3).indices.tolist()
        rows.append(dict(id=m["id"], cls=m["cls"], split=m["split"], rank=int(rank[k, j]),
                         top3=" | ".join(names[t] for t in top),
                         others=", ".join(names[mids.index(l)] for l in m["labels"] if l != mid and l in mids)))
    r = pd.DataFrame(rows)
    # reference: every downloaded train clip that carries the label for >= 8.5 s
    df = pd.read_csv(f"{D}/audioset_train_strong.tsv", sep="\t")
    df["dur"] = df.end_time_seconds - df.start_time_seconds
    cov = df.groupby(["segment_id", "label"]).dur.sum().reset_index()
    ref = {}
    for c in r.cls.unique():
        mid = P.MID[c]; j = mids.index(mid)
        ks = [idx[("audio_clips_train", s)] for s in cov[(cov.label == mid) & (cov.dur >= 8.5)].segment_id
              if ("audio_clips_train", s) in idx]
        ref[c] = rank[ks, j].float()
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(r.to_string())
    print("\nper class: median rank of the true label among 456 (ours vs all clips with that label), "
          "share with rank<=10")
    for c, g in r.groupby("cls"):
        print(f"{c:14s} ours median {g['rank'].median():5.1f}  top10 {np.mean(g['rank'] <= 10):.0%}  |  "
              f"all-clips median {ref[c].median():5.1f}  top10 {(ref[c] <= 10).float().mean():.0%} (n={len(ref[c])})")
    if a.out:
        r.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
