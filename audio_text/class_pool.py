"""Per-label pool of clean clips: target covers >= MINCOV s, no speech/music/singing label covering > 3 s
(unless that is the target), and CLAP ranks the target label within top RANK of all 456 labels."""
import glob, sys, pandas as pd, torch
D = "/fs/gamma-projects/audio/audio_datasets/audioset_strong"
MINCOV, RANK = float(sys.argv[1]), int(sys.argv[2])
parts = [torch.load(f) for f in sorted(glob.glob("data/clap_all/audio_*.pt"))]
ids = sum([p["ids"] for p in parts], []); dirs = sum([p["dir"] for p in parts], [])
emb = torch.cat([p["emb"] for p in parts]); lab = torch.load("data/clap_all/labels.pt")
rank = (emb @ lab["emb"].T).argsort(1, descending=True).argsort(1) + 1
key = {(d, i): k for k, (d, i) in enumerate(zip(dirs, ids))}
names = dict(zip(lab["mids"], lab["names"]))
VOX = [m for m, n in names.items() if any(w in n.lower() for w in ["speech", "speaking", "singing", "music", "narration", "conversation", "rapping", "choir", "song", "babble", "television", "radio"])]
rows = []
for d, tsv in [("audio_clips_train", "audioset_train_strong.tsv"), ("audio_clips", "audioset_eval_strong.tsv")]:
    df = pd.read_csv(f"{D}/{tsv}", sep="\t")
    df["dur"] = df.end_time_seconds - df.start_time_seconds
    cov = df.groupby(["segment_id", "label"]).dur.sum().clip(upper=10).reset_index()
    vox = cov[cov.label.isin(VOX)].groupby("segment_id").dur.max()
    c = cov[cov.dur >= MINCOV].copy()
    c["k"] = [key.get((d, s), -1) for s in c.segment_id]; c = c[c.k >= 0]
    c["vox"] = c.segment_id.map(vox).fillna(0)
    c["r"] = [int(rank[k, lab["mids"].index(l)]) if l in lab["mids"] else 999 for k, l in zip(c.k, c.label)]
    c["dir"] = d; rows.append(c)
c = pd.concat(rows)
ok = c[(c.r <= RANK) & ((c.vox <= 3) | c.label.isin(VOX))]
t = pd.DataFrame({"cov": c.groupby("label").size(), "clean": ok.groupby("label").size()}).fillna(0).astype(int)
t["name"] = t.index.map(names); t = t.sort_values("clean", ascending=False)
pd.set_option("display.width", 200); print(t[t.clean >= 60].to_string())
ok.to_csv(f"data/pool_cov{MINCOV}_r{RANK}.csv", index=False)
