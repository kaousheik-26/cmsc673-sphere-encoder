"""25 classes x up to 1000 train clips, from AudioSet-strong + AudioCaps, label-checked.

Test set = the same 250 clips (10 per class) as data25, so results compare with earlier runs.
Train candidates per class, in order of preference:
  1. AudioSet-strong clips passing the data25 rules (class covers >= 5 s, no speech/singing/music > 3 s unless
     that is the class, no other selected class >= 3 s, CLAP rank of the class <= 10, then <= 20),
  2. AudioCaps clips whose human captions match exactly one of the 25 classes (keyword rules below) and where
     CLAP ranks that class within its top 10 of all 456 AudioSet labels.
A YouTube video is used at most once overall and never if it is in the test set.
Writes <out>/meta.json, mels.pt [N,80,861] fp16, text_emb.pt, wav/ (test clips only, symlinks to data25).
Needs data/clap_all (AudioSet-strong CLAP), data/clap_audiocaps (AudioCaps CLAP), data/pool_cov5.0_r20.csv."""
import argparse, glob, json, os, os.path as osp, random, re, sys
from multiprocessing import Pool

import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import resample_poly

sys.path.insert(0, osp.dirname(__file__))
from audiomel import SR, mel_spectrogram, LO  # noqa
from prep_data25 import CLASSES  # noqa

D = "/fs/gamma-projects/audio/audio_datasets/audioset_strong"
AC = "/fs/gamma-projects/audio/audio_datasets/audiocaps/2.0"
NFR = 861  # mel frames of 10 s at 22.05 kHz / hop 256
# caption keyword rules (lower-case, whole caption set of a clip joined)
KW = {
    "male_speech": r"\b(man|men|male|guy|boy)s? (is )?(speak|speaks|speaking|talk|talks|talking|narrat)",
    "female_speech": r"\b(woman|women|female|lady|girl)s? (is )?(speak|speaks|speaking|talk|talks|talking|narrat)",
    "child_speech": r"\b(child|children|kid|kids|toddler)s? (is )?(speak|speaks|speaking|talk|talks|talking)",
    "male_singing": r"\b(man|male|men) (is )?(sing|sings|singing)",
    "female_singing": r"\b(woman|female|women|girl) (is )?(sing|sings|singing)",
    "choir": r"\bchoir|\bchorus",
    "rapping": r"\brap\b|\brapping|\braps\b",
    "crowd_babble": r"\bcrowd|\bbabble|\bchatter|people (talk|chat|convers)",
    "bird_song": r"\bbirds? (chirp|sing|tweet|call|whistl)|\bbirdsong|\bbird song",
    "train": r"\btrain\b|\btrains\b|\blocomotive",
    "race_car": r"\brace ?car|\bracing",
    "motorcycle": r"\bmotorcycle|\bmotorbike|\bdirt bike",
    "truck": r"\btruck\b|\btrucks\b|\blorry",
    "subway": r"\bsubway|\bmetro\b|\bunderground train",
    "helicopter": r"\bhelicopter",
    "chainsaw": r"\bchain ?saw",
    "power_tool": r"\bdrill|\bpower tool|\belectric saw|\bcircular saw|\bsander|\bgrinder",
    "vacuum_cleaner": r"\bvacuum",
    "water_tap": r"\bfaucet|\btap water|\bwater tap|\bsink\b|water (runs|running|pours|pouring) (from|into)",
    "church_bell": r"\bchurch bell|\bbells? (ring|rings|ringing|toll|tolling|chime|chiming)",
    "civil_defense_siren": r"\b(air raid|civil defense|tornado|warning|emergency) siren|\bsiren (wails|wailing)",
    "fire_engine_siren": r"\bfire (truck|engine)",
    "rain": r"\brain\b|\braining|\brainfall|\brains\b",
    "waves": r"\bwaves\b|\bsurf\b|\bocean\b",
    "engine_knocking": r"\bengine (knock|knocks|knocking|clatter|clatters|sputter|sputters|tick)",
}
# pairs that are not a conflict (a fire-engine siren caption usually says "siren"; the class word wins)
SUB = {("fire_engine_siren", "truck"), ("fire_engine_siren", "civil_defense_siren"), ("subway", "train"),
       ("race_car", "motorcycle")}


def load_clap(d):
    parts = [torch.load(f) for f in sorted(glob.glob(f"{d}/audio_*.pt"))]
    ids = sum([p["ids"] for p in parts], []); dirs = sum([p["dir"] for p in parts], [])
    return {(dr, i): k for k, (dr, i) in enumerate(zip(dirs, ids))}, torch.cat([p["emb"] for p in parts])


def work(job):
    path = job
    torch.set_num_threads(1)
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    w = w.mean(1)
    if len(w) < 9.0 * sr or np.sqrt((w ** 2).mean()) < 1e-3:
        return None
    w = w[:10 * sr]
    if sr != SR:
        w = resample_poly(w, SR, sr).astype(np.float32)
    w = np.clip(w / max(1.0, np.abs(w).max()), -1, 1)
    m = mel_spectrogram(torch.from_numpy(w)[None])[0]
    nfr = min(NFR, m.shape[-1])
    if m.shape[-1] < NFR:  # pad short clips by repeating the last frames' floor
        m = torch.cat([m, torch.full((m.shape[0], NFR - m.shape[-1]), LO)], -1)
    return m[:, :NFR].half().numpy(), nfr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clap_ac", default="data/clap_audiocaps")
    a = ap.parse_args()
    os.makedirs(osp.join(a.out, "wav"), exist_ok=True)
    m25 = json.load(open("data25/meta.json")); mel25 = torch.load("data25/mels.pt")
    test = [c for c in m25["clips"] if c["split"] == "test"]
    yt = lambda sid: sid.rsplit("_", 1)[0]  # noqa
    used = {yt(c["id"]) for c in test}

    # ---- AudioSet-strong pool (same rules as data25) ----
    pool = pd.read_csv("data/pool_cov5.0_r20.csv")
    covs = []
    for d, tsv in [("audio_clips_train", "audioset_train_strong.tsv"), ("audio_clips", "audioset_eval_strong.tsv")]:
        df = pd.read_csv(f"{D}/{tsv}", sep="\t"); df["dur"] = df.end_time_seconds - df.start_time_seconds
        c = df.groupby(["segment_id", "label"]).dur.sum().clip(upper=10).reset_index(); c["dir"] = d; covs.append(c)
    cov = pd.concat(covs)
    sel_mids = {m: k for k, (ms, _, _) in CLASSES.items() for m in ms}
    big = cov[cov.label.isin(sel_mids) & (cov.dur >= 3)]
    classes_of = big.assign(c=big.label.map(sel_mids)).groupby(["dir", "segment_id"]).c.apply(set).to_dict()

    # ---- AudioCaps pool ----
    caps = pd.concat([pd.read_csv(f"{AC}/csvs/train.csv"), pd.read_csv(f"{AC}/csvs/test.csv")])
    caps = caps.dropna(subset=["youtube_id", "start_time", "caption"])
    caps["fid"] = caps.youtube_id + "_" + caps.start_time.astype(int).astype(str)
    capl = caps.groupby("fid").caption.apply(list).to_dict()
    lab = torch.load("data/clap_all/labels.pt"); mids = lab["mids"]
    ac_idx, ac_emb = load_clap(a.clap_ac)
    rank = ((ac_emb @ lab["emb"].T).argsort(1, descending=True).argsort(1) + 1)
    ac_c = {k: [] for k in CLASSES}
    nmatch = {k: 0 for k in CLASSES}
    for (d, fid), k in ac_idx.items():
        if fid not in capl:
            continue
        text = " ".join(capl[fid]).lower()
        hit = {c for c, p in KW.items() if re.search(p, text)}
        hit = {c for c in hit if not any((o, c) in SUB for o in hit)}
        if len(hit) != 1:
            continue
        c = hit.pop(); nmatch[c] += 1
        r = min(int(rank[k, mids.index(m)]) for m in CLASSES[c][0])
        if r <= 10:
            ac_c[c].append((fid, r))

    rng = random.Random(0)
    clips, jobs = [], []
    for c in test:
        clips.append(dict(c)); jobs.append(("test", c["id"]))
    report = {}
    for cname, (cm, tcaps, _) in CLASSES.items():
        p = pool[pool.label.isin(cm)].sort_values("r").drop_duplicates(["dir", "segment_id"])
        p = p[[classes_of.get((d, s), {cname}) <= {cname} for d, s in zip(p.dir, p.segment_id)]]
        cands = []
        for t in (p[p.r <= 10], p[p.r > 10]):
            t = [("strong", d, s) for d, s in zip(t.dir, t.segment_id)]; rng.shuffle(t); cands += t
        ac = [("audiocaps", "audiocaps", f) for f, _ in ac_c[cname]]; rng.shuffle(ac); cands += ac
        report[cname] = dict(strong_candidates=len(p), audiocaps_caption_match=nmatch[cname],
                             audiocaps_clap_ok=len(ac_c[cname]))
        jobs_c = []
        for src, d, sid in cands:
            if yt(sid) in used:
                continue
            used.add(yt(sid)); jobs_c.append((src, d, sid))
            if len(jobs_c) >= int(a.n_train * 1.08) + 5:
                break
        report[cname]["queued"] = len(jobs_c)
        for src, d, sid in jobs_c:
            path = f"{D}/{d}/{sid}.wav" if src == "strong" else f"{AC}/audios/audiocaps_raw_audio/{sid}.wav"
            clips.append(dict(id=sid, dir=d, src=src, cls=cname, split="train", captions=tcaps,
                              hcaps=capl.get(sid, []) if src == "audiocaps" else [], path=path))
            jobs.append(("train", path))
    print("queued", len(jobs), flush=True)

    # ---- mels (test clips reuse data25 mels) ----
    t_idx = {c["id"]: i for i, c in enumerate(m25["clips"])}
    tr_jobs = [j[1] for j in jobs if j[0] == "train"]
    with Pool(a.workers) as P:
        tr_mels = P.map(work, tr_jobs, chunksize=16)
    out_clips, mels = [], []
    per = {k: 0 for k in CLASSES}
    it = iter(tr_mels)
    for c in clips:
        if c["split"] == "test":
            c["nfr"] = NFR; out_clips.append(c); mels.append(mel25[t_idx[c["id"]]])
            dst = osp.join(a.out, "wav", f"{c['id']}.wav")
            if not osp.exists(dst):
                os.symlink(osp.abspath(f"data25/wav/{c['id']}.wav"), dst)
            continue
        r = next(it)
        if r is None or per[c["cls"]] >= a.n_train:
            continue
        per[c["cls"]] += 1; c.pop("path"); c["nfr"] = r[1]; out_clips.append(c); mels.append(torch.from_numpy(r[0]))
    for k in CLASSES:
        report[k]["train"] = per[k]
        report[k]["from_audiocaps"] = sum(1 for c in out_clips if c["cls"] == k and c["split"] == "train" and c["src"] == "audiocaps")
        print(f"{k:20s} {report[k]}", flush=True)
    mels = torch.stack(mels)
    print("mels", tuple(mels.shape), flush=True)
    torch.save(mels, osp.join(a.out, "mels.pt"))

    # ---- CLAP text embeddings: templates, unseen prompts, AudioCaps captions ----
    from transformers import ClapModel, ClapProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    texts = sorted({t for _, cs, _ in CLASSES.values() for t in cs} | {u for _, _, u in CLASSES.values()}
                   | {h for c in out_clips for h in c.get("hcaps", [])})
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            t = proc(text=texts[i:i + 256], return_tensors="pt", padding=True).to(dev)
            e = clap.get_text_features(**t); e = getattr(e, "pooler_output", e)
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    torch.save({"texts": texts, "emb": torch.cat(embs)}, osp.join(a.out, "text_emb.pt"))
    json.dump(dict(hi=m25["hi"], lo=LO, classes={k: v[1] for k, v in CLASSES.items()},
                   unseen={k: v[2] for k, v in CLASSES.items()}, report=report, clips=out_clips),
              open(osp.join(a.out, "meta.json"), "w"), indent=0)
    print("PREP_DONE", len(out_clips), "texts", len(texts), flush=True)


if __name__ == "__main__":
    main()
