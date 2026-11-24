"""CLAP audio embeddings for every downloaded AudioSet-strong clip (train + eval dirs), one shard per GPU.
Also writes CLAP text embeddings for every AudioSet label name (shard 0).
Usage: python clap_embed_all.py --shard i --nshard 4 --out data/clap_all"""
import argparse, os, os.path as osp
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from scipy.signal import resample_poly
from concurrent.futures import ThreadPoolExecutor
from transformers import ClapModel, ClapProcessor

D = "/fs/gamma-projects/audio/audio_datasets/audioset_strong"


def load(path):
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None, 0.0, 0.0
    w = w.mean(1)
    dur, rms = len(w) / sr, float(np.sqrt((w ** 2).mean())) if len(w) else 0.0
    w = w[:10 * sr]
    if len(w) < sr:
        return None, dur, rms
    w = np.pad(w, (0, 10 * sr - len(w)))
    return resample_poly(w, 48000, sr).astype(np.float32), dur, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshard", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="audioset_strong", choices=["audioset_strong", "audiocaps"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N files only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.src == "audiocaps":
        R = "/fs/gamma-projects/audio/audio_datasets/audiocaps/2.0/audios"
        files = [(f"{R}/audiocaps_raw_audio/{f}", "audiocaps") for f in sorted(os.listdir(f"{R}/audiocaps_raw_audio"))
                 if f.endswith(".wav")]
    else:
        files = [(f"{D}/{d}/{f}", d) for d in ("audio_clips_train", "audio_clips") for f in sorted(os.listdir(f"{D}/{d}"))
                 if f.endswith(".wav")]
    if a.limit:
        files = files[:a.limit]
    files = files[a.shard::a.nshard]
    dev = "cuda"
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval().half()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    if a.shard == 0 and a.src == "audioset_strong":
        names = pd.read_csv(f"{D}/mid_to_display_name.tsv", sep="\t", header=None)
        texts = [f"the sound of {n.lower()}" for n in names[1]]
        with torch.no_grad():
            t = proc(text=texts, return_tensors="pt", padding=True).to(dev)
            e = clap.get_text_features(**t); e = F.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
        torch.save({"mids": list(names[0]), "names": list(names[1]), "emb": e.cpu()}, f"{a.out}/labels.pt")

    ids, split, durs, rmss, embs = [], [], [], [], []
    B = 64
    pool = ThreadPoolExecutor(12)
    for i in range(0, len(files), B):
        chunk = files[i:i + B]
        res = list(pool.map(lambda p: load(p[0]), chunk))
        ok = [(c, r) for c, r in zip(chunk, res) if r[0] is not None]
        if not ok:
            continue
        with torch.no_grad():
            inp = proc(audio=[r[0] for _, r in ok], sampling_rate=48000, return_tensors="pt").to(dev)
            inp = {k: (v.half() if v.is_floating_point() else v) for k, v in inp.items()}
            e = clap.get_audio_features(**inp); e = F.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
        embs.append(e.cpu())
        for (c, r) in ok:
            ids.append(osp.basename(c[0])[:-4]); split.append(c[1]); durs.append(r[1]); rmss.append(r[2])
        if (i // B) % 50 == 0:
            print(a.shard, i, len(files), flush=True)
    torch.save({"ids": ids, "dir": split, "dur": durs, "rms": rmss, "emb": torch.cat(embs)},
               f"{a.out}/audio_{a.shard}.pt")
    print("SHARD_DONE", a.shard, len(ids), flush=True)


if __name__ == "__main__":
    main()
