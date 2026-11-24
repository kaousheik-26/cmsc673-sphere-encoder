"""Pick 5 AudioSet-strong classes x 25 clips (20 train / 5 test), compute BigVGAN 22k mels,
fix the global log-mel scale, precompute CLAP text embeddings for all captions.
Writes <out>/meta.json, <out>/mels.pt, <out>/text_emb.pt, <out>/wav/*.wav (22.05 kHz mono)."""
import argparse, json, os, os.path as osp, random, sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, osp.dirname(__file__))
from audiomel import SR, mel_spectrogram, LO  # noqa

D = "/fs/gamma-projects/audio/audio_datasets/audioset_strong"
CLASSES = {
    "church_bell": ("/m/03w41f", ["church bells ringing", "the sound of a church bell tolling",
                                  "a large bell ringing in a tower"]),
    "siren": ("/m/0dgbq", ["a civil defense siren wailing", "a loud air raid siren rising and falling",
                           "an emergency warning siren"]),
    "rain": ("/m/06mb1", ["rain falling", "the sound of heavy rain pouring", "rain hitting the ground"]),
    "chainsaw": ("/m/01j4z9", ["a chainsaw cutting wood", "the buzzing of a chainsaw engine",
                               "a chainsaw revving"]),
    "helicopter": ("/m/09ct_", ["a helicopter flying overhead", "the chopping sound of helicopter rotor blades",
                                "a helicopter hovering"]),
}
# prompts never seen in training (test text generalisation)
UNSEEN = {"church_bell": "bells ringing at a cathedral", "siren": "a tornado warning siren",
          "rain": "a rainstorm", "chainsaw": "someone using a chainsaw in the forest",
          "helicopter": "a helicopter passing by"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(osp.join(a.out, "wav"), exist_ok=True)

    have = {f[:-4] for f in os.listdir(f"{D}/audio_clips_train")}
    df = pd.read_csv(f"{D}/audioset_train_strong.tsv", sep="\t")
    df = df[df.segment_id.isin(have)]
    df["dur"] = df.end_time_seconds - df.start_time_seconds
    cov = df.groupby(["segment_id", "label"]).dur.sum().clip(upper=10).reset_index()
    nlab = df.groupby("segment_id").label.nunique()

    rng = random.Random(0)
    used, meta, mels = set(), [], []
    for cname, (mid, caps) in CLASSES.items():
        ids = sorted(cov[(cov.label == mid) & (cov.dur >= 8.5)].segment_id)
        ids = [i for i in ids if nlab[i] <= 3 and i not in used]
        rng.shuffle(ids)
        picked = []
        for sid in ids:
            w, sr = sf.read(f"{D}/audio_clips_train/{sid}.wav", dtype="float32", always_2d=True)
            w = w.mean(1)
            if len(w) < 10 * sr or np.sqrt((w ** 2).mean()) < 1e-3:  # too short / near-silent
                continue
            w = resample_poly(w[:10 * sr], SR, sr).astype(np.float32)
            w = np.clip(w / max(1.0, np.abs(w).max()), -1, 1)
            mel = mel_spectrogram(torch.from_numpy(w)[None])[0]
            sf.write(osp.join(a.out, "wav", f"{sid}.wav"), w, SR, subtype="PCM_16")
            picked.append(sid); mels.append(mel)
            split = "train" if len(picked) <= a.n_train else "test"
            labels = sorted(set(df[df.segment_id == sid].label))
            meta.append(dict(id=sid, cls=cname, split=split, captions=caps, labels=labels))
            used.add(sid)
            if len(picked) == a.n_train + a.n_test:
                break
        print(cname, len(picked), "of", len(ids), "candidates", flush=True)

    mels = torch.stack(mels)  # [N, 80, 861]
    tr = torch.tensor([m["split"] == "train" for m in meta])
    hi = 1.5  # fixed global top; 99.9th pct was 0.94, max 1.95 -> little clipping, small range cost
    print("mel frames", mels.shape, "LO", LO, "HI(99.9%)", hi, "max", mels.max().item())
    torch.save(mels.half(), osp.join(a.out, "mels.pt"))

    # CLAP text embeddings (frozen): every training caption, unseen prompts, and class names
    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    texts = sorted({c for _, cs in CLASSES.values() for c in cs} | set(UNSEEN.values()))
    with torch.no_grad():
        t = proc(text=texts, return_tensors="pt", padding=True)
        e = clap.get_text_features(**t)
        e = getattr(e, "pooler_output", e)
        e = torch.nn.functional.normalize(e, dim=-1)
    torch.save({"texts": texts, "emb": e}, osp.join(a.out, "text_emb.pt"))
    json.dump(dict(hi=hi, lo=LO, classes={k: v[1] for k, v in CLASSES.items()}, unseen=UNSEEN, clips=meta),
              open(osp.join(a.out, "meta.json"), "w"), indent=1)
    print("text emb", e.shape, "PREP_DONE")


if __name__ == "__main__":
    main()
