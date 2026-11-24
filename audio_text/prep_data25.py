"""25-class AudioSet-strong subset with label checking.
A clip is kept for class C only if
  * C's strong label covers >= 5 s of the 10 s clip,
  * no speech / singing / music label covers > 3 s (unless C itself is one of those),
  * no other selected class covers >= 3 s (so each clip has one clear class),
  * CLAP ranks C's name within the top 10 of all 456 AudioSet labels for this audio (top 20 used only
    to fill classes that are short), i.e. the audio actually sounds like the label.
Per class: 10 held-out test clips, then up to 100 train clips.
Needs data/pool_cov5.0_r20.csv (scripts/class_pool.py 5 20) and data/clap_all (scripts/clap_embed_all.py).
Writes <out>/meta.json, mels.pt, text_emb.pt, wav/."""
import argparse, json, os, os.path as osp, random, sys
import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import resample_poly

sys.path.insert(0, osp.dirname(__file__))
from audiomel import SR, mel_spectrogram, LO  # noqa

D = "/fs/gamma-projects/audio/audio_datasets/audioset_strong"
# name: (mids, 3 training captions, 1 unseen caption)
CLASSES = {
    "male_speech": (["/m/05zppz"], ["a man speaking", "a male voice talking", "a man giving a speech"], "a guy talking into a microphone"),
    "female_speech": (["/m/02zsn"], ["a woman speaking", "a female voice talking", "a woman giving a speech"], "a lady talking calmly"),
    "child_speech": (["/m/0ytgt"], ["a child speaking", "a young kid talking", "the voice of a little child"], "a toddler chatting"),
    "male_singing": (["/t/dd00003"], ["a man singing", "a male vocalist singing a song", "a man singing with music"], "a male singer performing a ballad"),
    "female_singing": (["/t/dd00004"], ["a woman singing", "a female vocalist singing a song", "a woman singing with music"], "a female singer performing a ballad"),
    "choir": (["/m/0l14jd"], ["a choir singing", "a chorus of many voices singing together", "a church choir"], "a group of singers in harmony"),
    "rapping": (["/m/06bxc"], ["a man rapping over a beat", "hip hop rap vocals", "fast rapping with drums"], "a rapper performing a verse"),
    "crowd_babble": (["/m/07qfr4h"], ["a crowd of people talking", "the babble of many voices", "people chatting in a busy room"], "a noisy cafeteria full of conversation"),
    "bird_song": (["/m/020bb7"], ["birds singing", "birds chirping and calling", "bird song in the morning"], "songbirds in a forest"),
    "train": (["/m/07jdr"], ["a train passing by", "a train rumbling on the tracks", "a train horn and wheels on rails"], "a locomotive going past a station"),
    "race_car": (["/m/0ltv"], ["a race car speeding by", "race cars on a track", "the roar of a racing car engine"], "a formula one car zooming past"),
    "motorcycle": (["/m/04_sv"], ["a motorcycle engine", "a motorbike revving", "a motorcycle driving by"], "a dirt bike accelerating"),
    "truck": (["/m/07r04"], ["a truck engine", "a large truck driving by", "a diesel truck idling"], "a lorry rumbling down the road"),
    "subway": (["/m/0195fx"], ["a subway train", "a metro train in a tunnel", "an underground train arriving"], "riding the metro"),
    "helicopter": (["/m/09ct_"], ["a helicopter flying overhead", "the chopping sound of helicopter rotor blades", "a helicopter hovering"], "a helicopter passing by"),
    "chainsaw": (["/m/01j4z9"], ["a chainsaw cutting wood", "the buzzing of a chainsaw engine", "a chainsaw revving"], "someone using a chainsaw in the forest"),
    "power_tool": (["/m/0_ksk"], ["a power tool running", "an electric drill whirring", "a power saw cutting"], "someone working with power tools in a workshop"),
    "vacuum_cleaner": (["/m/0d31p"], ["a vacuum cleaner running", "the hum of a vacuum cleaner", "someone vacuuming the floor"], "a hoover cleaning a carpet"),
    "water_tap": (["/m/02jz0l"], ["water running from a tap", "a faucet running", "water pouring into a sink"], "someone washing hands at a sink"),
    "church_bell": (["/m/03w41f"], ["church bells ringing", "the sound of a church bell tolling", "a large bell ringing in a tower"], "bells ringing at a cathedral"),
    "civil_defense_siren": (["/m/0dgbq"], ["a civil defense siren wailing", "a loud air raid siren rising and falling", "an emergency warning siren"], "a tornado warning siren"),
    "fire_engine_siren": (["/m/012ndj"], ["a fire truck siren", "a fire engine with its siren on", "an emergency vehicle siren passing"], "a fire brigade rushing to a fire"),
    "rain": (["/m/06mb1", "/t/dd00038"], ["rain falling", "the sound of heavy rain pouring", "rain hitting the ground"], "a rainstorm"),
    "waves": (["/m/034srq"], ["ocean waves crashing", "waves breaking on the shore", "the surf at the beach"], "the sea at the seaside"),
    "engine_knocking": (["/m/01h82_"], ["an engine knocking", "a rough idling engine ticking", "a diesel engine clattering"], "an old car engine running unevenly"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(osp.join(a.out, "wav"), exist_ok=True)
    pool = pd.read_csv("data/pool_cov5.0_r20.csv")
    # coverage of every label per clip (both tsvs) for the exclusivity check
    covs = []
    for d, tsv in [("audio_clips_train", "audioset_train_strong.tsv"), ("audio_clips", "audioset_eval_strong.tsv")]:
        df = pd.read_csv(f"{D}/{tsv}", sep="\t"); df["dur"] = df.end_time_seconds - df.start_time_seconds
        c = df.groupby(["segment_id", "label"]).dur.sum().clip(upper=10).reset_index(); c["dir"] = d; covs.append(c)
    cov = pd.concat(covs)
    labels_of = cov.groupby(["dir", "segment_id"]).label.apply(list).to_dict()
    sel_mids = {m: k for k, (ms, _, _) in CLASSES.items() for m in ms}
    big = cov[cov.label.isin(sel_mids) & (cov.dur >= 3)]
    classes_of = big.assign(c=big.label.map(sel_mids)).groupby(["dir", "segment_id"]).c.apply(set).to_dict()

    rng = random.Random(0)
    used, meta, mels, report = set(), [], [], {}
    # smallest pools first so they get first pick of shared clips
    order = sorted(CLASSES, key=lambda k: pool[pool.label.isin(CLASSES[k][0])].segment_id.nunique())
    for cname in order:
        mids, caps, _ = CLASSES[cname]
        p = pool[pool.label.isin(mids)].sort_values("r").drop_duplicates(["dir", "segment_id"])
        p = p[[classes_of.get((d, s), {cname}) <= {cname} for d, s in zip(p.dir, p.segment_id)]]
        tiers = [p[p.r <= 10], p[p.r > 10]]
        cands = []
        for t in tiers:
            t = list(zip(t.dir, t.segment_id)); rng.shuffle(t); cands += t
        picked = []
        for d, sid in cands:
            if sid in used:
                continue
            w, sr = sf.read(f"{D}/{d}/{sid}.wav", dtype="float32", always_2d=True)
            w = w.mean(1)
            if len(w) < 10 * sr or np.sqrt((w ** 2).mean()) < 1e-3:
                continue
            w = resample_poly(w[:10 * sr], SR, sr).astype(np.float32)
            w = np.clip(w / max(1.0, np.abs(w).max()), -1, 1)
            mels.append(mel_spectrogram(torch.from_numpy(w)[None])[0])
            sf.write(osp.join(a.out, "wav", f"{sid}.wav"), w, SR, subtype="PCM_16")
            split = "test" if len(picked) < a.n_test else "train"
            picked.append(sid); used.add(sid)
            row = p[(p.dir == d) & (p.segment_id == sid)].iloc[0]
            meta.append(dict(id=sid, dir=d, cls=cname, split=split, captions=caps,
                             labels=sorted(set(labels_of[(d, sid)])), clap_rank=int(row.r), cov_s=float(row.dur)))
            if len(picked) == a.n_train + a.n_test:
                break
        ntr = len(picked) - min(len(picked), a.n_test)
        report[cname] = dict(train=ntr, test=min(len(picked), a.n_test), candidates=len(cands))
        print(f"{cname:20s} train {ntr:3d} test {min(len(picked), a.n_test):2d}  (candidates {len(cands)})", flush=True)

    mels = torch.stack(mels)
    hi = 1.5
    print("mels", mels.shape, "max", mels.max().item(), "99.9pct", torch.quantile(mels[::7].flatten()[::50].float(), 0.999).item())
    torch.save(mels.half(), osp.join(a.out, "mels.pt"))
    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    texts = sorted({c for _, cs, _ in CLASSES.values() for c in cs} | {u for _, _, u in CLASSES.values()})
    with torch.no_grad():
        t = proc(text=texts, return_tensors="pt", padding=True)
        e = clap.get_text_features(**t); e = getattr(e, "pooler_output", e)
        e = torch.nn.functional.normalize(e, dim=-1)
    torch.save({"texts": texts, "emb": e}, osp.join(a.out, "text_emb.pt"))
    json.dump(dict(hi=hi, lo=LO, classes={k: v[1] for k, v in CLASSES.items()}, unseen={k: v[2] for k, v in CLASSES.items()},
                   mids={k: v[0] for k, v in CLASSES.items()}, report=report, clips=meta),
              open(osp.join(a.out, "meta.json"), "w"), indent=1)
    print("PREP_DONE", len(meta))


if __name__ == "__main__":
    main()
