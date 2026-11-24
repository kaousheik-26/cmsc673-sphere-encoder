"""Evaluate the latent flow DiT (frozen SE2 encoder/decoder) on the 25-class test set.
 1. Ceiling: frozen SE2 encode -> decode of the 250 held-out clips (mel error dB / PSNR).
 2. CLAP reference accuracy: real test audio, vocoded true mel, frozen-SE2 reconstruction.
 3. Generation grid (Euler steps x CFG scale), train captions and unseen captions, n seeds per prompt:
    25-way CLAP accuracy, distance to the nearest training crop (memorisation), within-prompt diversity.
Writes <out>/metrics.json, wav/, img/.
python scripts/eval_latent.py --ckpt runs/latent1k/ckpt_last.pt --out runs/latent1k/eval"""
import argparse, json, os, os.path as osp, sys

import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from scipy.signal import resample_poly
from torchvision.utils import save_image

sys.path.insert(0, osp.dirname(__file__))
from audiomel import FRAMES, SR, DB, LO, to_img, from_img, colour  # noqa
from evaluate import load_vocoder  # noqa
from latent_dit import LatentDiT, sample  # noqa
from train_latent import Frozen  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", default="25,50")
    ap.add_argument("--cfg", default="1,2,3,5")
    ap.add_argument("--main", default="50,3")
    ap.add_argument("--save_wav_n", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(osp.join(a.out, "wav"), exist_ok=True); os.makedirs(osp.join(a.out, "img"), exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); ra = ck["args"]
    DATA = ra["data"]
    meta = json.load(open(f"{DATA}/meta.json")); hi = meta["hi"]
    mels = torch.load(f"{DATA}/mels.pt", mmap=True)
    te = torch.load(f"{DATA}/text_emb.pt"); t2i = {t: i for i, t in enumerate(te["texts"])}
    temb_all = te["emb"].to(dev)
    clips = meta["clips"]; keep = sorted(meta["classes"])
    tr = [i for i, c in enumerate(clips) if c["split"] == "train"]
    ts = [i for i, c in enumerate(clips) if c["split"] == "test"]
    model = LatentDiT(hidden=ra["hidden"], depth=ra["depth"], heads=ra["heads"]).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    ae = Frozen(ra["ae_ckpt"], dev)
    voc = load_vocoder(dev)
    res = {"ckpt": a.ckpt, "train_step": ck["step"], "classes": keep, "n_train": len(tr), "n_test": len(ts)}

    def vocode(m):
        with torch.inference_mode():
            return voc(m[None].to(dev)).squeeze().clamp(-1, 1).float().cpu().numpy()

    # ---- 1. frozen SE2 ceiling ----
    tm = torch.stack([mels[i][:, :FRAMES].float() for i in ts]).clamp(LO, hi).to(dev)
    rec = torch.cat([ae.decode(ae.encode(to_img(tm[i:i + 16], hi))) for i in range(0, len(ts), 16)])
    rm = from_img(rec, hi)
    res["recon_frozen_se2"] = {"mel_err_db": float(DB * (rm - tm).abs().mean()),
                               "mel_psnr": float(-10 * torch.log10((((rm - tm) / (hi - LO)) ** 2).mean()))}
    print("RECON", res["recon_frozen_se2"], flush=True)
    for k, i in enumerate(ts):
        c = clips[i]["cls"]
        if any(clips[j]["cls"] == c for j in ts[:k]):
            continue
        sf.write(osp.join(a.out, "wav", f"recon_{c}_original.wav"), sf.read(f"{DATA}/wav/{clips[i]['id']}.wav")[0][:FRAMES * 256], SR, subtype="PCM_16")
        sf.write(osp.join(a.out, "wav", f"recon_{c}_vocoder.wav"), vocode(tm[k]), SR, subtype="PCM_16")
        sf.write(osp.join(a.out, "wav", f"recon_{c}_frozen_se2.wav"), vocode(rm[k]), SR, subtype="PCM_16")

    # ---- 2. CLAP ----
    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        t = proc(text=[f"the sound of {meta['classes'][c][0]}" for c in keep], return_tensors="pt", padding=True).to(dev)
        ct = clap.get_text_features(**t); ct = F.normalize(getattr(ct, "pooler_output", ct), dim=-1)

    def clap_audio(ws):
        out = []
        for i in range(0, len(ws), 16):
            with torch.no_grad():
                inp = proc(audio=[resample_poly(w, 48000, SR).astype(np.float32) for w in ws[i:i + 16]],
                           sampling_rate=48000, return_tensors="pt").to(dev)
                e = clap.get_audio_features(**inp); out.append(F.normalize(getattr(e, "pooler_output", e), dim=-1))
        return torch.cat(out)

    def acc(emb, lab):
        return float(((emb @ ct.T).argmax(1).cpu() == torch.tensor(lab)).float().mean())

    lab_ts = [keep.index(clips[i]["cls"]) for i in ts]
    real_w = [sf.read(f"{DATA}/wav/{clips[i]['id']}.wav")[0][:FRAMES * 256].astype(np.float32) for i in ts]
    res["clap_acc_reference"] = {"real_test_audio": acc(clap_audio(real_w), lab_ts),
                                 "vocoded_test_mel": acc(clap_audio([vocode(m) for m in tm]), lab_ts),
                                 "frozen_se2_recon": acc(clap_audio([vocode(m) for m in rm]), lab_ts)}
    print("CLAP_REF", res["clap_acc_reference"], flush=True)

    # ---- nearest training crop (chunked over train clips; crops at stride 32 frames) ----
    def nn_dist(q):  # [N, 80, 768] on dev -> [N]
        best = torch.full((len(q),), float("inf"), device=dev)
        for s in range(0, len(tr), 256):
            trm = torch.stack([mels[i].float() for i in tr[s:s + 256]]).to(dev).clamp(LO, hi)
            crops = torch.stack([trm[:, :, o:o + FRAMES] for o in range(0, trm.shape[-1] - FRAMES + 1, 32)], 1).flatten(0, 1)
            for qi in range(len(q)):
                best[qi] = torch.minimum(best[qi], (DB * (crops - q[qi]).abs().mean((1, 2))).min())
        return best.cpu()

    res["nn_train_db_real_test"] = float(nn_dist(tm[::5]).mean())  # 50 test clips
    print("NN_REAL", res["nn_train_db_real_test"], flush=True)

    # ---- 3. generation ----
    prompts = [(c, "train_caption", meta["classes"][c][0]) for c in keep] + [(c, "unseen_caption", meta["unseen"][c]) for c in keep]
    noise = torch.randn(a.n, 256, 384, device=dev, generator=torch.Generator(device=dev).manual_seed(7))
    main = tuple(float(x) for x in a.main.split(","))
    gens = []
    for st in [int(s) for s in a.steps.split(",")]:
        for cf in [float(s) for s in a.cfg.split(",")]:
            is_main = (st, cf) == (int(main[0]), main[1])
            embs, labs, nns, divs = [], [], [], []
            for c, kind, p in prompts:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = sample(model, temb_all[t2i[p]][None].expand(a.n, -1), steps=st, cfg=cf, shift=ra["shift"], noise=noise)
                gm = from_img(ae.decode(model.denorm(x)), hi)
                ws = [vocode(m) for m in gm]
                embs.append(clap_audio(ws)); labs += [keep.index(c)] * a.n
                divs.append(float((DB * (gm[:, None] - gm[None]).abs().mean((2, 3))).sum() / (a.n * (a.n - 1))))
                if is_main or kind == "train_caption" and c in keep[:5]:
                    nns.append(gm[:2])
                if is_main:
                    for j in range(min(a.save_wav_n, a.n)):
                        sf.write(osp.join(a.out, "wav", f"gen_{c}_{kind}_{j}.wav"), ws[j], SR, subtype="PCM_16")
                    save_image(torch.stack([colour(m, hi) for m in gm[:4]]), osp.join(a.out, "img", f"gen_{c}_{kind}.png"),
                               nrow=1, padding=4, pad_value=1)
            emb = torch.cat(embs); lab = torch.tensor(labs); half = len(lab) // 2
            pred = (emb @ ct.T).argmax(1).cpu()
            r = {"steps": st, "cfg": cf,
                 "clap_acc_train_caption": float((pred[:half] == lab[:half]).float().mean()),
                 "clap_acc_unseen_caption": float((pred[half:] == lab[half:]).float().mean()),
                 "clap_acc_per_class_train_caption": {c: float((pred[:half][lab[:half] == k] == k).float().mean()) for k, c in enumerate(keep)},
                 "clap_acc_per_class_unseen_caption": {c: float((pred[half:][lab[half:] == k] == k).float().mean()) for k, c in enumerate(keep)},
                 "nn_train_db_mean": float(nn_dist(torch.cat(nns)).mean()),
                 "within_prompt_diversity_db": float(np.mean(divs))}
            gens.append(r)
            print("GEN", json.dumps(r), flush=True)
            json.dump({**res, "generation": gens}, open(osp.join(a.out, "metrics.json"), "w"), indent=1)
    res["generation"] = gens
    res["main_setting"] = {"steps": int(main[0]), "cfg": main[1]}
    res["prompts"] = prompts
    json.dump(res, open(osp.join(a.out, "metrics.json"), "w"), indent=1)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
