"""Evaluate a text-conditioned SE2 run.
 1. Reconstruction of the 25 held-out test clips (angle 0): released SE2 vs fine-tuned. Mel error dB / mel PSNR.
 2. Text -> mel generation for each class with (a) a training caption and (b) an unseen caption,
    over a small sampler grid (forward steps x angle-CFG). Mel -> audio with BigVGAN v2 22 kHz.
 3. CLAP zero-shot classification of the generated audio among the 5 classes (and of real test audio as reference),
    and a memorisation check: distance of each generated mel to its nearest training crop vs. the same for real test clips.
Writes <out>/metrics.json, wavs, mel pictures.
"""
import argparse, json, os, os.path as osp, sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torchvision.utils import save_image

sys.path.insert(0, osp.dirname(__file__))
from audiomel import FRAMES, SR, DB, LO, to_img, from_img, colour, mel_spectrogram  # noqa
from textse2 import build, load_run, ROOT  # noqa

BIGVGAN = f"{ROOT}/hf/hub/models--nvidia--bigvgan_v2_22khz_80band_fmax8k_256x/snapshots/a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8"


def load_vocoder(dev):
    sys.path.append(f"{ROOT}/BigVGAN")
    import bigvgan
    v = bigvgan.BigVGAN._from_pretrained(model_id=BIGVGAN, revision=None, cache_dir=None, force_download=False,
                                         proxies=None, resume_download=False, local_files_only=True, token=None,
                                         use_cuda_kernel=False)
    v.remove_weight_norm()
    return v.eval().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="samples per prompt")
    ap.add_argument("--steps", default="1,2,4")
    ap.add_argument("--cfg", default="0,2,4")
    ap.add_argument("--use_ema", type=int, default=1)
    ap.add_argument("--save_wav_n", type=int, default=3, help="wavs saved per prompt (main setting)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(osp.join(a.out, "wav"), exist_ok=True); os.makedirs(osp.join(a.out, "img"), exist_ok=True)
    DATA = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"].get("data", f"{ROOT}/data")
    meta = json.load(open(f"{DATA}/meta.json")); hi = meta["hi"]
    mels = torch.load(f"{DATA}/mels.pt").float()
    te = torch.load(f"{DATA}/text_emb.pt"); t2i = {t: i for i, t in enumerate(te["texts"])}
    temb_all = te["emb"].to(dev)
    clips = meta["clips"]
    run_args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"]
    keep = sorted(meta["classes"]) if run_args["classes"] == "all" else sorted(run_args["classes"].split(","))
    tr = [i for i, c in enumerate(clips) if c["split"] == "train" and c["cls"] in keep]
    ts = [i for i, c in enumerate(clips) if c["split"] == "test" and c["cls"] in keep]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    voc = load_vocoder(dev)
    res = {"ckpt": a.ckpt, "ema": bool(a.use_ema), "classes": keep}

    def vocode(mel):  # [80, T] -> np audio
        with torch.inference_mode():
            return voc(mel[None].to(dev)).squeeze().clamp(-1, 1).float().cpu().numpy()

    def mel_err(m, ref):
        return float(DB * (m - ref).abs().mean())

    def mel_psnr(m, ref):
        mse = (((m - ref) / (hi - LO)) ** 2).mean()
        return float(-10 * torch.log10(mse))

    # ---------------- 1. reconstruction ----------------
    tm = mels[ts][..., :FRAMES].clamp(LO, hi).to(dev)
    x = to_img(tm, hi)
    base, _ = build(for_train=False, load_released=True); base = base.to(dev).eval()
    with autocast:
        rec0 = torch.cat([base.reconstruct(x[i:i + 8] * 2 - 1) for i in range(0, len(x), 8)])
    del base; torch.cuda.empty_cache()
    model, ck = load_run(a.ckpt, use_ema=bool(a.use_ema), device=dev)
    res["train_step"] = ck["step"]
    with autocast:
        rec1 = torch.cat([model.reconstruct(x[i:i + 8] * 2 - 1) for i in range(0, len(x), 8)])
        tcap = temb_all[[t2i[clips[i]["captions"][0]] for i in ts]]
        rec2 = torch.cat([model.reconstruct(x[i:i + 8] * 2 - 1, tcap[i:i + 8]) for i in range(0, len(x), 8)])
    rows = []
    for k, i in enumerate(ts):
        r = {"id": clips[i]["id"], "cls": clips[i]["cls"]}
        for name, rec in (("released", rec0), ("finetuned", rec1), ("finetuned_textcond", rec2)):
            m = from_img(rec[k], hi)
            r[name] = {"mel_err_db": mel_err(m, tm[k]), "mel_psnr": mel_psnr(m, tm[k])}
        rows.append(r)
    res["recon"] = {n: {q: float(np.mean([r[n][q] for r in rows])) for q in ("mel_err_db", "mel_psnr")}
                    for n in ("released", "finetuned", "finetuned_textcond")}
    res["recon_clips"] = rows
    print("RECON", json.dumps(res["recon"]), flush=True)
    # audio for the first test clip of each class
    for k, i in enumerate(ts):
        c = clips[i]["cls"]
        if any(clips[j]["cls"] == c for j in ts[:k]):
            continue
        orig = sf.read(f"{DATA}/wav/{clips[i]['id']}.wav")[0][:FRAMES * 256]
        sf.write(osp.join(a.out, "wav", f"recon_{c}_original.wav"), orig, SR, subtype="PCM_16")
        sf.write(osp.join(a.out, "wav", f"recon_{c}_vocoder.wav"), vocode(tm[k]), SR, subtype="PCM_16")
        sf.write(osp.join(a.out, "wav", f"recon_{c}_released.wav"), vocode(from_img(rec0[k], hi)), SR, subtype="PCM_16")
        sf.write(osp.join(a.out, "wav", f"recon_{c}_finetuned.wav"), vocode(from_img(rec1[k], hi)), SR, subtype="PCM_16")
        save_image(torch.stack([colour(tm[k], hi), colour(from_img(rec0[k], hi), hi), colour(from_img(rec1[k], hi), hi)]),
                   osp.join(a.out, "img", f"recon_{c}.png"), nrow=1, padding=4, pad_value=1)

    # ---------------- CLAP ----------------
    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    cls_text = [f"the sound of {meta['classes'][c][0]}" for c in keep]
    with torch.no_grad():
        t = proc(text=cls_text, return_tensors="pt", padding=True).to(dev)
        ct = clap.get_text_features(**t); ct = F.normalize(getattr(ct, "pooler_output", ct), dim=-1)

    def clap_audio(wavs):  # list of np arrays at SR -> [N, 512]
        w48 = [resample_poly(w, 48000, SR).astype(np.float32) for w in wavs]
        out = []
        for i in range(0, len(w48), 16):
            with torch.no_grad():
                inp = proc(audio=w48[i:i + 16], sampling_rate=48000, return_tensors="pt").to(dev)
                e = clap.get_audio_features(**inp); out.append(F.normalize(getattr(e, "pooler_output", e), dim=-1))
        return torch.cat(out)

    def clap_acc(emb, labels):
        pred = (emb @ ct.T).argmax(1).cpu()
        return float((pred == torch.tensor(labels)).float().mean())

    lab_ts = [keep.index(clips[i]["cls"]) for i in ts]
    real_w = [sf.read(f"{DATA}/wav/{clips[i]['id']}.wav")[0][:FRAMES * 256].astype(np.float32) for i in ts]
    voc_w = [vocode(tm[k]) for k in range(len(ts))]
    ft_w = [vocode(from_img(rec1[k], hi)) for k in range(len(ts))]
    res["clap_acc_reference"] = {"real_test_audio": clap_acc(clap_audio(real_w), lab_ts),
                                 "vocoded_test_mel": clap_acc(clap_audio(voc_w), lab_ts),
                                 "finetuned_recon": clap_acc(clap_audio(ft_w), lab_ts)}
    print("CLAP_REF", res["clap_acc_reference"], flush=True)

    # nearest-train distance (memorisation): min over train clips and frame offsets (stride 8) of mean |dB|
    trm = mels[tr].clamp(LO, hi).to(dev)  # [Ntr, 80, 861]
    crops = torch.stack([trm[:, :, s:s + FRAMES] for s in range(0, trm.shape[-1] - FRAMES + 1, 16)], 1).flatten(0, 1)

    def nn_dist(m):  # [N, 80, 768] -> [N] dB
        return torch.stack([(DB * (crops - q).abs().mean((1, 2))).min() for q in m]).cpu()

    res["nn_train_db_real_test"] = float(nn_dist(tm).mean())

    # ---------------- 2. generation ----------------
    prompts = [(c, "train_caption", meta["classes"][c][0]) for c in keep] + [(c, "unseen_caption", meta["unseen"][c]) for c in keep]
    gen_res = []
    g = torch.Generator(device=dev).manual_seed(7)
    noise = torch.randn(a.n, *model.g1.latent_shape[1:], device=dev, generator=g)
    steps_l = [int(s) for s in a.steps.split(",")]; cfg_l = [float(s) for s in a.cfg.split(",")]
    main_setting = (2, 0.0) if 2 in steps_l else (steps_l[0], cfg_l[0])
    for st in steps_l:
        for cf in cfg_l:
            if st == 1 and cf > 0:
                continue  # angle CFG needs >= 2 steps
            all_emb, all_lab, all_nn, grid = [], [], [], []
            for c, kind, p in prompts:
                with autocast:
                    img = model.generate(temb_all[t2i[p]][None].expand(a.n, -1), steps=st, cfg_deg=cf, noise=noise)
                gm = from_img(img, hi)
                ws = [vocode(m) for m in gm]
                all_emb.append(clap_audio(ws)); all_lab += [keep.index(c)] * a.n
                all_nn.append(nn_dist(gm))
                if (st, cf) == main_setting:
                    for j in range(min(a.save_wav_n, a.n)):
                        sf.write(osp.join(a.out, "wav", f"gen_{c}_{kind}_{j}.wav"), ws[j], SR, subtype="PCM_16")
                    save_image(torch.stack([colour(m, hi) for m in gm[:4]]), osp.join(a.out, "img", f"gen_{c}_{kind}.png"),
                               nrow=1, padding=4, pad_value=1)
                grid.append(img[:4])
            emb = torch.cat(all_emb); lab = torch.tensor(all_lab); nn = torch.cat(all_nn)
            half = len(lab) // 2
            pred = (emb @ ct.T).argmax(1).cpu()
            r = {"steps": st, "cfg_deg": cf,
                 "clap_acc_train_caption": float((pred[:half] == lab[:half]).float().mean()),
                 "clap_acc_unseen_caption": float((pred[half:] == lab[half:]).float().mean()),
                 "clap_acc_per_class_train_caption": {c: float((pred[:half][lab[:half] == k] == k).float().mean()) for k, c in enumerate(keep)},
                 "nn_train_db_mean": float(nn.mean()),
                 "within_prompt_diversity_db": float(np.mean([DB * (from_img(gg, hi)[:, None] - from_img(gg, hi)[None]).abs().mean((2, 3)).sum().item()
                                                              / (len(gg) * (len(gg) - 1)) for gg in grid]))}
            gen_res.append(r)
            print("GEN", json.dumps(r), flush=True)
            save_image(torch.cat(grid), osp.join(a.out, "img", f"grid_steps{st}_cfg{cf:g}.png"), nrow=4, padding=2, pad_value=1)
    res["generation"] = gen_res
    res["main_setting"] = {"steps": main_setting[0], "cfg_deg": main_setting[1]}
    res["prompts"] = prompts
    json.dump(res, open(osp.join(a.out, "metrics.json"), "w"), indent=1)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
