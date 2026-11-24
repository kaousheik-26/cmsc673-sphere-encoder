"""Train a text-conditioned flow-matching DiT in the latent space of a FROZEN SE2 (encoder + decoder frozen).

Each step: random 8.9 s mel crop (+-gain) -> image -> frozen SE2 encoder -> spherified latent z [256, 384]
-> normalise -> flow-matching loss for the DiT, conditioned on a CLAP text embedding (class template caption
or, for AudioCaps clips, a human caption). The frozen SE2 decoder is only used to listen to samples.
Resumes from <out>/ckpt_last.pt. Stops after --max_hours (wall clock) or --max_steps.
torchrun --nproc_per_node 4 scripts/train_latent.py --data data1k --out runs/latent1k"""
import argparse, copy, datetime, json, math, os, os.path as osp, random, sys, time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))
from audiomel import FRAMES, LO, SR, to_img, from_img, colour  # noqa
from textse2 import load_run, ROOT  # noqa
from latent_dit import LatentDiT, fm_loss, sample  # noqa


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=f"{ROOT}/data1k")
    p.add_argument("--out", required=True)
    p.add_argument("--ae_ckpt", default=f"{ROOT}/runs/text25c_20k/ckpt_last.pt", help="frozen SE2 (EMA weights)")
    p.add_argument("--bs", type=int, default=64, help="per GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--drop_text", type=float, default=0.1)
    p.add_argument("--hcap_p", type=float, default=0.5, help="prob. of a human AudioCaps caption when the clip has one")
    p.add_argument("--gain_db", type=float, default=3.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--max_steps", type=int, default=10 ** 9)
    p.add_argument("--epochs", type=float, default=0, help=">0: train this many passes over the train clips (overrides max_steps)")
    p.add_argument("--max_hours", type=float, default=20.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=10000)
    p.add_argument("--eval_n", type=int, default=4, help="samples per prompt in the periodic eval")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class Frozen:
    """Frozen SE2 autoencoder: mel image <-> spherified latent."""

    def __init__(self, ckpt, dev):
        self.m, _ = load_run(ckpt, use_ema=True, device=dev)
        self.m.requires_grad_(False).eval()
        self.g1 = self.m.g1
        self.dev = dev

    def y(self, B):
        return torch.full((B,), self.g1.num_classes, dtype=torch.long, device=self.dev)

    @torch.no_grad()
    def encode(self, img01):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.g1.encoder(img01 * 2 - 1, self.y(img01.shape[0]))
        return self.g1.f(z.float())

    @torch.no_grad()
    def decode(self, z):
        g1, B = self.g1, z.shape[0]
        z = g1.f(z.float())
        c = self.m.tc.null.expand(B, -1)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = g1.decoder(z, self.y(B), alpha=g1._dec_angle(B, 0.0, self.dev), cond_embed=c.to(torch.bfloat16))
        return (x.float() * 0.5 + 0.5).clamp(0, 1)


def main():
    a = get_args()
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lrank); dev = torch.device("cuda", lrank)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    os.makedirs(osp.join(a.out, "vis"), exist_ok=True)
    t_start = time.time()

    meta = json.load(open(osp.join(a.data, "meta.json"))); hi = meta["hi"]
    mels = torch.load(osp.join(a.data, "mels.pt"), mmap=True)
    te = torch.load(osp.join(a.data, "text_emb.pt")); t2i = {t: i for i, t in enumerate(te["texts"])}
    temb_all = te["emb"].to(dev)
    clips = meta["clips"]
    tr = [i for i, c in enumerate(clips) if c["split"] == "train"]
    classes = sorted(meta["classes"])
    tmpl = [[t2i[t] for t in clips[i]["captions"]] for i in tr]
    hcap = [[t2i[t] for t in clips[i].get("hcaps", [])] for i in tr]
    nfr = torch.tensor([clips[i].get("nfr", mels.shape[-1]) for i in tr])
    if rank == 0:
        print(f"train clips {len(tr)}, classes {len(classes)}, texts {len(te['texts'])}, world {world}", flush=True)
        json.dump(vars(a), open(osp.join(a.out, "args.json"), "w"), indent=1)

    if a.epochs > 0:
        a.max_steps = max(1, math.ceil(a.epochs * len(tr) / (a.bs * world)))
        a.warmup = min(a.warmup, max(1, a.max_steps // 10))
        if rank == 0:
            print(f"{a.epochs} epoch(s) = {a.max_steps} steps at global batch {a.bs * world}; warmup {a.warmup}", flush=True)
    ae = Frozen(a.ae_ckpt, dev)
    model = LatentDiT(hidden=a.hidden, depth=a.depth, heads=a.heads).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    step = 0
    ck_path = osp.join(a.out, "ckpt_last.pt")
    if osp.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if rank == 0:
            print("resumed from step", step, flush=True)
    if rank == 0:
        print("DiT params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    g = torch.Generator().manual_seed(a.seed * 1000 + rank + 7919 * step)
    rng = random.Random(a.seed * 1000 + rank + 7919 * step)

    def batch(B):
        k = torch.randint(len(tr), (B,), generator=g)
        ms = []
        for j in k.tolist():
            s = int(torch.randint(max(1, int(nfr[j]) - FRAMES + 1), (1,), generator=g))
            ms.append(mels[tr[j]][:, s:s + FRAMES].float())
        m = torch.stack(ms).to(dev, non_blocking=True)
        m = m + (torch.rand(B, 1, 1, device=dev) * 2 - 1) * (a.gain_db / (20 / math.log(10)))
        img = to_img(m.clamp(LO, hi), hi)
        ti = [rng.choice(hcap[j]) if hcap[j] and rng.random() < a.hcap_p else rng.choice(tmpl[j]) for j in k.tolist()]
        return ae.encode(img), temb_all[ti]

    # per-channel latent statistics (fixed for the whole run)
    if step == 0:
        s1 = torch.zeros(384, device=dev, dtype=torch.float64); s2 = torch.zeros_like(s1); n = torch.zeros(1, device=dev, dtype=torch.float64)
        for _ in range(16):
            z, _ = batch(a.bs)
            s1 += z.double().sum((0, 1)); s2 += (z.double() ** 2).sum((0, 1)); n += z.shape[0] * z.shape[1]
        for t in (s1, s2, n):
            dist.all_reduce(t)
        mu = s1 / n; sd = (s2 / n - mu ** 2).clamp_min(1e-8).sqrt()
        model.mu.copy_(mu.float()); model.sd.copy_(sd.float()); ema.mu.copy_(mu.float()); ema.sd.copy_(sd.float())
        if rank == 0:
            print(f"latent stats: |mu| mean {mu.abs().mean().item():.4f}, sd mean {sd.mean().item():.4f} "
                  f"(min {sd.min().item():.4f}, max {sd.max().item():.4f})", flush=True)

    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lrank])
    eval_state = {}

    def evaluate(step):
        """rank 0: generate 25 classes x (train caption, unseen caption) x eval_n; CLAP 25-way accuracy."""
        from evaluate import load_vocoder
        from transformers import ClapModel, ClapProcessor
        from scipy.signal import resample_poly
        from torchvision.utils import save_image
        import soundfile as sf
        if not eval_state:
            eval_state["voc"] = load_vocoder(dev)
            eval_state["clap"] = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
            eval_state["proc"] = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            with torch.no_grad():
                t = eval_state["proc"](text=[f"the sound of {meta['classes'][c][0]}" for c in classes],
                                       return_tensors="pt", padding=True).to(dev)
                ct = eval_state["clap"].get_text_features(**t)
                eval_state["ct"] = F.normalize(getattr(ct, "pooler_output", ct), dim=-1)
        voc, clap, proc, ct = eval_state["voc"], eval_state["clap"], eval_state["proc"], eval_state["ct"]
        res = {"step": step}
        gn = torch.Generator(device=dev).manual_seed(123)
        noise = torch.randn(a.eval_n, 256, 384, device=dev, generator=gn)
        for kind, prompts in (("train", [meta["classes"][c][0] for c in classes]),
                              ("unseen", [meta["unseen"][c] for c in classes])):
            preds, pics = [], []
            for ci, p in enumerate(prompts):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = sample(ema, temb_all[t2i[p]][None].expand(a.eval_n, -1), steps=50, cfg=3.0, shift=a.shift,
                               noise=noise)
                m = from_img(ae.decode(ema.denorm(x)), hi)
                with torch.no_grad():
                    w = [voc(mm[None]).squeeze().clamp(-1, 1).float().cpu().numpy() for mm in m]
                    inp = proc(audio=[resample_poly(ww, 48000, SR).astype(np.float32) for ww in w],
                               sampling_rate=48000, return_tensors="pt").to(dev)
                    e = clap.get_audio_features(**inp); e = F.normalize(getattr(e, "pooler_output", e), dim=-1)
                preds += ((e @ ct.T).argmax(1) == ci).float().tolist()
                pics.append(colour(m[0], hi))
                if kind == "train" and ci % 5 == 0:
                    sf.write(osp.join(a.out, "vis", f"s{step:07d}_{classes[ci]}.wav"), w[0], SR, subtype="PCM_16")
            res[f"clap_acc_{kind}"] = float(np.mean(preds))
            save_image(torch.stack(pics), osp.join(a.out, "vis", f"s{step:07d}_{kind}.png"), nrow=5, padding=2, pad_value=1)
        return res

    log = open(osp.join(a.out, "log.jsonl"), "a")
    t0, last = time.time(), step
    model.train()
    while step < a.max_steps:
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr
        x0, temb = batch(a.bs)
        drop = torch.rand(a.bs, device=dev) < a.drop_text
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fm_loss(ddp, model.norm(x0), temb, drop, a.shift)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1 - d)
        step += 1
        if step % a.log_every == 0:
            l = loss.detach(); dist.all_reduce(l); l = l.item() / world
            if rank == 0:
                r = dict(step=step, loss=round(l, 5), gn=round(gnorm.item(), 3), lr=lr,
                         it_s=round((step - last) / (time.time() - t0), 2), hours=round((time.time() - t_start) / 3600, 3),
                         mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
                print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            t0, last = time.time(), step
        out_of_time = (time.time() - t_start) / 3600 > a.max_hours
        flag = torch.tensor([float(out_of_time)], device=dev); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        stop = flag.item() > 0 or step >= a.max_steps
        if (step % a.save_every == 0 or stop) and rank == 0:
            tmp = ck_path + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(a), "hi": hi}, tmp)
            os.replace(tmp, ck_path)
            if step % (a.save_every * 10) == 0:
                torch.save({"ema": ema.state_dict(), "step": step, "args": vars(a), "hi": hi},
                           osp.join(a.out, f"ema_step{step:07d}.pt"))
        if (step % a.eval_every == 0 or stop) and rank == 0:
            try:
                r = evaluate(step)
                print("EVAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            except Exception as ex:  # an eval failure must not kill training
                print("EVAL_FAILED", repr(ex), flush=True)
            model.train()
        if step % a.eval_every == 0 or stop:
            dist.barrier()
        if stop:
            break
    if rank == 0:
        print("TRAIN_DONE", step, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
