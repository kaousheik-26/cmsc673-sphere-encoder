"""Fine-tune SE2 (released large fd-lite 256px) on AudioSet mel images with CLAP text conditioning.
Run: torchrun --standalone --nproc_per_node 4 scripts/train.py --out runs/<name>
Losses = released recipe minus the score / FD terms: L1 (low angles) + LPIPS + ConvNeXt perceptual,
latent consistency (high angles), weight anchor 70-5-soft, noise angles U[0, 90] deg.
"""
import argparse, json, math, os, os.path as osp, random, sys, time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

sys.path.insert(0, osp.dirname(__file__))
from audiomel import FRAMES, to_img, from_img, DB  # noqa
from textse2 import build, SE2_REPO, ROOT  # noqa
from sphere.loss import G1Loss  # noqa


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=f"{ROOT}/data")
    p.add_argument("--out", required=True)
    p.add_argument("--classes", default="all", help="comma list of class names, or 'all'")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=16, help="per GPU")
    p.add_argument("--lr", type=float, default=5e-5, help="decoder lr")
    p.add_argument("--enc_lr_scale", type=float, default=0.1)
    p.add_argument("--cond_lr", type=float, default=5e-4, help="text-cond MLP lr")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--drop_text", type=float, default=0.1)
    p.add_argument("--gain_db", type=float, default=3.0, help="random +-gain augmentation")
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act_ckpt", type=int, default=0)
    return p.parse_args()


def main():
    a = get_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    torch.manual_seed(a.seed + rank); random.seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    os.makedirs(osp.join(a.out, "vis"), exist_ok=True)
    log = open(osp.join(a.out, "log.jsonl"), "a") if rank == 0 else None
    if rank == 0:
        json.dump(vars(a), open(osp.join(a.out, "args.json"), "w"), indent=1)

    # ---------------- data (tiny: keep on GPU) ----------------
    meta = json.load(open(osp.join(a.data, "meta.json")))
    hi = meta["hi"]
    mels = torch.load(osp.join(a.data, "mels.pt")).float()
    te = torch.load(osp.join(a.data, "text_emb.pt"))
    t2i = {t: i for i, t in enumerate(te["texts"])}
    temb_all = te["emb"].to(dev)
    keep = set(meta["classes"]) if a.classes == "all" else set(a.classes.split(","))
    tr = [i for i, c in enumerate(meta["clips"]) if c["split"] == "train" and c["cls"] in keep]
    tst = [i for i, c in enumerate(meta["clips"]) if c["split"] == "test" and c["cls"] in keep]
    tr_mels = mels[tr].to(dev)
    tr_caps = [[t2i[c] for c in meta["clips"][i]["captions"]] for i in tr]
    te_mels = mels[tst][..., :FRAMES].to(dev)
    classes = sorted(keep)
    if rank == 0:
        print(f"train clips {len(tr)}, test clips {len(tst)}, classes {classes}", flush=True)

    def batch(B):
        idx = torch.randint(len(tr), (B,))
        T = tr_mels.shape[-1]
        st = torch.randint(0, T - FRAMES + 1, (B,))
        m = torch.stack([tr_mels[i, :, s:s + FRAMES] for i, s in zip(idx.tolist(), st.tolist())])
        m = m + (torch.rand(B, 1, 1, device=dev) * 2 - 1) * a.gain_db / DB  # gain in log-magnitude
        x = to_img(m, hi) * 2 - 1
        cap = torch.tensor([random.choice(tr_caps[i]) for i in idx.tolist()], device=dev)
        drop = torch.rand(B, device=dev) < a.drop_text
        return x, temb_all[cap], drop

    # ---------------- model ----------------
    model, sargs = build(for_train=True, load_released=True)
    model.g1.ddp_rank, model.g1.ddp_world_size = rank, world
    model.to(dev)
    import copy
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    loss_fn = G1Loss(
        latent_spherify_mode=sargs.spherify_mode,
        perceptual_ckpt_path=osp.join(ROOT, "workspace", "pretrained", "lpips"),
        perceptual_convnext_aug=sargs.perceptual_convnext_aug,
        perceptual_convnext_loss_type=sargs.perceptual_convnext_loss_type,
        perceptual_convnext_loss_reduction=sargs.perceptual_convnext_loss_reduction,
        distance_loss_type=sargs.distance_loss_type,
        distance_weight=sargs.pix_recon_dist_loss_weight,
        distance_apply_regime=sargs.pix_recon_dist_apply_regime,
        perceptual_weight=sargs.pix_recon_perc_loss_weight,
        latent_consistency_weight=sargs.lat_con_loss_weight,
        latent_consistency_apply_regime=sargs.lat_con_apply_regime,
        weight_anchor_loss=sargs.weight_anchor_loss,
        weight_anchor_cutoff_params=sargs.weight_anchor_cutoff_params,
        use_score=False, fd_weight=0.0, fd_judge_weight=0.0,
        ptdtype=torch.bfloat16,
    ).to(dev)

    enc_p = [p for n, p in model.g1.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    dec_p = [p for n, p in model.g1.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    tc_p = list(model.tc.parameters())
    groups = [dict(params=dec_p, lr=a.lr, base=a.lr), dict(params=enc_p, lr=a.lr * a.enc_lr_scale, base=a.lr * a.enc_lr_scale),
              dict(params=tc_p, lr=a.cond_lr, base=a.cond_lr)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    if rank == 0:
        n = lambda ps: sum(p.numel() for p in ps) / 1e6
        print(f"params: dec {n(dec_p):.1f}M enc {n(enc_p):.1f}M textcond {n(tc_p):.2f}M", flush=True)
    ddp = DDP(model, device_ids=[lrank], find_unused_parameters=False)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    # fixed eval inputs
    ev_prompts = [meta["classes"][c][0] for c in classes] + [meta["unseen"][c] for c in classes]
    ev_temb = temb_all[[t2i[p] for p in ev_prompts]]
    g = torch.Generator(device=dev).manual_seed(123)
    ev_noise = torch.randn(4 * len(ev_prompts), *model.g1.latent_shape[1:], device=dev, generator=g)

    def evaluate(step, m, tag):
        m.eval()
        with autocast:
            x = to_img(te_mels, hi)
            rec = torch.cat([m.reconstruct(x[i:i + 8] * 2 - 1) for i in range(0, len(x), 8)])
            temb = ev_temb.repeat_interleave(4, 0)
            gen = torch.cat([m.generate(temb[i:i + 8], steps=2, noise=ev_noise[i:i + 8]) for i in range(0, len(temb), 8)])
        mel_err = (DB * (from_img(rec, hi) - from_img(x, hi)).abs().mean()).item()
        save_image(torch.cat([x[:10], rec[:10]]), osp.join(a.out, "vis", f"rec_{tag}_{step:06d}.png"), nrow=10, padding=2, pad_value=1)
        save_image(gen, osp.join(a.out, "vis", f"gen_{tag}_{step:06d}.png"), nrow=8, padding=2, pad_value=1)
        m.train()
        return {f"test_rec_mel_err_db_{tag}": mel_err}

    # ---------------- train ----------------
    model.train()
    t0 = time.time()
    for step in range(a.steps + 1):
        if rank == 0 and (step % a.eval_every == 0 or step == a.steps):
            r = evaluate(step, model, "raw"); r.update(evaluate(step, ema, "ema"))
            r["step"] = step; print("EVAL", r, flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if rank == 0 and step > 0 and (step % a.save_every == 0 or step == a.steps):
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step, "args": vars(a), "hi": hi},
                       osp.join(a.out, "ckpt_last.pt"))
            print("saved", step, flush=True)
        dist.barrier()
        if step == a.steps:
            break

        lr_f = min(1.0, (step + 1) / a.warmup)
        for gr in opt.param_groups:
            gr["lr"] = gr["base"] * lr_f
        x, temb, drop = batch(a.bs)
        model.g1.step_seed = a.seed * 1000003 + step
        with autocast:
            pixels, hs, lat = ddp(x, temb, drop)
        loss = loss_fn(input=pixels, target=x, epoch=0, step=step, hidden_states=hs, latent_list=lat,
                       autocast_ctx=autocast, class_labels=None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        with torch.no_grad():
            d = a.ema * min(1.0, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1 - d) if pm.requires_grad else pe.copy_(pm)
        if rank == 0 and step % 50 == 0:
            ld = {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else None) for k, v in loss_fn.log_dict.items()}
            ld = {k: round(v, 5) for k, v in ld.items() if v is not None}
            r = dict(step=step, loss=round(loss.item(), 5), gn=round(gn.item(), 3), lr=opt.param_groups[0]["lr"],
                     sec=round(time.time() - t0, 1), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), **ld)
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    dist.destroy_process_group()
    if rank == 0:
        print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
