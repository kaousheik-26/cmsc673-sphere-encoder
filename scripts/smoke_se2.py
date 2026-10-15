import argparse
import json
import logging
import math
import os.path as osp
import time

import torch

import sphere.rng as rng
from cli_utils import str2bool
from halfsphere.backbone import SE2Backbone
from halfsphere.data import balanced_subset, image_dataset, load_batch
from halfsphere.geometry import angle_deg, f, token_manifold_gap
from halfsphere.masks import half_mask
from halfsphere.metrics import HalfLPIPS, psnr_masked
from halfsphere.records import Run, save_grid
from halfsphere.ridgemap import DecoderOps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="SE2 adapter smoke tests")
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--job_dir", type=str, default="sphere2-base-flowers-256px")
parser.add_argument("--n_images", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--equivalence", type=str2bool, default=False, help="also run the gate here; prefer --only_equivalence True in a fresh process, since cuDNN may pick a different bf16 conv algorithm after other models ran")
parser.add_argument("--only_equivalence", type=str2bool, default=False, help="skip the adapter checks and run only the gate, in a fresh process")
parser.add_argument("--equiv_job_dir", type=str, default="sphere2-large-flowers-512px")
parser.add_argument("--equiv_png", type=str, default="workspace/visualization/sphere2-large-flowers-512px/imgs_try=00_clss=00000_pth=ep1999_ema=True_cfg=0.0-angle_steps=4_cache=True_init=84.0_loop=84.0.png")
args = parser.parse_args()

results = []


def check(name, ok, value, expect):
    results.append({"check": name, "pass": bool(ok), "value": value, "expect": expect})
    logger.info(f"{'PASS' if ok else 'FAIL'}  {name}: {value}  (expect {expect})")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    run = Run("smoke_se2", vars(args), dev_dir=args.dev_dir, seeds={"base": args.seed})
    if args.only_equivalence:
        equivalence_gate(run)
        return finish(run)
    bb = SE2Backbone.from_checkpoint(args.job_dir, dev_dir=args.dev_dir, device=dev)
    run.meta["checkpoint"]["path"] = bb.ckpt_path
    g = torch.Generator().manual_seed(args.seed)

    ds = image_dataset(bb.cfg["dataset_name"], bb.image_size, args.dev_dir)
    idx = balanced_subset(ds, args.n_images, seed=args.seed)
    x, cls, _ = load_batch(ds, idx, dev)
    B = x.shape[0]

    check("latent_shape", bb.latent_shape == (256, 384), str(bb.latent_shape), "(256, 384)")
    check("L", bb.L == 98304, bb.L, 98304)
    z = bb.encode_raw(x)
    v = f(z)
    r = (v.flatten(1).pow(2).sum(1) / bb.L)
    check("||encode(x)||^2/L", bool(((r - 1).abs() < 1e-4).all()), [round(t, 6) for t in r.tolist()], "1 +- 1e-4")
    check("our f == model g.f (exact)", torch.equal(v, bb.g.f(z)), float((v - bb.g.f(z)).abs().max()), "0.0")
    gap = token_manifold_gap(v)
    check("data latents on token manifold", bool((gap["dist_frac"] < 1e-4).all()), [f"{t:.1e}" for t in gap["dist_frac"].tolist()], "< 1e-4")

    with torch.no_grad():
        xr_true = bb.decode(v, 0.0, cls)
        xr_null = bb.decode(v, 0.0, None)
    lp = HalfLPIPS(dev)
    ones = torch.ones_like(x[:, :1])
    for tag, xr in (("true", xr_true), ("null", xr_null)):
        p = psnr_masked(xr.clamp(-1, 1), x, ones)
        l = lp.full(xr, x)
        check(f"reconstruction@0deg ({tag} class)", bool((p > 18).all()), {"psnr": [round(t, 2) for t in p.tolist()], "lpips": [round(t, 3) for t in l.tolist()]}, "psnr > 18 dB, visually identical")
    save_grid([x, xr_true, xr_null], run.path("recon_x_true_null.png"))

    e_hat = bb.random_latent(B, dev, g)
    a85 = angle_deg(v, bb.rotate(v, e_hat, 85.0))
    check("angle(v, rotate(v,e,85))", bool(((a85 - 85).abs() < 0.3).all()), [round(t, 3) for t in a85.tolist()], "85 +- 0.3")
    a90 = angle_deg(v, bb.random_latent(B, dev, g))
    sd = math.degrees(1 / math.sqrt(bb.L))
    check("angle(v, random latent)", bool(((a90 - 90).abs() < 4 * sd).all()), [round(t, 3) for t in a90.tolist()], f"90 +- {sd:.3f} (pass within 4 sd)")

    with torch.no_grad():
        d0, d85 = bb.decode(v, 0.0, cls), bb.decode(v, 85.0, cls)
        samp = bb.decode(bb.random_latent(B, dev, g), 85.0, cls)
    diff = (d0 - d85).abs().mean().item()
    check("decode(z,0) != decode(z,85)", diff > 1e-2, round(diff, 4), "> 0.01 mean abs")
    rng_out = (float(samp.min()), float(samp.max()))
    check("decoder output unclamped [-1,1]-ish", rng_out[0] > -1.6 and rng_out[1] < 1.6, [round(t, 3) for t in rng_out], "roughly [-1,1], not clamped")
    save_grid(samp, run.path("one_step_samples_85deg.png"), nrow=B)

    xx = x.clone().requires_grad_(True)
    bb.encode(xx).sum().backward()
    gn = xx.grad.abs().sum().item()
    check("input gradient of encode(x).sum()", gn > 0, f"{gn:.3e}", "> 0")

    with torch.no_grad():
        with bb.differentiable_attention():
            d0m = bb.decode(v, 0.0, cls)
            zm = bb.encode_raw(x)
    check("decoder manual vs sdpa_fp32", float((d0m - d0).abs().max()) < 1e-4, f"{float((d0m - d0).abs().max()):.2e}", "< 1e-4 max abs")
    check("encoder eager vs sdpa", float((zm - z).abs().max()) < 1e-3, f"{float((zm - z).abs().max()):.2e}", "< 1e-3 max abs")

    mA = half_mask(1, bb.image_size, bb.image_size, device=dev)
    t0 = time.time()
    xi = None
    Js = {}
    for mode in ("dvjp", "fwd", "fd"):
        ops = DecoderOps(bb, v[:1], mA, 0.0, cls[:1], jvp_mode=mode)
        if xi is None:
            xi = ops.P(torch.randn(v[:1].shape, generator=g).to(dev))
        Js[mode] = ops.J(xi).detach()
    rel = lambda a, b: float((a - b).norm() / b.norm())
    check("J dvjp vs fwd", rel(Js["dvjp"], Js["fwd"]) < 1e-4, f"{rel(Js['dvjp'], Js['fwd']):.2e}", "< 1e-4 relative")
    check("J fd vs fwd", rel(Js["fd"], Js["fwd"]) < 1e-2, f"{rel(Js['fd'], Js['fwd']):.2e}", "< 1e-2 relative")
    logger.info(f"jvp checks took {time.time() - t0:.1f}s")

    if args.equivalence:
        equivalence_gate(run)
    finish(run)


def finish(run):
    json.dump(results, open(run.path("check.json"), "w"), indent=2)
    n_fail = sum(not r_["pass"] for r_ in results)
    run.finish("ok" if n_fail == 0 else f"{n_fail} checks failed", n_checks=len(results), n_fail=n_fail)
    logger.info(f"{len(results) - n_fail}/{len(results)} checks passed; run dir {run.dir}")
    raise SystemExit(1 if n_fail else 0)


def equivalence_gate(run):
    from PIL import Image
    import numpy as np

    dev = "cuda"
    seed = 99

    def noise_and_label():
        torch.manual_seed(seed)
        with torch.random.fork_rng(devices=[torch.device(dev)]):
            torch.manual_seed(rng.fold_in(seed, 0, 0))
            e = torch.randn(1, 256, 384).to(dev)
        return e

    for prec in ("fp32", "bf16"):
        wdt = torch.float32 if prec == "fp32" else torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = prec == "bf16"
        torch.backends.cudnn.allow_tf32 = prec == "bf16"
        bb = SE2Backbone.from_checkpoint(args.equiv_job_dir, dev_dir=args.dev_dir, device=dev, compute_dtype=wdt, weights_dtype=wdt)
        if prec == "bf16":
            bb.g.to(memory_format=torch.channels_last)
        y = torch.tensor([17], device=dev)
        e = noise_and_label()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
            torch.manual_seed(0)
            g = bb.g
            g.cached_noise = None
            g.cached_noise = g.f(e)
            alpha = g._dec_angle(1, 84.0, dev)
            xm = g.decoder(g.f(e), y, alpha=alpha)
            for _ in range(3):
                zz = g.encoder(xm, y)
                zz, alpha = g.spherify(zz, 84.0, cache_noise=True)
                xm = g.decoder(zz, y, alpha=alpha)
        with torch.no_grad():
            e_hat = f(e)
            xo = bb.decode(e_hat, 84.0, y)
            for _ in range(3):
                xo = bb.decode(bb.rotate(bb.encode(xo, y), e_hat, 84.0), 84.0, y)
        d = float((xo - xm.float()).abs().max())
        if prec == "fp32":
            check("equivalence fp32: adapter == G1 code path", d == 0.0, f"{d:.2e}", "0.0 (identical)")
        else:
            check("equivalence bf16: adapter vs G1 code path", d == 0.0, f"{d:.2e}", "0.0 (identical)")
            if osp.isfile(args.equiv_png):
                import io

                from torchvision.utils import make_grid, save_image

                ref = np.asarray(Image.open(args.equiv_png).convert("RGB")).astype(int)
                for tag, xx in (("G1 code path", xm), ("adapter", xo)):
                    img = torch.clamp(xx.to(torch.bfloat16) * 0.5 + 0.5, 0, 1).cpu()
                    buf = io.BytesIO()
                    save_image(make_grid(img, nrow=1, padding=max(0, (int(img.shape[-1] / 32) - 2) // 2), pad_value=1), buf, format="png")
                    buf.seek(0)
                    diff = np.abs(np.asarray(Image.open(buf).convert("RGB")).astype(int) - ref)
                    check(f"bf16 {tag} vs saved sample.py PNG", diff.max() == 0, {"max": int(diff.max()), "mean": round(float(diff.mean()), 4)}, "0 (pixel-identical)")
            save_grid([xm.float(), xo], run.path("equivalence_bf16_g1_vs_adapter.png"))
        del bb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
