"""
roundtrip_prior.py  --  "double diffusion" prior-consistency probe for sphere2 (G1).

Question: given an image of class A, edit it to a *different* class B (one trip
through the diffusion/edit process), then edit that new image back to class A.
Does the prior (the class-independent content latent produced by the frozen
encoder) survive the round trip? i.e. does A -> B -> A return to the same prior?

For each input image x (true class A), with a chosen target class B:
    z0    = enc(x)                        # the prior of the original
    x_B   = edit(x,   y=B)                # one trip: content of x rendered as B
    x_ABA = edit(x_B, y=A)               # trip back: content of x_B rendered as A
    x_AA  = edit(x,   y=A)               # reference: single direct edit to A

We compare priors by cosine similarity in the encoder's latent space:
    cos(z0, enc(x_B))   -- how far class-swap B pushed the prior (one way)
    cos(z0, enc(x_ABA)) -- MAIN: prior after the full A->B->A round trip
    cos(z0, enc(x_AA))  -- reference: prior after a single edit back to A
and by pixel fidelity to the original (PSNR / L1) for x_ABA and x_AA.

Outputs under <out_dir>:
    per-image grids   row_<i>_<label>.png   [ input | ->B | ->A (via B) | ->A direct ]
    montage           montage_steps=<S>_ang=<ang>.png
    metrics           metrics_steps=<S>_ang=<ang>.json  (+ a printed summary table)
"""
import argparse, json, os, os.path as osp, math, datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torchvision import datasets

from sphere.loader import center_crop_arr
from sphere.builder import build_model
from sphere.ema import ModuleEMA
from sphere.utils import load_ckpt


def patch_dinov3_offline(hidden_act="silu"):
    """
    The encoder is a *frozen* DINOv3 ViT-S/16+ whose weights are already inside
    the sphere2 checkpoint, but the repo instantiates it with
    AutoModel.from_pretrained("facebook/dinov3-vits16plus...") which is gated on
    HF (no token here). We only need the architecture: build DINOv3ViTModel from
    a local config (inferred from the checkpoint's tensor shapes) and let
    load_ckpt fill the weights. Patch AutoModel.from_pretrained in sphere.encoder
    to return that model for any dinov3 name.
    """
    import sphere.encoder as enc_mod
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    orig = enc_mod.AutoModel.from_pretrained

    def from_pretrained(name, *a, **k):
        if "dinov3" in str(name).lower():
            cfg = DINOv3ViTConfig(
                patch_size=16, hidden_size=384, intermediate_size=1536,
                num_hidden_layers=12, num_attention_heads=6,
                num_register_tokens=4, use_gated_mlp=True,
                hidden_act=hidden_act, image_size=256,
                query_bias=True, key_bias=False, value_bias=True,
                proj_bias=True, mlp_bias=True,
            )
            return DINOv3ViTModel(cfg)
        return orig(name, *a, **k)

    enc_mod.AutoModel.from_pretrained = staticmethod(from_pretrained)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True, help="experiment dir with cfg.json + ckpt/")
    p.add_argument("--items", required=True, help="items.json: [{path, class_id, label}]")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--forward_steps", type=int, default=4)
    p.add_argument("--edit_angle", type=float, default=85.0)
    p.add_argument("--use_ema_model", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--dino_act", default="silu", help="gated-MLP activation for the DINOv3 encoder")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--sanity_only", action="store_true", help="just reconstruct inputs and report PSNR, to validate the encoder")
    return p.parse_args()


def load_image_to_tensor(path, image_size):
    x = datasets.folder.pil_loader(path)
    x = np.array(center_crop_arr(x, image_size))
    x = x.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    return x * 2 - 1  # [-1, 1]


def cos(a, b):
    a = a.float().flatten(1)
    b = b.float().flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()


def psnr(a, b):
    # a, b in [0,1]
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)


def main():
    args = parse()

    # ---- single-process dist group (load_ckpt / helpers call dist.barrier) ----
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device,
                            timeout=datetime.timedelta(hours=1))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    # ---- config ----
    with open(osp.join(args.exp_dir, "cfg.json")) as f:
        cfg = json.load(f)
    cfg_ns = SimpleNamespace(**cfg)
    image_size = cfg["image_size"]
    num_classes = cfg["num_classes"]

    # ---- model (DINOv3 encoder built offline from local config) ----
    patch_dinov3_offline(hidden_act=args.dino_act)
    model = build_model(cfg_ns, for_train=False)
    model.to(dtype=ptdtype, device=device, memory_format=torch.channels_last)
    ema_model = ModuleEMA(model)
    ema_model.eval().requires_grad_(False)

    import glob
    ckpts = sorted(glob.glob(osp.join(args.exp_dir, "ckpt", "*.pth")))
    assert ckpts, f"no ckpt in {args.exp_dir}/ckpt"
    load_ckpt(model, ckpt_path=ckpts[-1], ema_model=ema_model, strict=False,
              override_model_with_ema=args.use_ema_model, verbose=False)
    model.eval().requires_grad_(False)
    print(f"[ok] model loaded from {ckpts[-1]}  (ema={args.use_ema_model})")

    # ---- items; pick B = class of the *next* item (a different animal) ----
    items = json.load(open(args.items))
    # resolve each image path: absolute "path", else "input" relative to items.json
    items_root = osp.dirname(osp.abspath(args.items))
    for it in items:
        it["path"] = it.get("path") or osp.join(items_root, it["input"])
    classesA = [it["class_id"] for it in items]
    labelsA = [it["label"] for it in items]
    n = len(items)
    classesB = [classesA[(i + 1) % n] for i in range(n)]
    labelsB = [labelsA[(i + 1) % n] for i in range(n)]

    os.makedirs(args.out_dir, exist_ok=True)
    S, ANG = args.forward_steps, args.edit_angle

    # ---- sanity check: encode->decode reconstruction should resemble the input
    # (validates the DINOv3 activation choice / that weights loaded correctly) ----
    if args.sanity_only:
        print(f"[sanity] dino_act={args.dino_act}")
        for it in items:
            x = load_image_to_tensor(it["path"], image_size).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=ptdtype):
                xr = model.reconstruct(x.to(ptdtype)).float().clamp(0, 1)  # [0,1]
            x01 = (x * 0.5 + 0.5).clamp(0, 1)
            print(f"  {it['label']:16s} recon PSNR = {psnr(x01, xr):.2f} dB")
        dist.destroy_process_group()
        return

    @torch.no_grad()
    def enc(x_pm1):  # x in [-1,1] -> prior latent
        with torch.autocast("cuda", dtype=ptdtype):
            return model.encoder(x_pm1.to(device=device, dtype=ptdtype))

    @torch.no_grad()
    def edit(x_pm1, cls):
        y = torch.tensor([cls], device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=ptdtype):
            out = model.edit(x_pm1.to(device=device, dtype=ptdtype), y,
                             forward_steps=S, edit_angle=ANG,
                             cache_sampling_noise=True)
        return out.float().clamp(0, 1)  # [1,3,H,W] in [0,1]

    rows, records = [], []
    for i, it in enumerate(items):
        A, B = classesA[i], classesB[i]
        x = load_image_to_tensor(it["path"], image_size).unsqueeze(0).to(device)  # [-1,1]

        z0 = enc(x)
        x_B = edit(x, B)                       # A -> B
        x_ABA = edit(x_B * 2 - 1, A)           # B -> A  (round trip)
        x_AA = edit(x, A)                      # direct A (reference)

        zB = enc(x_B * 2 - 1)
        zABA = enc(x_ABA * 2 - 1)
        zAA = enc(x_AA * 2 - 1)

        x01 = (x * 0.5 + 0.5).clamp(0, 1)
        rec = {
            "idx": i, "label_A": labelsA[i], "A": A, "label_B": labelsB[i], "B": B,
            "cos_z0_zB_oneway": cos(z0, zB),
            "cos_z0_zABA_roundtrip": cos(z0, zABA),
            "cos_z0_zAA_direct": cos(z0, zAA),
            "cos_zB_zABA": cos(zB, zABA),
            "psnr_x_ABA": psnr(x01, x_ABA), "psnr_x_AA": psnr(x01, x_AA),
            "l1_x_ABA": torch.mean((x01 - x_ABA).abs()).item(),
            "l1_x_AA": torch.mean((x01 - x_AA).abs()).item(),
        }
        records.append(rec)
        print(f"[{i:2d}] {labelsA[i]:16s} -> {labelsB[i]:16s} -> back | "
              f"cos(z0,zB)={rec['cos_z0_zB_oneway']:.3f}  "
              f"cos(z0,zABA)={rec['cos_z0_zABA_roundtrip']:.3f}  "
              f"cos(z0,zAA)={rec['cos_z0_zAA_direct']:.3f}  "
              f"psnr(ABA)={rec['psnr_x_ABA']:.1f}dB")

        row = torch.cat([x01, x_B, x_ABA, x_AA], dim=0).cpu()  # [4,3,H,W]
        rows.append(row)
        grid = torchvision.utils.make_grid(row, nrow=4, padding=4, pad_value=1.0)
        torchvision.utils.save_image(
            grid, osp.join(args.out_dir, f"row_{i:02d}_{labelsA[i]}_to_{labelsB[i]}.png"))

    # montage: one image per row, 4 columns
    montage = torch.cat(rows, dim=0)
    grid = torchvision.utils.make_grid(montage, nrow=4, padding=4, pad_value=1.0)
    mpath = osp.join(args.out_dir, f"montage_steps={S}_ang={ANG}.png")
    torchvision.utils.save_image(grid, mpath)

    def mean(k):
        return float(np.mean([r[k] for r in records]))

    summary = {k: mean(k) for k in [
        "cos_z0_zB_oneway", "cos_z0_zABA_roundtrip", "cos_z0_zAA_direct",
        "cos_zB_zABA", "psnr_x_ABA", "psnr_x_AA", "l1_x_ABA", "l1_x_AA"]}
    out = {"forward_steps": S, "edit_angle": ANG, "n": n,
           "columns": "input | ->B | ->A(via B, round trip) | ->A(direct)",
           "summary": summary, "per_image": records}
    jpath = osp.join(args.out_dir, f"metrics_steps={S}_ang={ANG}.json")
    json.dump(out, open(jpath, "w"), indent=2)

    print("\n==== SUMMARY (means over", n, "images) ====")
    print(f"  prior cos(z0, z_B)   one-way A->B            : {summary['cos_z0_zB_oneway']:.4f}")
    print(f"  prior cos(z0, z_ABA) round trip A->B->A      : {summary['cos_z0_zABA_roundtrip']:.4f}  <-- MAIN")
    print(f"  prior cos(z0, z_AA)  single direct edit ->A  : {summary['cos_z0_zAA_direct']:.4f}  (reference)")
    print(f"  pixel PSNR(x, x_ABA) round trip              : {summary['psnr_x_ABA']:.2f} dB")
    print(f"  pixel PSNR(x, x_AA)  direct                  : {summary['psnr_x_AA']:.2f} dB")
    print(f"  montage: {mpath}")
    print(f"  metrics: {jpath}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
