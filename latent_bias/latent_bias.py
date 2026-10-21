"""
latent_bias.py -- is there a systematic *bias* (a dominant shared direction /
anisotropy) in the sphere encoder's spherical latent space?

Mechanism under test: the model spherifies with f = vector_rms_norm(zero_mean=False),
so the DC / shared-mean component of enc(x) is NOT removed before projecting onto
the hypersphere. If the encoder's outputs share a common component, every image
latent lands in a narrow cone around one direction -> "the bias".

We encode many real images, look at the latent geometry directly (not by feeding
one image), and compare against the generator's isotropic noise prior f(randn):

  u_i = enc(x_i)            raw encoder output   [N, D]
  z_i = f(u_i)              spherified latent the decoder consumes (unit RMS)
  r_j = f(randn)            the sampling prior (isotropic baseline)

Metrics:
  - DC scalar: mean over all elements of u_i (should be ~0 if unbiased)
  - mean-direction energy: ||mean_i z_i||  (0 for isotropic, ->1 for a cone)
  - mean pairwise cosine among distinct z_i  vs among r_j
  - cos(z_i, bias_dir) where bias_dir = normalized mean latent
  - top-1 Gram eigenvalue fraction (uncentered) for z vs r

We also DECODE the latent space directly:
  - the bias direction  bias_dir  (null class + a few classes)   -> "the bias image"
  - random sphere points r_j (the generator prior)               -> contrast
  - each z_i with the bias projected out, re-spherified           -> does content survive?
"""
import argparse, json, os, os.path as osp, math, glob, datetime
from types import SimpleNamespace
import numpy as np, torch, torch.distributed as dist, torchvision
from torchvision import datasets

from sphere.loader import center_crop_arr
from sphere.builder import build_model
from sphere.ema import ModuleEMA
from sphere.utils import load_ckpt


def patch_dinov3_offline(hidden_act="silu"):
    import sphere.encoder as em
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    orig = em.AutoModel.from_pretrained
    def fp(name, *a, **k):
        if "dinov3" in str(name).lower():
            return DINOv3ViTModel(DINOv3ViTConfig(
                patch_size=16, hidden_size=384, intermediate_size=1536,
                num_hidden_layers=12, num_attention_heads=6, num_register_tokens=4,
                use_gated_mlp=True, hidden_act=hidden_act, image_size=256,
                query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True))
        return orig(name, *a, **k)
    em.AutoModel.from_pretrained = staticmethod(fp)


def load_img(path, size):
    x = datasets.folder.pil_loader(path)
    x = np.array(center_crop_arr(x, size)).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1) * 2 - 1  # [-1,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dino_act", default="silu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for k, v in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                     MASTER_ADDR="127.0.0.1", MASTER_PORT="29531").items():
        os.environ.setdefault(k, v)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(hours=1))
    torch.manual_seed(args.seed)

    cfg = json.load(open(osp.join(args.exp_dir, "cfg.json")))
    image_size, num_classes = cfg["image_size"], cfg["num_classes"]
    patch_dinov3_offline(args.dino_act)
    model = build_model(SimpleNamespace(**cfg), for_train=False)
    model.to(dtype=torch.bfloat16, device=device, memory_format=torch.channels_last)
    ema = ModuleEMA(model); ema.eval().requires_grad_(False)
    ckpt = sorted(glob.glob(osp.join(args.exp_dir, "ckpt", "*.pth")))[-1]
    load_ckpt(model, ckpt_path=ckpt, ema_model=ema, strict=False,
              override_model_with_ema=True, verbose=False)
    model.eval().requires_grad_(False)
    os.makedirs(args.out_dir, exist_ok=True)

    items = json.load(open(args.items))
    # resolve each image path: absolute "path", else "input" relative to items.json
    items_root = osp.dirname(osp.abspath(args.items))
    for it in items:
        it["path"] = it.get("path") or osp.join(items_root, it["input"])
    N_tok, D = model.latent_shape[1], model.latent_shape[2]
    print(f"[ok] {ckpt}  latent_shape={tuple(model.latent_shape)}  ({N_tok*D} dims)")

    # ---- encode all images: raw u and spherified z ----
    U, Z = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for it in items:
            x = load_img(it["path"], image_size).unsqueeze(0).to(device, torch.bfloat16)
            u = model.encoder(x)                 # [1, N, D] raw encoder output
            z = model.f(u)                       # spherified (zero_mean=False)
            U.append(u.float().flatten(1).cpu())
            Z.append(z.float().flatten(1).cpu())
        U = torch.cat(U); Z = torch.cat(Z)       # [M, N*D]
        # isotropic baseline: the generator's own prior f(randn)
        R = []
        for _ in range(len(items)):
            e = model.f(torch.randn(1, N_tok, D, device=device))
            R.append(e.float().flatten(1).cpu())
        R = torch.cat(R)

    M, Ddim = Z.shape
    def unit(v):  # L2-normalize rows
        return v / v.norm(dim=1, keepdim=True)

    Zu, Ru = unit(Z), unit(R)

    # 1) DC scalar of the raw encoder output
    dc = U.mean(dim=1)                                   # mean over all elements
    dc_rel = dc / U.std(dim=1)                           # relative to per-latent spread

    # 2) mean-direction energy (cone): ||mean of unit latents||; 0=isotropic, 1=colinear
    md_Z = Zu.mean(0).norm().item()
    md_R = Ru.mean(0).norm().item()

    # 3) mean pairwise cosine among distinct samples
    def mean_offdiag_cos(Vu):
        G = Vu @ Vu.t()
        m = (G.sum() - G.diag().sum()) / (Vu.shape[0] * (Vu.shape[0] - 1))
        return m.item()
    cos_Z = mean_offdiag_cos(Zu)
    cos_R = mean_offdiag_cos(Ru)
    cos_iso = 1.0 / math.sqrt(Ddim)                      # expected scale if isotropic

    # 4) cosine of each latent to the bias direction (normalized mean)
    bias_dir = Zu.mean(0); bias_dir = bias_dir / bias_dir.norm()
    cos_to_bias = (Zu @ bias_dir)                        # [M]

    # 5) top-1 uncentered Gram eigenvalue fraction (how much variance is 1 direction)
    def top1_frac(Vu):
        G = Vu @ Vu.t()                                  # [M,M], tr(G)=M
        ev = torch.linalg.eigvalsh(G)
        return (ev[-1] / ev.sum()).item()
    t1_Z, t1_R = top1_frac(Zu), top1_frac(Ru)
    # centered (remove the bias direction) -> residual anisotropy
    Zc = Zu - (Zu @ bias_dir).unsqueeze(1) * bias_dir
    t1_Zc = top1_frac(unit(Zc))

    stats = {
        "num_images": M, "latent_dims": Ddim,
        "raw_DC_scalar_mean": dc.mean().item(), "raw_DC_scalar_std": dc.std().item(),
        "raw_DC_rel_to_std_mean": dc_rel.mean().item(),
        "mean_direction_energy_encoded": md_Z,
        "mean_direction_energy_noise": md_R,
        "mean_pairwise_cos_encoded": cos_Z,
        "mean_pairwise_cos_noise": cos_R,
        "isotropic_cos_scale_1_over_sqrtD": cos_iso,
        "cos_to_bias_mean": cos_to_bias.mean().item(),
        "cos_to_bias_min": cos_to_bias.min().item(),
        "cos_to_bias_max": cos_to_bias.max().item(),
        "frac_variance_in_bias_dir_encoded": md_Z**2,   # = ||mean unit||^2
        "top1_gram_frac_encoded": t1_Z,
        "top1_gram_frac_noise": t1_R,
        "top1_gram_frac_encoded_after_removing_bias": t1_Zc,
    }
    json.dump(stats, open(osp.join(args.out_dir, "bias_stats.json"), "w"), indent=2)

    print("\n==== LATENT-SPACE BIAS (", M, "images,", Ddim, "dims ) ====")
    print(f"  raw encoder DC scalar           : mean={stats['raw_DC_scalar_mean']:+.4f} "
          f"(rel to per-latent std {stats['raw_DC_rel_to_std_mean']:+.3f})")
    print(f"  mean-direction energy  encoded  : {md_Z:.3f}   (noise prior {md_R:.3f}; 0=isotropic,1=cone)")
    print(f"  mean pairwise cosine   encoded  : {cos_Z:+.3f}   (noise prior {cos_R:+.3f}; isotropic ~{cos_iso:.4f})")
    print(f"  cos(latent, bias dir)           : mean={cos_to_bias.mean():+.3f}  "
          f"[min {cos_to_bias.min():+.3f}, max {cos_to_bias.max():+.3f}]")
    print(f"  variance in the single bias dir : {md_Z**2*100:.1f}%  of the total")
    print(f"  top-1 Gram frac  encoded/noise  : {t1_Z:.3f} / {t1_R:.3f}   "
          f"(encoded after removing bias: {t1_Zc:.3f})")

    # ---- decode the latent space DIRECTLY (not from an image) ----
    @torch.no_grad()
    def decode_flat(v_flat, y=None):
        z = v_flat.view(1, N_tok, D).to(device, torch.bfloat16)
        if y is None:
            y = torch.full((1,), num_classes, device=device, dtype=torch.long)  # null
        else:
            y = torch.tensor([y], device=device, dtype=torch.long)
        a = model._dec_angle(1, 0.0, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.decoder(model.f(z), y, alpha=a)
        return (x.float() * 0.5 + 0.5).clamp(0, 1).cpu()

    panels = []
    # the bias direction, decoded under the null class and a few classes
    bias_cols = [decode_flat(bias_dir)]                       # null
    for c in [535, 923, 294, 792]:                            # golden, zebra, lion, flamingo
        bias_cols.append(decode_flat(bias_dir, y=c))
    panels.append(("bias_direction_null+golden+zebra+lion+flamingo", torch.cat(bias_cols)))
    # random sphere points (the generator's isotropic prior), null class
    rnd = torch.cat([decode_flat(Ru[j]) for j in range(5)])
    panels.append(("random_sphere_points_null", rnd))
    # a few real latents, and the same with the bias removed (null class)
    keep = torch.cat([decode_flat(Zu[j]) for j in range(5)])
    rem = torch.cat([decode_flat(unit(Zc)[j]) for j in range(5)])
    panels.append(("encoded_latents_null", keep))
    panels.append(("encoded_minus_bias_null", rem))

    for name, t in panels:
        torchvision.utils.save_image(
            torchvision.utils.make_grid(t, nrow=t.shape[0], padding=4, pad_value=1.0),
            osp.join(args.out_dir, f"decode_{name}.png"))
    # one stacked montage (rows = the 4 panels, each 5 wide)
    rows = [p[1][:5] for p in panels]
    mg = torch.cat([r for r in rows])
    torchvision.utils.save_image(
        torchvision.utils.make_grid(mg, nrow=5, padding=4, pad_value=1.0),
        osp.join(args.out_dir, "montage_latent_probe.png"))
    print(f"\n  decoded probes + montage_latent_probe.png in {args.out_dir}")
    print("  rows: [bias dir | random sphere | encoded | encoded-minus-bias]")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
