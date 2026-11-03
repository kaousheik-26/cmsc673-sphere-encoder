"""
Latent-bias experiment suite (base imagenet-256 model), one model load:
  #1 per-class: are there per-class biases or one global bias?
  #2 OOD: do flowers/objects/gray/noise land in the SAME cone? (bias intrinsic?)
  #3 structure: is the bias a per-channel DC shared across tokens, or spatial?
  #6 cone vs angle: how far does injected noise push a latent out of the cone?
  #7 real vs generated: manifold gap along the bias direction
  #8 debias / seed-from-bias decodes
"""
import sys, json, os, math, os.path as osp
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import numpy as np, torch, torchvision, torch.distributed as dist
import exp_common as C

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
ANIMALS = "workspace/latent_bias/items.json"
OOD = "workspace/latent_bias/ood_items.json"
OUT = "workspace/latent_bias/exp"
CLASSES = [535, 384, 923, 294, 976, 792, 926, 585, 909, 761]  # 10 round-trip classes
CLABEL = ["golden","tabby","zebra","lion","panda","flamingo","penguin","fox","husky","peacock"]
K_GEN = 8  # generations per class

dev = C.init_dist()
m, cfg, ck = C.load_model(EXP, dev)
N, D = m.latent_shape[1], m.latent_shape[2]
os.makedirs(OUT, exist_ok=True)
print(f"[ok] {ck}  latent {N}x{D}={N*D}")

# ---------- encode real animals (define the bias) ----------
animals = json.load(open(ANIMALS))
Z, U = C.encode_paths(m, [a["path"] for a in animals], dev, return_raw=True)
Zu = C.unit(Z)
bias = Zu.mean(0); bias = bias / bias.norm()          # the global bias direction
print(f"[animals] n={len(Zu)}  mean-dir energy={C.mean_dir_energy(Z):.3f}  "
      f"pairwise cos={C.mean_offdiag_cos(Zu):+.3f}  top1={C.top1_gram_frac(Zu):.3f}")

report = {"model": ck, "n_animals": len(Zu),
          "animal_mean_dir_energy": C.mean_dir_energy(Z),
          "animal_pairwise_cos": C.mean_offdiag_cos(Zu),
          "animal_top1_gram": C.top1_gram_frac(Zu)}

# ======================================================================
# #2 OOD: do non-animal / degenerate inputs fall in the same cone?
# ======================================================================
ood = json.load(open(OOD))
Zood = C.unit(C.encode_paths(m, [o["path"] for o in ood], dev))
cos_ood = (Zood @ bias)
print("\n#2 OOD  cos(input, animal-bias-dir):")
e2 = {}
for kind in ["real_nonanimal", "synthetic"]:
    idx = [i for i, o in enumerate(ood) if o["kind"] == kind]
    cs = cos_ood[idx]
    e2[kind] = {"mean": cs.mean().item(), "min": cs.min().item(), "max": cs.max().item()}
    print(f"   {kind:16s} mean={cs.mean():+.3f}  [{cs.min():+.3f},{cs.max():+.3f}]  (animals ~0.84)")
for i, o in enumerate(ood):
    print(f"      {o['title']:22s} cos={cos_ood[i]:+.3f}")
report["exp2_ood_cos_to_bias"] = e2

# ======================================================================
# #3 structure of the bias vector: per-channel DC vs spatial pattern
# ======================================================================
B = bias.view(N, D)                                   # [tokens, channels]
chan_mean = B.mean(0, keepdim=True)                   # [1, D] per-channel DC
token_const = chan_mean.expand(N, D)                  # replicate across tokens
frac_token_const = (token_const.norm()**2 / B.norm()**2).item()
tok_norm = B.norm(dim=1)                              # [N] energy per token position
print(f"\n#3 bias structure: token-constant (per-channel DC) fraction = {frac_token_const*100:.1f}%  "
      f"of bias energy;  per-token norm cv = {(tok_norm.std()/tok_norm.mean()):.3f}")
report["exp3_bias_token_constant_frac"] = frac_token_const
report["exp3_token_norm_cv"] = (tok_norm.std()/tok_norm.mean()).item()

# ======================================================================
# generate class-conditional samples -> Zg (for #1 and #7)
# ======================================================================
@torch.no_grad()
def encode_tensor_batch(x01):  # [B,3,H,W] in [0,1]
    xs = (x01 * 2 - 1).to(dev, torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(xs)).float().flatten(1).cpu())

gen_imgs, Zg, gen_cls = [], [], []
per_class_dir = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for c in CLASSES:
        y = torch.full((K_GEN,), c, device=dev, dtype=torch.long)
        _, x = m.generate(K_GEN, y=y, forward_steps=4, sampling_init_angle=84.0, device=dev)
        zc = encode_tensor_batch(x.float().cpu())
        Zg.append(zc); gen_cls += [c]*K_GEN; gen_imgs.append(x[0].float().cpu())
        per_class_dir.append(C.unit(zc.mean(0, keepdim=True))[0])
Zg = torch.cat(Zg)
per_class_dir = torch.stack(per_class_dir)            # [n_cls, ND] unit class-mean dirs

# ======================================================================
# #1 per-class: between-class vs within-class vs to-global-bias
# ======================================================================
Gc = per_class_dir @ per_class_dir.t()
btw = (Gc.sum() - Gc.diag().sum()) / (len(CLASSES)*(len(CLASSES)-1))
cos_cls_bias = (per_class_dir @ bias)
# within-class coherence
within = []
for i, c in enumerate(CLASSES):
    zc = Zg[[j for j in range(len(gen_cls)) if gen_cls[j]==c]]
    within.append(C.mean_offdiag_cos(C.unit(zc)))
print(f"\n#1 per-class (from {K_GEN} gens/class): between-class cos={btw:+.3f}  "
      f"within-class cos={np.mean(within):+.3f}  cos(class-mean, global bias)={cos_cls_bias.mean():+.3f}")
for i, lab in enumerate(CLABEL):
    print(f"      {lab:10s} cos(to bias)={cos_cls_bias[i]:+.3f}")
report["exp1_between_class_cos"] = btw.item()
report["exp1_within_class_cos"] = float(np.mean(within))
report["exp1_classmean_to_bias_cos"] = cos_cls_bias.mean().item()

# ======================================================================
# #7 real vs generated: anisotropy + projection offset on the bias axis
# ======================================================================
proj_real = (Zu @ bias); proj_gen = (Zg @ bias)
gen_bias = C.unit(Zg).mean(0); gen_bias = gen_bias/gen_bias.norm()
print(f"\n#7 real vs generated:")
print(f"   mean-dir energy  real={C.mean_dir_energy(Z):.3f}  gen={C.mean_dir_energy(Zg):.3f}")
print(f"   proj on bias     real={proj_real.mean():+.3f}  gen={proj_gen.mean():+.3f}  (gap {proj_gen.mean()-proj_real.mean():+.3f})")
print(f"   cos(real-bias, gen-bias) = {(bias@gen_bias):+.3f}")
report["exp7"] = {"real_mean_dir_energy": C.mean_dir_energy(Z), "gen_mean_dir_energy": C.mean_dir_energy(Zg),
                  "proj_real": proj_real.mean().item(), "proj_gen": proj_gen.mean().item(),
                  "cos_real_gen_bias": (bias@gen_bias).item()}

# ======================================================================
# #6 cone vs injected-noise angle: spherify(z, angle) then cos to bias
# ======================================================================
print(f"\n#6 cone vs injected angle (cos to bias as noise is mixed in):")
e6 = {}
torch.manual_seed(1)
Zf = Z.to(dev)                                        # spherified real latents (not unit-normalized flat)
for ang in [0, 20, 40, 60, 85]:
    a = np.deg2rad(ang)
    cs = []
    for i in range(Zf.shape[0]):
        z = Zf[i:i+1]
        e = m.f(torch.randn(1, N, D, device=dev)).flatten(1)
        za = math.cos(a)*C.unit(z) + math.sin(a)*C.unit(e)
        cs.append(torch.nn.functional.cosine_similarity(C.unit(za), bias.to(dev).unsqueeze(0)).item())
    e6[ang] = float(np.mean(cs))
    print(f"   angle={ang:2d}deg  cos(z_noised, bias)={np.mean(cs):+.3f}")
report["exp6_cos_to_bias_by_angle"] = e6
report["exp6_cone_half_angle_deg"] = math.degrees(math.acos(min(1.0, report["animal_pairwise_cos"]**0.5)))

# ======================================================================
# #8 debias / seed-from-bias decodes (visual)
# ======================================================================
@torch.no_grad()
def decode(vflat, y=None):
    z = vflat.view(1, N, D).to(dev, torch.bfloat16)
    yy = torch.full((1,), cfg["num_classes"], device=dev, dtype=torch.long) if y is None \
         else torch.tensor([y], device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = m.decoder(m.f(z), yy, alpha=m._dec_angle(1, 0.0, dev))
    return (x.float()*0.5+0.5).clamp(0, 1).cpu()

# (a) bias-scale sweep on 4 real latents: z - gamma*(z.bias)bias
rows8a = []
for i in range(4):
    z = Zu[i]; p = (z @ bias)
    cols = [decode(z - g*p*bias) for g in [0.0, 1.0, 1.5]]  # gamma=0 keep, 1 remove, 1.5 over-remove
    rows8a.append(torch.cat(cols))
torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat(rows8a), nrow=3, padding=4, pad_value=1),
                             osp.join(OUT, "exp8_bias_scale_sweep.png"))
# (b) seed-from-bias vs pure-noise generation, class-conditioned
rows8b = []
torch.manual_seed(7)
for c, lab in [(535,"golden"), (923,"zebra")]:
    cols = []
    for a in [90, 60, 40]:  # 90deg = pure noise seed, smaller = closer to the bias
        ar = np.deg2rad(a); e = m.f(torch.randn(1, N, D, device=dev)).flatten(1).cpu()
        seed = math.cos(ar)*bias + math.sin(ar)*C.unit(e)[0]
        cols.append(decode(seed, y=c))
    rows8b.append(torch.cat(cols))
torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat(rows8b), nrow=3, padding=4, pad_value=1),
                             osp.join(OUT, "exp8_seed_from_bias.png"))
print(f"\n#8 decodes saved: exp8_bias_scale_sweep.png (cols: keep|remove-bias|over-remove), "
      f"exp8_seed_from_bias.png (cols: noise-seed|mid|near-bias; rows golden,zebra)")

json.dump(report, open(osp.join(OUT, "geometry_report.json"), "w"), indent=2)
print(f"\nreport -> {osp.join(OUT,'geometry_report.json')}")
dist.destroy_process_group()
