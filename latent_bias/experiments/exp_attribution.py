"""
#4 attribution: where does the bias come from?
  (a) anisotropy per DINOv3 block (does the cone grow with depth?)
  (b) raw encoder output U vs spherified latent f(U): spherify is radial, so it
      cannot change direction -> same anisotropy (bias is not from spherify)
  (c) base vs fd-lite: the encoder is FROZEN & shared, so the latent bias should
      be identical regardless of the sphere2 training objective
"""
import sys, json, os.path as osp
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import torch, torch.distributed as dist
import exp_common as C

BASE = "workspace/experiments/sphere2-base-imagenet-256px"
FDL  = "workspace/experiments/sphere2-base-imagenet-fd-lite-256px"
animals = json.load(open("workspace/latent_bias/items.json"))
paths = [a["path"] for a in animals]

dev = C.init_dist(port="29572")
m, cfg, ck = C.load_model(BASE, dev)
N, D = m.latent_shape[1], m.latent_shape[2]

# (a) per-block anisotropy + (b) raw U vs spherified Z
LAYERS = [0, 3, 6, 9, 11]
feats = {i: [] for i in LAYERS}
U, Z = [], []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for p in paths:
        x = C.load_img(p).unsqueeze(0).to(dev, torch.bfloat16)
        z, hs = m.encoder(x, extract_layer_idxs=LAYERS)
        for i in LAYERS:
            feats[i].append(hs[i].float().flatten(1).cpu())
        U.append(z.float().flatten(1).cpu())
        Z.append(m.f(z).float().flatten(1).cpu())
U = torch.cat(U); Z = torch.cat(Z)

print("#4a per-DINOv3-block anisotropy (mean pairwise cos / mean-dir energy):")
for i in LAYERS:
    F = torch.cat(feats[i])
    print(f"   block {i:2d}: cos={C.mean_offdiag_cos(C.unit(F)):+.3f}  dir-energy={C.mean_dir_energy(F):.3f}")
print(f"   final U : cos={C.mean_offdiag_cos(C.unit(U)):+.3f}  dir-energy={C.mean_dir_energy(U):.3f}")
print(f"   spherified f(U): cos={C.mean_offdiag_cos(C.unit(Z)):+.3f}  dir-energy={C.mean_dir_energy(Z):.3f}")
print("   (raw U == spherified: spherify is a per-sample radial rescale, directions unchanged)")

# (c) base vs fd-lite: frozen encoder -> identical latents?
mf, cfgf, ckf = C.load_model(FDL, dev, seed=0)
Zf = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for p in paths:
        x = C.load_img(p).unsqueeze(0).to(dev, torch.bfloat16)
        Zf.append(mf.f(mf.encoder(x)).float().flatten(1).cpu())
Zf = torch.cat(Zf)
cos_bf = torch.nn.functional.cosine_similarity(C.unit(Z), C.unit(Zf), dim=1)
bias_b = C.unit(Z).mean(0); bias_b /= bias_b.norm()
bias_f = C.unit(Zf).mean(0); bias_f /= bias_f.norm()
print(f"\n#4c base vs fd-lite (encoder frozen & shared):")
print(f"   per-image cos(base latent, fdlite latent) = {cos_bf.mean():.4f}  [min {cos_bf.min():.4f}]")
print(f"   fd-lite anisotropy: cos={C.mean_offdiag_cos(C.unit(Zf)):+.3f}  dir-energy={C.mean_dir_energy(Zf):.3f} (base {C.mean_dir_energy(Z):.3f})")
print(f"   cos(base bias dir, fdlite bias dir) = {(bias_b@bias_f):.4f}")
dist.destroy_process_group()
