"""
#5 re-score the A->B->A round trip with the bias removed.

Raw cosine in this space is inflated by the shared bias direction. We recompute
the prior-preservation cosines using the RESIDUAL (bias projected out), so the
comparison reflects content, not the shared pedestal.

bias is estimated from the 49-animal set (global bias direction).
"""
import sys, json, os.path as osp
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import numpy as np, torch, torch.distributed as dist
import exp_common as C

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
RT_ITEMS = "roundtrip_prior/items.json"           # the 10 labeled round-trip images
BIAS_SET = "workspace/latent_bias/items.json"      # 49 animals -> bias direction

dev = C.init_dist(port="29573")
m, cfg, ck = C.load_model(EXP, dev)
N, D = m.latent_shape[1], m.latent_shape[2]

# global bias direction from the 49-animal set
Zb = C.unit(C.encode_paths(m, [a["path"] for a in json.load(open(BIAS_SET))], dev))
bias = Zb.mean(0); bias = bias / bias.norm()

items = json.load(open(RT_ITEMS))
root = osp.dirname(osp.abspath(RT_ITEMS))
for it in items:
    it["path"] = it.get("path") or osp.join(root, it["input"])
A = [it["class_id"] for it in items]
B = [A[(i+1) % len(A)] for i in range(len(A))]

def raw_cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
def res_cos(a, b):   # bias removed
    ra = a - (a @ bias) * bias; rb = b - (b @ bias) * bias
    return torch.nn.functional.cosine_similarity(ra, rb, dim=0).item()

@torch.no_grad()
def enc(x_pm1):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(x_pm1.to(dev, torch.bfloat16))).float().flatten(1).cpu())[0]
@torch.no_grad()
def edit(x_pm1, c, ang):
    y = torch.tensor([c], device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m.edit(x_pm1.to(dev, torch.bfloat16), y, forward_steps=4,
                      edit_angle=ang, cache_sampling_noise=True).float().clamp(0, 1)

for ANG in [85, 40]:
    rows = {"raw_zB": [], "raw_zABA": [], "raw_zAA": [], "res_zB": [], "res_zABA": [], "res_zAA": []}
    for i, it in enumerate(items):
        x = C.load_img(it["path"]).unsqueeze(0).to(dev)
        z0 = enc(x)
        xB = edit(x, B[i], ANG); zB = enc(xB*2-1)
        xABA = edit(xB*2-1, A[i], ANG); zABA = enc(xABA*2-1)
        xAA = edit(x, A[i], ANG); zAA = enc(xAA*2-1)
        rows["raw_zB"].append(raw_cos(z0, zB));   rows["res_zB"].append(res_cos(z0, zB))
        rows["raw_zABA"].append(raw_cos(z0, zABA)); rows["res_zABA"].append(res_cos(z0, zABA))
        rows["raw_zAA"].append(raw_cos(z0, zAA));  rows["res_zAA"].append(res_cos(z0, zAA))
    mu = {k: float(np.mean(v)) for k, v in rows.items()}
    print(f"\n#5 round trip @ edit_angle={ANG}  (means over {len(items)} images)")
    print(f"   {'':14s}   raw-cos   residual-cos (bias removed)")
    print(f"   one-way  A->B   {mu['raw_zB']:+.3f}     {mu['res_zB']:+.3f}")
    print(f"   round    A->B->A{mu['raw_zABA']:+.3f}     {mu['res_zABA']:+.3f}")
    print(f"   direct   ->A    {mu['raw_zAA']:+.3f}     {mu['res_zAA']:+.3f}")

dist.destroy_process_group()
