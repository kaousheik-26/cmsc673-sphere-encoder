"""
Step-by-step round trip:  lion --edit--> cat  (show every diffusion step),
then take the GENERATED cat and  --edit--> lion  (show every step), and check
whether the final lion still carries the prior of the lion we started with.
"""
import sys, os, os.path as osp, math, json
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import numpy as np, torch, torchvision, torch.distributed as dist
import exp_common as C

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
LION_IMG = "workspace/roundtrip/inputs/03_lion.jpg"
LION, CAT = 294, 384          # model-space class ids: lion, tabby cat
STEPS, ANG = 6, 85.0
OUT = "workspace/latent_bias/stepwise"

dev = C.init_dist(port="29575")
m, cfg, ck = C.load_model(EXP, dev)
N, D = m.latent_shape[1], m.latent_shape[2]
os.makedirs(OUT, exist_ok=True)

# bias direction (from the 49 animals) for the debiased / content-only cosine
Zb = C.unit(C.encode_paths(m, [a["path"] for a in json.load(open("workspace/latent_bias/items.json"))], dev))
bias = Zb.mean(0); bias = bias / bias.norm()

@torch.no_grad()
def enc_unit(x_pm1):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(x_pm1.to(dev, torch.bfloat16))).float().flatten(1).cpu())[0]
def res_cos(a, b):
    ra = a - (a @ bias) * bias; rb = b - (b @ bias) * bias
    return torch.nn.functional.cosine_similarity(ra, rb, dim=0).item()
def raw_cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
def psnr(a, b):
    e = torch.mean((a - b) ** 2).item(); return 99.0 if e < 1e-12 else 10 * math.log10(1 / e)

# original lion
x = C.load_img(LION_IMG).unsqueeze(0).to(dev)      # [-1,1]
x01 = (x * 0.5 + 0.5).clamp(0, 1).cpu()
z_orig = enc_unit(x)                                # the prior we start from

@torch.no_grad()
def edit_steps(x_pm1, cls):
    y = torch.tensor([cls], device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, steps = m.edit(x_pm1.to(dev, torch.bfloat16), y, forward_steps=STEPS,
                            edit_angle=ANG, cache_sampling_noise=True, return_step_images=True)
    return out.float().clamp(0, 1).cpu(), [s.float().clamp(0, 1).cpu() for s in steps]

# PASS 1: lion -> cat
cat, steps_fwd = edit_steps(x, CAT)
# PASS 2: generated cat -> lion
lion2, steps_bwd = edit_steps(cat * 2 - 1, LION)

# ---- track the prior across all steps (content-only, debiased) ----
print(f"lion({LION}) -> cat({CAT}) -> lion({LION})   edit_angle={ANG}, steps={STEPS}")
print("\nprior vs the ORIGINAL lion, step by step  (residual=content-only cosine):")
def track(tag, imgs):
    for i, im in enumerate(imgs):
        z = enc_unit(im * 2 - 1)
        print(f"  {tag} step {i}:  content-cos={res_cos(z_orig, z):+.3f}   raw-cos={raw_cos(z_orig, z):+.3f}")
track("lion->cat", steps_fwd)
track("cat->lion", steps_bwd)
z_final = enc_unit(lion2 * 2 - 1)
print(f"\nFINAL lion vs original:  content-cos={res_cos(z_orig, z_final):+.3f}   "
      f"raw-cos={raw_cos(z_orig, z_final):+.3f}   pixel PSNR={psnr(x01, lion2):.2f} dB")

# ---- figures ----
def strip(first, steps, path):
    row = torch.cat([first] + steps, dim=0)      # [1+STEPS, 3, H, W]
    torchvision.utils.save_image(
        torchvision.utils.make_grid(row, nrow=1 + len(steps), padding=4, pad_value=1.0), path)

strip(x01, steps_fwd, osp.join(OUT, "pass1_lion_to_cat.png"))      # [orig | s0..s5(=cat)]
strip(cat, steps_bwd, osp.join(OUT, "pass2_cat_to_lion.png"))      # [cat  | s0..s5(=lion)]
# combined 2-row strip
both = torch.cat([torch.cat([x01] + steps_fwd), torch.cat([cat] + steps_bwd)])
torchvision.utils.save_image(
    torchvision.utils.make_grid(both, nrow=1 + STEPS, padding=4, pad_value=1.0),
    osp.join(OUT, "roundtrip_stepwise.png"))
# start vs end
torchvision.utils.save_image(
    torchvision.utils.make_grid(torch.cat([x01, lion2]), nrow=2, padding=4, pad_value=1.0),
    osp.join(OUT, "start_vs_end.png"))
print(f"\nsaved: roundtrip_stepwise.png (row1 lion->cat, row2 cat->lion), "
      f"pass1_*, pass2_*, start_vs_end.png  in {OUT}")
dist.destroy_process_group()
