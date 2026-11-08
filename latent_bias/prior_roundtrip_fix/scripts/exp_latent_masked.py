"""
Tier 1 - training-free LATENT-space masked edit.

The latent is a 16x16x384 token grid (global-spherified). We downsample the
rembg foreground mask to the token grid and, at every step, PIN the background
tokens to the original content latent while only the foreground tokens rotate
toward class B. Blend is re-spherified so it stays on the global sphere. Unlike
pixel compositing, the decoder jointly harmonizes the blend.

Compares, one-way A->B, background-PSNR (scene kept) and foreground-PSNR (subject
changed) for: uncontrolled | pixel-composite | latent-masked. Then runs the full
A->B->A round trip with latent masking and reports debiased content-cos.
"""
import sys, os, os.path as osp, math, json
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, torchvision, torch.distributed as dist
from PIL import Image
import exp_common as C

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
IN = "workspace/roundtrip/inputs"; MSK = "workspace/latent_bias/masks"
OUT = "workspace/latent_bias/tier1"; os.makedirs(f"{OUT}/img", exist_ok=True)
STEPS, ANG, GRID = 6, 85.0, 16
EX = [
    ("03_lion.jpg",            294, "lion",    923, "zebra",    "lion_mask.png"),
    ("00_golden_retriever.jpg",535, "golden",  384, "tabbycat", "golden_mask.png"),
    ("04_giant_panda.jpg",     976, "panda",   792, "flamingo", "panda_mask.png"),
    ("06_king_penguin.jpg",    926, "penguin", 585, "redfox",   "penguin_mask.png"),
    ("09_peacock.jpg",         761, "peacock", 909, "husky",    "peacock_mask.png"),
]

dev = C.init_dist(port="29580")
m, cfg, ck = C.load_model(EXP, dev)
NUMCLS = cfg["num_classes"]

Zb = C.unit(C.encode_paths(m, [a["path"] for a in json.load(open("workspace/latent_bias/items.json"))], dev))
bias = Zb.mean(0); bias = bias / bias.norm()

@torch.no_grad()
def enc_unit(x_pm1):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(x_pm1.to(dev, torch.bfloat16))).float().flatten(1).cpu())[0]
def res_cos(a, b):
    ra = a-(a@bias)*bias; rb = b-(b@bias)*bias
    return F.cosine_similarity(ra, rb, dim=0).item()
def masked_psnr(a, b, w):   # w in [0,1] weight map [1,1,H,W], a,b in [0,1]
    e = (((a-b)**2)*w).sum().item() / (w.sum().item()*3 + 1e-9)
    return 99.0 if e < 1e-12 else 10*math.log10(1/e)

@torch.no_grad()
def latent_masked_edit(x_pm1, mtok, y, steps, ang, z0=None):
    """edit toward y; background tokens pinned to z0 (original content latent)."""
    m.cached_noise = None
    yt = torch.tensor([y], device=dev, dtype=torch.long)
    ynull = torch.full_like(yt, NUMCLS)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if z0 is None:
            z0 = m.f(m.encoder(x_pm1, ynull))                 # clean scene latent
        z = m.encoder(x_pm1, ynull)
        zr, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
        z = m.f(mtok*zr + (1-mtok)*z0)                        # pin background tokens
        xd = m.decoder(z, yt, alpha=a); x01 = (xd.float()*0.5+0.5).clamp(0, 1); imgs=[x01.cpu()]
        for _ in range(steps-1):
            z = m.encoder((x01*2-1).to(torch.bfloat16), yt)
            zr, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
            z = m.f(mtok*zr + (1-mtok)*z0)
            xd = m.decoder(z, yt, alpha=a); x01 = (xd.float()*0.5+0.5).clamp(0, 1); imgs.append(x01.cpu())
    return x01, imgs, z0

@torch.no_grad()
def plain_edit(x_pm1, y, steps, ang):
    yt = torch.tensor([y], device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m.edit(x_pm1.to(dev, torch.bfloat16), yt, forward_steps=steps,
                     edit_angle=ang, cache_sampling_noise=True)
    return out.float().clamp(0, 1).cpu()

@torch.no_grad()
def pixel_composite_edit(x_pm1, orig01, keep, y, steps, ang):
    m.cached_noise = None
    yt = torch.tensor([y], device=dev, dtype=torch.long)
    ynull = torch.full_like(yt, NUMCLS)
    def comp(x01): return keep*orig01 + (1-keep)*x01
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = m.encoder(x_pm1, ynull); z, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
        xd = m.decoder(z, yt, alpha=a); x01 = comp((xd.float()*0.5+0.5).clamp(0, 1))
        for _ in range(steps-1):
            z = m.encoder((x01*2-1).to(torch.bfloat16), yt); z, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
            xd = m.decoder(z, yt, alpha=a); x01 = comp((xd.float()*0.5+0.5).clamp(0, 1))
    return x01.cpu()

rows = []
for i, (fn, A, la, B, lb, mf) in enumerate(EX, 1):
    x = C.load_img(f"{IN}/{fn}").unsqueeze(0).to(dev)
    orig01 = (x*0.5+0.5).clamp(0, 1).to(dev, torch.bfloat16)
    o = orig01.float().cpu()
    M = torch.from_numpy(np.array(Image.open(f"{MSK}/{mf}").convert("L"), np.float32)/255.0).view(1, 1, 256, 256).to(dev)
    fg = (M > 0.5).float().cpu(); bg = 1 - fg
    mtok = F.adaptive_avg_pool2d(M, GRID).reshape(1, GRID*GRID, 1).to(torch.bfloat16)   # [1,256,1]
    keep = (1.0 - M).to(torch.bfloat16)

    z0 = enc_unit(x)
    # one-way A->B, three methods
    uB   = plain_edit(x, B, STEPS, ANG)
    pB   = pixel_composite_edit(x, orig01, keep, B, STEPS, ANG)
    lB, _, zscene = latent_masked_edit(x, mtok, B, STEPS, ANG)
    lB = lB.cpu()

    def bgfg(img): return masked_psnr(o, img, bg), masked_psnr(o, img, fg)
    u_bg, u_fg = bgfg(uB); p_bg, p_fg = bgfg(pB); l_bg, l_fg = bgfg(lB)

    # full round trip (latent-masked), anchor both passes to original scene z0
    xB, _, _ = latent_masked_edit(x, mtok, B, STEPS, ANG, z0=zscene)
    xA, _, _ = latent_masked_edit((xB*2-1).to(dev, torch.bfloat16), mtok, A, STEPS, ANG, z0=zscene)
    rt_cos = res_cos(z0, enc_unit(xA*2-1))

    # figure: original | mask | uncontrolled | pixel-composite | latent-masked (one-way B)
    mprev = (o*fg + o*bg*0.25)
    cmp = torch.cat([o, mprev, uB, pB, lB])
    cf = f"tier1_cmp_ex{i}_{la}_{lb}.png"
    torchvision.utils.save_image(torchvision.utils.make_grid(cmp, nrow=5, padding=4, pad_value=1.0), f"{OUT}/img/{cf}")

    rec = {"i": i, "A": la, "B": lb, "fg_cov": float(fg.mean()*1),
           "bg_psnr": {"uncontrolled": u_bg, "pixel": p_bg, "latent": l_bg},
           "fg_psnr": {"uncontrolled": u_fg, "pixel": p_fg, "latent": l_fg},
           "rt_content_cos_latent": rt_cos, "cmp": f"img/{cf}"}
    rows.append(rec)
    print(f"ex{i} {la:8s}->{lb:9s} fg{rec['fg_cov']*100:3.0f}%  "
          f"BG-PSNR unc={u_bg:4.1f} pix={p_bg:4.1f} lat={l_bg:4.1f} | "
          f"FG-PSNR unc={u_fg:4.1f} pix={p_fg:4.1f} lat={l_fg:4.1f} | RT-cos(lat)={rt_cos:+.3f}")

json.dump({"steps": STEPS, "angle": ANG, "grid": GRID, "examples": rows}, open(f"{OUT}/tier1.json", "w"), indent=2)
def mbg(k): return np.mean([r["bg_psnr"][k] for r in rows])
def mfg(k): return np.mean([r["fg_psnr"][k] for r in rows])
print(f"\nMEAN BG-PSNR (scene kept):  uncontrolled {mbg('uncontrolled'):.1f}  pixel {mbg('pixel'):.1f}  latent {mbg('latent'):.1f} dB")
print(f"MEAN FG-PSNR (subject moved): uncontrolled {mfg('uncontrolled'):.1f}  pixel {mfg('pixel'):.1f}  latent {mfg('latent'):.1f} dB (lower=more change)")
print(f"MEAN round-trip content-cos (latent-masked): {np.mean([r['rt_content_cos_latent'] for r in rows]):+.3f}")
dist.destroy_process_group()
