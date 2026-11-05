"""
Segmentation-controlled round trip.

A foreground mask (rembg / U^2-Net) separates the main figure from the background.
We run A->B->A, but at every diffusion step we composite the ORIGINAL background
back in (anchor='bg'): keep = (1 - mask). The scene is held fixed and only the
masked subject is edited, so the composition can't drift. Compared against the
plain (uncontrolled) round trip.
"""
import sys, os, os.path as osp, math, json
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import numpy as np, torch, torchvision, torch.distributed as dist
from PIL import Image
import exp_common as C

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
IN = "workspace/roundtrip/inputs"; MSK = "workspace/latent_bias/masks"; SITE = "workspace/latent_bias/site"
STEPS, ANG = 6, 85.0
EX = [  # (file, A, Alabel, B, Blabel, maskfile)
    ("03_lion.jpg",            294, "lion",    923, "zebra",    "lion_mask.png"),
    ("00_golden_retriever.jpg",535, "golden",  384, "tabbycat", "golden_mask.png"),
    ("04_giant_panda.jpg",     976, "panda",   792, "flamingo", "panda_mask.png"),
    ("06_king_penguin.jpg",    926, "penguin", 585, "redfox",   "penguin_mask.png"),
    ("09_peacock.jpg",         761, "peacock", 909, "husky",    "peacock_mask.png"),
]

dev = C.init_dist(port="29577")
m, cfg, ck = C.load_model(EXP, dev)
N, D = m.latent_shape[1], m.latent_shape[2]; NUMCLS = cfg["num_classes"]
os.makedirs(f"{SITE}/img", exist_ok=True)

Zb = C.unit(C.encode_paths(m, [a["path"] for a in json.load(open("workspace/latent_bias/items.json"))], dev))
bias = Zb.mean(0); bias = bias / bias.norm()

@torch.no_grad()
def enc_unit(x_pm1):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(x_pm1.to(dev, torch.bfloat16))).float().flatten(1).cpu())[0]
def res_cos(a, b):
    ra = a-(a@bias)*bias; rb = b-(b@bias)*bias
    return torch.nn.functional.cosine_similarity(ra, rb, dim=0).item()
def psnr(a, b):
    e = torch.mean((a-b)**2).item(); return 99.0 if e < 1e-12 else 10*math.log10(1/e)

@torch.no_grad()
def controlled_edit(x_start_pm1, orig01, keep, y, steps, ang):
    """edit toward class y, compositing keep*orig at every step. returns (x01, [steps])."""
    m.cached_noise = None
    yt = torch.tensor([y], device=dev, dtype=torch.long)
    ynull = torch.full_like(yt, NUMCLS)
    def comp(x01): return keep*orig01 + (1-keep)*x01
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = m.encoder(x_start_pm1, ynull); z, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
        x = m.decoder(z, yt, alpha=a); x01 = comp((x.float()*0.5+0.5).clamp(0, 1)); imgs = [x01.cpu()]
        x = (x01*2-1).to(torch.bfloat16)
        for _ in range(steps-1):
            z = m.encoder(x, yt); z, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
            x = m.decoder(z, yt, alpha=a); x01 = comp((x.float()*0.5+0.5).clamp(0, 1)); imgs.append(x01.cpu())
            x = (x01*2-1).to(torch.bfloat16)
    return x01, imgs

@torch.no_grad()
def plain_edit(x_pm1, y, steps, ang):
    yt = torch.tensor([y], device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, st = m.edit(x_pm1.to(dev, torch.bfloat16), yt, forward_steps=steps,
                         edit_angle=ang, cache_sampling_noise=True, return_step_images=True)
    return out.float().clamp(0, 1).cpu(), [s.float().clamp(0, 1).cpu() for s in st]

gallery = []
for i, (fn, A, la, B, lb, mf) in enumerate(EX, 1):
    x = C.load_img(f"{IN}/{fn}").unsqueeze(0).to(dev)
    orig01 = (x*0.5+0.5).clamp(0, 1).to(dev, torch.bfloat16)
    z0 = enc_unit(x)
    M = torch.from_numpy(np.array(Image.open(f"{MSK}/{mf}").convert("L"), np.float32)/255.0)
    M = M.view(1, 1, 256, 256).to(dev, torch.bfloat16)        # foreground matte
    keep = (1.0 - M)                                           # anchor = background

    # controlled round trip (scene locked)
    xB, sB = controlled_edit(x, orig01, keep, B, STEPS, ANG)
    xA, sA = controlled_edit((xB*2-1).to(dev, torch.bfloat16), orig01, keep, A, STEPS, ANG)
    # uncontrolled baseline
    uB, _ = plain_edit(x, B, STEPS, ANG)
    uA, _ = plain_edit((uB*2-1).to(dev), A, STEPS, ANG)

    o = (x*0.5+0.5).clamp(0, 1).cpu()
    ctl_cos, ctl_ps = res_cos(z0, enc_unit(xA*2-1)), psnr(o, xA.cpu())
    unc_cos, unc_ps = res_cos(z0, enc_unit(uA*2-1)), psnr(o, uA.cpu())
    # stepwise strip (controlled): row A->B, row B->A
    both = torch.cat([torch.cat([o]+[s for s in sB]), torch.cat([xB.cpu()]+[s for s in sA])])
    strip = f"ctrl_stepwise_ex{i}_{la}_{lb}.png"
    torchvision.utils.save_image(torchvision.utils.make_grid(both, nrow=1+STEPS, padding=4, pad_value=1.0), f"{SITE}/img/{strip}")
    # comparison: original | mask preview | uncontrolled final | controlled final
    mprev = (o*M.cpu().float()*1 + o*(1-M.cpu().float())*0.25)
    cmp = torch.cat([o, mprev, uA.cpu(), xA.cpu()])
    cmpf = f"ctrl_cmp_ex{i}_{la}_{lb}.png"
    torchvision.utils.save_image(torchvision.utils.make_grid(cmp, nrow=4, padding=4, pad_value=1.0), f"{SITE}/img/{cmpf}")
    gallery.append({"i": i, "A": la, "B": lb, "strip": f"img/{strip}", "cmp": f"img/{cmpf}",
                    "ctl_cos": ctl_cos, "ctl_psnr": ctl_ps, "unc_cos": unc_cos, "unc_psnr": unc_ps,
                    "fg_cov": float((M > 0.5).float().mean().item())})
    print(f"ex{i} {la:8s}->{lb:9s}:  controlled cos={ctl_cos:+.3f} psnr={ctl_ps:4.1f} | "
          f"uncontrolled cos={unc_cos:+.3f} psnr={unc_ps:4.1f}  (fg {gallery[-1]['fg_cov']*100:.0f}%)")

json.dump({"steps": STEPS, "angle": ANG, "examples": gallery}, open(f"{SITE}/controlled.json", "w"), indent=2)
mc = np.mean([g["ctl_cos"] for g in gallery]); mu = np.mean([g["unc_cos"] for g in gallery])
mp, mpu = np.mean([g["ctl_psnr"] for g in gallery]), np.mean([g["unc_psnr"] for g in gallery])
print(f"\nMEAN content-cos: controlled {mc:+.3f} vs uncontrolled {mu:+.3f}")
print(f"MEAN PSNR:        controlled {mp:.1f} dB vs uncontrolled {mpu:.1f} dB")
dist.destroy_process_group()
