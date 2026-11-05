"""5 step-by-step A->B->A round trips across different animal classes."""
import sys, os, os.path as osp, math, json
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import numpy as np, torch, torchvision, torch.distributed as dist
import exp_common as C

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
IN = "workspace/roundtrip/inputs"
SITE = "workspace/latent_bias/site"
STEPS, ANG = 6, 85.0

# (file, A id, A label, B id, B label)
EX = [
    (f"{IN}/03_lion.jpg",            294, "lion",     923, "zebra"),
    (f"{IN}/00_golden_retriever.jpg",535, "golden",   384, "tabby cat"),
    (f"{IN}/04_giant_panda.jpg",     976, "panda",    792, "flamingo"),
    (f"{IN}/06_king_penguin.jpg",    926, "penguin",  585, "red fox"),
    (f"{IN}/09_peacock.jpg",         761, "peacock",  909, "husky"),
]

dev = C.init_dist(port="29576")
m, cfg, ck = C.load_model(EXP, dev)
N, D = m.latent_shape[1], m.latent_shape[2]
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
def edit_steps(x_pm1, cls):
    y = torch.tensor([cls], device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, steps = m.edit(x_pm1.to(dev, torch.bfloat16), y, forward_steps=STEPS,
                            edit_angle=ANG, cache_sampling_noise=True, return_step_images=True)
    return out.float().clamp(0, 1).cpu(), [s.float().clamp(0, 1).cpu() for s in steps]

gallery = []
for i, (path, A, la, B, lb) in enumerate(EX, 1):
    x = C.load_img(path).unsqueeze(0).to(dev)
    x01 = (x*0.5+0.5).clamp(0, 1).cpu(); z0 = enc_unit(x)
    xB, sf = edit_steps(x, B)
    xA, sb = edit_steps(xB*2-1, A)
    drift = [res_cos(z0, enc_unit(s*2-1)) for s in sf] + [res_cos(z0, enc_unit(s*2-1)) for s in sb]
    fname = f"stepwise_ex{i}_{la}_{lb}.png"
    both = torch.cat([torch.cat([x01]+sf), torch.cat([xB]+sb)])
    torchvision.utils.save_image(
        torchvision.utils.make_grid(both, nrow=1+STEPS, padding=4, pad_value=1.0), f"{SITE}/img/{fname}")
    rec = {"i": i, "A": la, "B": lb, "img": f"img/{fname}", "drift": drift,
           "trough": min(drift[:STEPS]), "final": drift[-1], "psnr": psnr(x01, xA)}
    gallery.append(rec)
    print(f"ex{i} {la:8s}->{lb:10s}->{la:8s}  trough={rec['trough']:+.3f}  final={rec['final']:+.3f}  psnr={rec['psnr']:.1f}dB")

json.dump({"steps": STEPS, "angle": ANG, "examples": gallery},
          open(f"{SITE}/stepwise5.json", "w"), indent=2)
print(f"\nsaved 5 strips to {SITE}/img/ and stepwise5.json")
dist.destroy_process_group()
