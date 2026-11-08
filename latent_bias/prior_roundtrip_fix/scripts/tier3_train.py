"""
Tier 3 - improve the A->B->A prior with a CYCLE-CONSISTENCY objective, trained
on only 20 images of ONE class (lion, ImageNet id 294).

Direct objective (what earlier tiers missed): optimize the round trip itself.
  xB   = edit_theta(x, B)          outward to a random class B
  xABA = edit_theta(xB, lion)      return to lion
  L_cycle = L1(xABA, x) + feat(xABA, x)      <- return to the ORIGINAL lion
  L_self  = L1(edit_theta(x, lion), x)        <- lion->lion is ~identity (anti-collapse)
  L_bg    = masked L1(xB, x) on background    <- don't repaint scene on the way out
Perceptual uses the frozen DINOv3 encoder as the content critic (no lpips needed),
which also guards against the desaturation/bleaching failure seen when scaling LoRA.
Gentle LoRA (rank 16, scale 1, lr 5e-5, warmup). Encoder frozen. Eval MASK-FREE.
"""
import sys, os, os.path as osp, glob, math, json, random
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from types import SimpleNamespace
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from PIL import Image
import exp_common as C
from sphere.builder import build_model
from sphere.ema import ModuleEMA
from sphere.utils import load_ckpt

EXP = "workspace/experiments/sphere2-base-imagenet-256px"
DATA = "workspace/data/tier3_lion"
IN = "workspace/roundtrip/inputs"; MSK = "workspace/latent_bias/masks"
OUT = "workspace/latent_bias/tier3"; os.makedirs(f"{OUT}/img", exist_ok=True)
NUMCLS = 1000; LION = 294
ANG, ESTEPS = 85.0, 2                       # edit angle, differentiable steps/edit (train)
R, ALPHA, LR, STEPS, BS, WARM = 16, 16, 5e-5, 800, 4, 50
W_PIX, W_FEAT, W_SELF, W_BG = 1.0, 0.5, 0.5, 0.3
B_POOL = [923, 384, 792, 340, 386, 130, 948]  # zebra, tabby, flamingo, zebra2.., elephant, flamingo2, apple -> random targets

torch.manual_seed(0); random.seed(0)
dev = C.init_dist(port="29583")
cfg = json.load(open(osp.join(EXP, "cfg.json")))
C.patch_dinov3_offline("silu")
m = build_model(SimpleNamespace(**cfg), for_train=False)
m.to(dtype=torch.bfloat16, device=dev, memory_format=torch.channels_last)
ema = ModuleEMA(m); ema.eval().requires_grad_(False)
ck = sorted(glob.glob(osp.join(EXP, "ckpt", "*.pth")))[-1]
load_ckpt(m, ckpt_path=ck, ema_model=ema, strict=False, override_model_with_ema=True, verbose=False)
m.eval().requires_grad_(False)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.Bm = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.enabled = True
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + ((x.float() @ self.A.t() @ self.Bm.t()) * self.scale).to(out.dtype)
        return out

loras = []
for name, mod in list(m.decoder.named_modules()):
    if isinstance(mod, nn.Linear) and name.startswith("blocks."):
        *path, last = name.split("."); parent = m.decoder
        for p in path: parent = getattr(parent, p)
        lo = LoRALinear(mod, R, ALPHA); setattr(parent, last, lo); loras.append(lo)
def set_lora(f):
    for lo in loras: lo.enabled = f
params = [p for lo in loras for p in (lo.A, lo.Bm)]
print(f"LoRA: {len(loras)} linears, {sum(p.numel() for p in params)/1e6:.2f}M params")

def load_split(split):
    out = []
    for p in sorted(glob.glob(f"{DATA}/{split}/img/*.png")):
        tag = osp.splitext(osp.basename(p))[0]
        a = torch.from_numpy(np.array(Image.open(p).convert("RGB"), np.float32)/255.0).permute(2,0,1)*2-1
        msk = torch.from_numpy(np.array(Image.open(f"{DATA}/{split}/mask/{tag}.png").convert("L"), np.float32)/255.0)[None]
        out.append((a, msk))
    return out
train = load_split("train"); evalL = load_split("eval")
print(f"train {len(train)} lions, eval {len(evalL)} lions")

def edit_grad(x_pm1, y, steps, ang):
    m.cached_noise = None
    yt = y.to(dev, torch.long); ynull = torch.full_like(yt, NUMCLS)
    z = m.encoder(x_pm1, ynull); zr, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
    x01 = (m.decoder(zr, yt, alpha=a).float()*0.5+0.5).clamp(0, 1)
    for _ in range(steps-1):
        z = m.encoder((x01*2-1).to(torch.bfloat16), yt); zr, a = m.spherify(z, min(m.max_angle_deg, ang), cache_noise=True)
        x01 = (m.decoder(zr, yt, alpha=a).float()*0.5+0.5).clamp(0, 1)
    return x01
def feat(a01, b01):
    fa = m.encoder((a01*2-1).to(torch.bfloat16)).float()
    with torch.no_grad(): fb = m.encoder((b01*2-1).to(torch.bfloat16)).float()
    return (fa-fb).abs().mean()

opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
def lr_at(it):
    if it < WARM: return LR*it/WARM
    t = (it-WARM)/max(1, STEPS-WARM); return LR*(0.05 + 0.95*0.5*(1+math.cos(math.pi*t)))

for it in range(1, STEPS+1):
    for g in opt.param_groups: g["lr"] = lr_at(it)
    idx = random.sample(range(len(train)), BS)
    x = torch.stack([train[i][0] for i in idx]).to(dev, torch.bfloat16)
    M = torch.stack([train[i][1] for i in idx]).to(dev)
    xo = (x.float()*0.5+0.5).clamp(0, 1)
    yB = torch.tensor([random.choice(B_POOL) for _ in range(BS)], device=dev)
    yA = torch.full((BS,), LION, device=dev)
    set_lora(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xB = edit_grad(x, yB, ESTEPS, ANG)
        xABA = edit_grad((xB*2-1).to(torch.bfloat16), yA, ESTEPS, ANG)
        xAA = edit_grad(x, yA, ESTEPS, ANG)
        L_pix = (xABA-xo).abs().mean()
        L_feat = feat(xABA, xo)
        L_self = (xAA-xo).abs().mean()
        L_bg = (((xB-xo).abs())*(1-M)).sum() / ((1-M).sum()*3 + 1e-6)
        loss = W_PIX*L_pix + W_FEAT*L_feat + W_SELF*L_self + W_BG*L_bg
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if it % 50 == 0 or it == 1:
        print(f"  it {it:4d}/{STEPS}  loss {loss.item():.4f}  (pix {L_pix.item():.3f} feat {L_feat.item():.3f} "
              f"self {L_self.item():.3f} bg {L_bg.item():.3f})  lr {lr_at(it):.2e}")

torch.save({f"{i}.A": lo.A.detach().cpu() for i,lo in enumerate(loras)} |
           {f"{i}.B": lo.Bm.detach().cpu() for i,lo in enumerate(loras)}, f"{OUT}/lion_lora.pth")
print(f"saved -> {OUT}/lion_lora.pth")

# ---------------- eval (MASK-FREE) ------------------------------------------
Zb = C.unit(C.encode_paths(m, [a["path"] for a in json.load(open("workspace/latent_bias/items.json"))], dev))
bias = Zb.mean(0); bias = bias/bias.norm()
@torch.no_grad()
def enc_unit(x_pm1):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(x_pm1.to(dev,torch.bfloat16))).float().flatten(1).cpu())[0]
def res_cos(a,b):
    ra=a-(a@bias)*bias; rb=b-(b@bias)*bias; return F.cosine_similarity(ra,rb,dim=0).item()
def raw_cos(a,b): return F.cosine_similarity(a,b,dim=0).item()
@torch.no_grad()
def edit_eval(x_pm1, y, steps, ang):
    yt=torch.tensor([y],device=dev,dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m.edit(x_pm1.to(dev,torch.bfloat16), yt, forward_steps=steps, edit_angle=ang,
                      cache_sampling_noise=True).float().clamp(0,1).cpu()

import torchvision
EVAL_STEPS, EVAL_B = 6, [923, 384, 792]     # zebra, tabby, flamingo
# eval set: 4 held-out ImageNet lions + the original roundtrip lion
evalset = [(e[0], f"im{i}") for i,e in enumerate(evalL)]
evalset.append((C.load_img(f"{IN}/03_lion.jpg"), "rt_lion"))
rows=[]
for xi,(x1,tag) in enumerate(evalset):
    x = x1.unsqueeze(0).to(dev); o=(x*0.5+0.5).clamp(0,1).cpu(); z0=enc_unit(x)
    rec={"tag":tag,"base":{"rt":[]}, "lora":{"rt":[]}}
    figs=[o]
    for which,flag in [("base",False),("lora",True)]:
        set_lora(flag)
        for B in EVAL_B:
            xB=edit_eval(x,B,EVAL_STEPS,ANG); xA=edit_eval(xB*2-1,LION,EVAL_STEPS,ANG)
            rec[which]["rt"].append(res_cos(z0, enc_unit(xA*2-1)))
        # one representative (zebra) for the figure
        xB=edit_eval(x,923,EVAL_STEPS,ANG); xA=edit_eval(xB*2-1,LION,EVAL_STEPS,ANG)
        figs += [xB, xA]
    rec["base_rt"]=float(np.mean(rec["base"]["rt"])); rec["lora_rt"]=float(np.mean(rec["lora"]["rt"]))
    rows.append(rec)
    cmp=torch.cat(figs)  # orig | baseB baseABA | loraB loraABA
    torchvision.utils.save_image(torchvision.utils.make_grid(cmp,nrow=5,padding=4,pad_value=1.0), f"{OUT}/img/tier3_{tag}.png")
    print(f"{tag:8s}  RT-cos base={rec['base_rt']:+.3f}  lora={rec['lora_rt']:+.3f}  (avg over {len(EVAL_B)} B)")

json.dump({"train_imgs":len(train),"class":"lion","rank":R,"eval_steps":EVAL_STEPS,"eval_B":EVAL_B,"rows":rows},
          open(f"{OUT}/tier3.json","w"), indent=2)
mb=np.mean([r['base_rt'] for r in rows]); ml=np.mean([r['lora_rt'] for r in rows])
print(f"\nMEAN round-trip content-cos (mask-free, {len(rows)} held-out lions): base {mb:+.3f} -> lora {ml:+.3f}  (Tier1 masked=0.83, baseline~0.23)")
dist.destroy_process_group()
