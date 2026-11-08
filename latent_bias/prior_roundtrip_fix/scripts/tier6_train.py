"""
Tier 5 - SPATIAL-CONDITIONING GATE (the principled fix for the mask-free ceiling).

Root cause (confirmed across tiers): the class label is injected GLOBALLY to every
token, so the whole scene gets repainted. Fix: a tiny gate g(z) in [0,1] per token
picks, from the latent alone, where to apply the class. The decoder conditioning
becomes per-token:
    cond = g * emb(class) + (1-g) * emb(null)
Background tokens (g~0) get null conditioning -> reconstruct the ORIGINAL scene;
subject tokens (g~1) get the class -> edited. No decoder weights change; we feed
this through the decoder's existing cond_embed override. Mask supervises g during
training; at inference g is predicted from z (mask-free).

Trained on 20 lion images only. Decoder/encoder frozen; ONLY the gate is learned.
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
DATA = "workspace/data/tier3_lion"; IN = "workspace/roundtrip/inputs"
OUT = "workspace/latent_bias/tier6"; os.makedirs(f"{OUT}/img", exist_ok=True)
NUMCLS = 1000; LION = 294; GRID = 16
ANG, ESTEPS = 85.0, 2
LR, STEPS, BS, WARM = 3e-4, 1200, 4, 60
W_GATE, W_OUT, W_BG, W_CYC, W_FEAT = 1.0, 1.0, 1.0, 1.0, 0.5
B_POOL = [923, 384, 792, 340, 386, 130, 948]

torch.manual_seed(0); random.seed(0)
dev = C.init_dist(port="29585")
cfg = json.load(open(osp.join(EXP, "cfg.json")))
C.patch_dinov3_offline("silu")
m = build_model(SimpleNamespace(**cfg), for_train=False)
m.to(dtype=torch.bfloat16, device=dev, memory_format=torch.channels_last)
ema = ModuleEMA(m); ema.eval().requires_grad_(False)
ck = sorted(glob.glob(osp.join(EXP, "ckpt", "*.pth")))[-1]
load_ckpt(m, ckpt_path=ck, ema_model=ema, strict=False, override_model_with_ema=True, verbose=False)
m.eval().requires_grad_(False)
TOKD = m.latent_shape[2]; print(f"latent {tuple(m.latent_shape)}  token dim {TOKD}")

class Gate(nn.Module):
    def __init__(self, din, dh=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias, 3.0)  # g~0.95 at init (==baseline)
    def forward(self, z):
        return torch.sigmoid(self.net(z.float()))   # [B,N,1]
gate = Gate(TOKD).to(dev)
params = list(gate.parameters())
print(f"gate params {sum(p.numel() for p in params)/1e3:.1f}K")

def load_split(split):
    out=[]
    for p in sorted(glob.glob(f"{DATA}/{split}/img/*.png")):
        tag=osp.splitext(osp.basename(p))[0]
        a=torch.from_numpy(np.array(Image.open(p).convert("RGB"),np.float32)/255.).permute(2,0,1)*2-1
        msk=torch.from_numpy(np.array(Image.open(f"{DATA}/{split}/mask/{tag}.png").convert("L"),np.float32)/255.)[None]
        out.append((a,msk))
    return out
train=load_split("train"); evalL=load_split("eval")
print(f"train {len(train)} eval {len(evalL)}")

def yemb(y):  # [B,1,H]
    return m.decoder.y_embedder(y.to(dev,torch.long), False)

def spatial_edit(x_pm1, y, steps, ang, reset=True, z0=None, g=None):
    """LEARNED latent-masking (mask-free Tier 1):
    gate g read from the CLEAN latent z0 pins background tokens to z0 (no noise)
    AND applies null conditioning there; foreground tokens are noised + class-y."""
    if reset: m.cached_noise=None
    yt=y.to(dev,torch.long); ynull=torch.full_like(yt,NUMCLS)
    if z0 is None: z0=m.f(m.encoder(x_pm1, ynull))      # clean content latent (scene)
    if g is None: g=gate(z0)                             # [B,N,1] from CLEAN latent
    def step(xin):
        z=m.encoder(xin, ynull if xin is x_pm1 else yt); zr,a=m.spherify(z,min(m.max_angle_deg,ang),cache_noise=True)
        z_mix=m.f(g*zr + (1-g)*z0)                       # pin background latent to clean scene
        cond=g*yemb(yt) + (1-g)*yemb(ynull)              # spatial conditioning
        return (m.decoder(z_mix,yt,alpha=a,cond_embed=cond).float()*0.5+0.5).clamp(0,1), a
    x01,_=step(x_pm1)
    for _ in range(steps-1):
        x01,_=step((x01*2-1).to(torch.bfloat16))
    return x01, g, z0

def base_edit(x_pm1, y, steps, ang, reset=True):   # full global conditioning (teacher)
    if reset: m.cached_noise=None
    yt=y.to(dev,torch.long); ynull=torch.full_like(yt,NUMCLS)
    z=m.encoder(x_pm1,ynull); zr,a=m.spherify(z,min(m.max_angle_deg,ang),cache_noise=True)
    x01=(m.decoder(zr,yt,alpha=a).float()*0.5+0.5).clamp(0,1)
    for _ in range(steps-1):
        z=m.encoder((x01*2-1).to(torch.bfloat16),yt); zr,a=m.spherify(z,min(m.max_angle_deg,ang),cache_noise=True)
        x01=(m.decoder(zr,yt,alpha=a).float()*0.5+0.5).clamp(0,1)
    return x01
def feat(a01,b01):
    fa=m.encoder((a01*2-1).to(torch.bfloat16)).float()
    with torch.no_grad(): fb=m.encoder((b01*2-1).to(torch.bfloat16)).float()
    return (fa-fb).abs().mean()

opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.0)
def lr_at(it):
    if it<WARM: return LR*it/WARM
    t=(it-WARM)/max(1,STEPS-WARM); return LR*(0.05+0.95*0.5*(1+math.cos(math.pi*t)))
bce=nn.BCELoss()
for it in range(1,STEPS+1):
    for g in opt.param_groups: g["lr"]=lr_at(it)
    idx=random.sample(range(len(train)),BS)
    x=torch.stack([train[i][0] for i in idx]).to(dev,torch.bfloat16)
    M=torch.stack([train[i][1] for i in idx]).to(dev)
    mtok=F.adaptive_avg_pool2d(M,GRID).reshape(BS,GRID*GRID,1).clamp(0,1)   # [B,N,1] target gate
    xo=(x.float()*0.5+0.5).clamp(0,1)
    yB=torch.tensor([random.choice(B_POOL) for _ in range(BS)],device=dev)
    yA=torch.full((BS,),LION,device=dev)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        with torch.no_grad(): teacherB=base_edit(x,yB,ESTEPS,ANG,reset=True)   # cache noise
        xB,g0,z0=spatial_edit(x,yB,ESTEPS,ANG,reset=False)                    # reuse noise; g,z0 from original scene
        xABA,_,_=spatial_edit((xB*2-1).to(torch.bfloat16),yA,ESTEPS,ANG,reset=True,z0=z0,g=g0)  # same scene
        nF=(M.sum()*3+1e-6); nB=((1-M).sum()*3+1e-6)
        L_gate=(g0.float()-mtok.float()).abs().mean()   # L1 on gate map (autocast-safe)
        L_out=(((xB-teacherB).abs())*M).sum()/nF          # subject matches real B edit
        L_bg =(((xB-xo).abs())*(1-M)).sum()/nB            # background == original scene
        L_cyc=(xABA-xo).abs().mean()
        L_feat=feat(xABA,xo)
        loss=W_GATE*L_gate+W_OUT*L_out+W_BG*L_bg+W_CYC*L_cyc+W_FEAT*L_feat
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if it%50==0 or it==1:
        print(f"  it {it:4d}/{STEPS} loss {loss.item():.4f} (gate {L_gate.item():.3f} out {L_out.item():.3f} "
              f"bg {L_bg.item():.3f} cyc {L_cyc.item():.3f} feat {L_feat.item():.3f}) lr {lr_at(it):.2e}")

torch.save(gate.state_dict(), f"{OUT}/gate.pth"); print(f"saved -> {OUT}/gate.pth")

# ---------------- eval (mask-free) -----------------------------------------
Zb=C.unit(C.encode_paths(m,[a["path"] for a in json.load(open("workspace/latent_bias/items.json"))],dev))
bias=Zb.mean(0); bias=bias/bias.norm()
@torch.no_grad()
def enc_unit(x_pm1):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return C.unit(m.f(m.encoder(x_pm1.to(dev,torch.bfloat16))).float().flatten(1).cpu())[0]
def res_cos(a,b):
    ra=a-(a@bias)*bias; rb=b-(b@bias)*bias; return F.cosine_similarity(ra,rb,dim=0).item()
@torch.no_grad()
def base_eval(x,y,steps,ang):
    yt=torch.tensor([y],device=dev,dtype=torch.long)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return m.edit(x.to(dev,torch.bfloat16),yt,forward_steps=steps,edit_angle=ang,cache_sampling_noise=True).float().clamp(0,1).cpu()
@torch.no_grad()
def spatial_eval(x,y,steps,ang):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        o,g,_=spatial_edit(x.to(dev,torch.bfloat16),torch.tensor([y],device=dev),steps,ang,reset=True)
    return o.float().clamp(0,1).cpu(), g
import torchvision
ESTEPS_E, EVAL_B = 6, [923,384,792]
evalset=[(e[0],f"im{i}") for i,e in enumerate(evalL)]; evalset.append((C.load_img(f"{IN}/03_lion.jpg"),"rt_lion"))
rows=[]
for x1,tag in evalset:
    x=x1.unsqueeze(0).to(dev); o=(x*0.5+0.5).clamp(0,1).cpu(); z0=enc_unit(x)
    rb=[]; rs=[]
    for B in EVAL_B:
        xB=base_eval(x,B,ESTEPS_E,ANG); xA=base_eval(xB*2-1,LION,ESTEPS_E,ANG); rb.append(res_cos(z0,enc_unit(xA*2-1)))
        sB,_=spatial_eval(x,B,ESTEPS_E,ANG); sA,_=spatial_eval(sB*2-1,LION,ESTEPS_E,ANG); rs.append(res_cos(z0,enc_unit(sA*2-1)))
    # figure with gate map (zebra)
    bB=base_eval(x,923,ESTEPS_E,ANG); bA=base_eval(bB*2-1,LION,ESTEPS_E,ANG)
    sB,g=spatial_eval(x,923,ESTEPS_E,ANG); sA,_=spatial_eval(sB*2-1,LION,ESTEPS_E,ANG)
    gmap=g.float().reshape(1,1,GRID,GRID); gmap=F.interpolate(gmap,size=256,mode="nearest").repeat(1,3,1,1).cpu()
    cmp=torch.cat([o,bB,bA,sB,sA,gmap])
    torchvision.utils.save_image(torchvision.utils.make_grid(cmp,nrow=6,padding=4,pad_value=1.0),f"{OUT}/img/tier5_{tag}.png")
    rows.append({"tag":tag,"base_rt":float(np.mean(rb)),"spatial_rt":float(np.mean(rs))})
    print(f"{tag:8s} RT-cos base={np.mean(rb):+.3f} spatial={np.mean(rs):+.3f}")
json.dump({"train_imgs":len(train),"class":"lion","method":"learned_latent_mask","rows":rows},open(f"{OUT}/tier5.json","w"),indent=2)
mb=np.mean([r['base_rt'] for r in rows]); ms=np.mean([r['spatial_rt'] for r in rows])
print(f"\nMEAN round-trip content-cos (mask-free): base {mb:+.3f} -> learned-mask {ms:+.3f}  (Tier1 rembg-mask=0.83, Tier4 LoRA=0.35)")
dist.destroy_process_group()
