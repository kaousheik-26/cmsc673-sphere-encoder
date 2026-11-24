import os, sys, os.path as osp, json, torch, torch.distributed as dist
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533", RANK="0", WORLD_SIZE="1")
dist.init_process_group("gloo")
sys.path.insert(0, osp.dirname(__file__))
from audiomel import to_img, from_img, FRAMES
from textse2 import build, ROOT
from sphere.loss import G1Loss
meta=json.load(open(f"{ROOT}/data/meta.json")); hi=meta["hi"]
mels=torch.load(f"{ROOT}/data/mels.pt").float()
x01=to_img(mels[:2,:,:FRAMES],hi); assert (from_img(x01,hi)-mels[:2,:,:FRAMES].clamp(meta['lo'],hi)).abs().max()<1e-4, "img<->mel not exact"
print("img<->mel exact OK")
te=torch.load(f"{ROOT}/data/text_emb.pt"); temb=te["emb"][:2]
m,sa=build(for_train=True); m.g1.ddp_rank,m.g1.ddp_world_size,m.g1.step_seed=0,1,0
lf=G1Loss(latent_spherify_mode=sa.spherify_mode, perceptual_ckpt_path=f"{ROOT}/workspace/pretrained/lpips",
  perceptual_convnext_aug=sa.perceptual_convnext_aug, perceptual_convnext_loss_type=sa.perceptual_convnext_loss_type,
  perceptual_convnext_loss_reduction=sa.perceptual_convnext_loss_reduction, distance_loss_type=sa.distance_loss_type,
  distance_weight=1.0, distance_apply_regime="lo", perceptual_weight=1.0, latent_consistency_weight=0.1,
  latent_consistency_apply_regime="hi", weight_anchor_loss=True, weight_anchor_cutoff_params=sa.weight_anchor_cutoff_params)
x=x01*2-1
px,hs,lat=m(x,temb,torch.tensor([False,True]))
loss=lf(input=px,target=x,hidden_states=hs,latent_list=lat); loss.backward()
print("loss",loss.item(),{k:round(float(v),4) for k,v in lf.log_dict.items() if torch.is_tensor(v) and v.numel()==1})
print("tc grad", m.tc.mlp[2].weight.grad.abs().sum().item(), "null grad", m.tc.null.grad.abs().sum().item(), "base grad", m.tc.base.grad.abs().sum().item())
unused=[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None]; print("unused trainable params:", len(unused), unused[:5])
m.eval()
r=m.reconstruct(x); print("recon mel err dB", (8.6859*(from_img(r,hi)-from_img(x01,hi)).abs().mean()).item())
g=m.generate(temb,steps=2,cfg_deg=0); g2=m.generate(temb,steps=2,cfg_deg=2.0); print("gen", g.shape, g.min().item(), g.max().item(), g2.shape)
print("SMOKE_OK")
