"""shared helpers for the latent-bias experiment suite."""
import os, json, glob, math, datetime
import os.path as osp
from types import SimpleNamespace
import numpy as np, torch, torch.distributed as dist
from torchvision import datasets
from sphere.loader import center_crop_arr
from sphere.builder import build_model
from sphere.ema import ModuleEMA
from sphere.utils import load_ckpt


def patch_dinov3_offline(hidden_act="silu"):
    import sphere.encoder as em
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    orig = em.AutoModel.from_pretrained
    def fp(name, *a, **k):
        if "dinov3" in str(name).lower():
            return DINOv3ViTModel(DINOv3ViTConfig(
                patch_size=16, hidden_size=384, intermediate_size=1536,
                num_hidden_layers=12, num_attention_heads=6, num_register_tokens=4,
                use_gated_mlp=True, hidden_act=hidden_act, image_size=256,
                query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True))
        return orig(name, *a, **k)
    em.AutoModel.from_pretrained = staticmethod(fp)


def init_dist(port="29571"):
    for k, v in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                     MASTER_ADDR="127.0.0.1", MASTER_PORT=port).items():
        os.environ.setdefault(k, v)
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(hours=1))
    return dev


def load_model(exp_dir, dev, dino_act="silu", seed=0):
    torch.manual_seed(seed)
    cfg = json.load(open(osp.join(exp_dir, "cfg.json")))
    patch_dinov3_offline(dino_act)
    m = build_model(SimpleNamespace(**cfg), for_train=False)
    m.to(dtype=torch.bfloat16, device=dev, memory_format=torch.channels_last)
    ema = ModuleEMA(m); ema.eval().requires_grad_(False)
    ck = sorted(glob.glob(osp.join(exp_dir, "ckpt", "*.pth")))[-1]
    load_ckpt(m, ckpt_path=ck, ema_model=ema, strict=False,
              override_model_with_ema=True, verbose=False)
    m.eval().requires_grad_(False)
    return m, cfg, ck


def load_img(path, size=256):
    x = datasets.folder.pil_loader(path)
    x = np.array(center_crop_arr(x, size)).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1) * 2 - 1  # [-1,1]


@torch.no_grad()
def encode_paths(m, paths, dev, size=256, return_raw=False):
    """-> (Z spherified flat [M,ND]), optionally (U raw flat)."""
    Z, U = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for p in paths:
            x = load_img(p, size).unsqueeze(0).to(dev, torch.bfloat16)
            u = m.encoder(x)
            Z.append(m.f(u).float().flatten(1).cpu())
            if return_raw:
                U.append(u.float().flatten(1).cpu())
    Z = torch.cat(Z)
    return (Z, torch.cat(U)) if return_raw else Z


def unit(v):
    return v / v.norm(dim=1, keepdim=True)


def mean_offdiag_cos(Vu):
    G = Vu @ Vu.t()
    n = Vu.shape[0]
    return ((G.sum() - G.diag().sum()) / (n * (n - 1))).item()


def mean_dir_energy(Vu):
    return unit(Vu).mean(0).norm().item()


def top1_gram_frac(Vu):
    ev = torch.linalg.eigvalsh(Vu @ Vu.t())
    return (ev[-1] / ev.sum()).item()


def psnr(a, b):
    e = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if e < 1e-12 else 10 * math.log10(1.0 / e)
