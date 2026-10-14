import contextlib
import hashlib
import json
import logging
import math
import os
import os.path as osp
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
import torch.nn as nn

from .geometry import f, f_local, rotate

logger = logging.getLogger(__name__)


class SphereBackbone(nn.Module, ABC):
    latent_shape: tuple
    max_angle_deg: float = 90.0
    angle_conditioned: bool = True
    num_classes: int = 0
    spherify_mode: str = "global"
    compute_dtype: torch.dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.counters = {"enc_fwd": 0, "dec_fwd": 0, "enc_bwd": 0, "dec_bwd": 0}

    @property
    def L(self) -> int:
        return math.prod(self.latent_shape)

    def reset_counters(self):
        for k in self.counters:
            self.counters[k] = 0

    @abstractmethod
    def encode_raw(self, x: torch.Tensor, c=None) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor, alpha_deg, c=None) -> torch.Tensor:
        pass

    def spherify(self, z):
        return f(z) if self.spherify_mode == "global" else f_local(z)

    def encode(self, x, c=None):
        return self.spherify(self.encode_raw(x, c))

    def noise_like(self, v, generator=None):
        dev = generator.device if generator is not None else v.device
        e = torch.randn(v.shape, device=dev, generator=generator).to(v.device)
        return self.spherify(e)

    def random_latent(self, B, device, generator=None):
        dev = generator.device if generator is not None else device
        e = torch.randn(B, *self.latent_shape, device=dev, generator=generator)
        return self.spherify(e.to(device))

    def rotate(self, v, e_hat, alpha_deg):
        if isinstance(alpha_deg, (int, float)) and alpha_deg <= 0:
            return v
        return rotate(v, e_hat, alpha_deg)

    def project(self, x, alpha_deg, e_hat, c=None, dec_alpha_deg=None):
        z = self.rotate(self.encode(x, c), e_hat, alpha_deg)
        return self.decode(z, alpha_deg if dec_alpha_deg is None else dec_alpha_deg, c)

    def class_tensor(self, B, c, device):
        if c is None:
            return torch.full((B,), self.num_classes, device=device, dtype=torch.long)
        c = torch.as_tensor(c, device=device, dtype=torch.long).reshape(-1)
        return c.expand(B) if c.numel() == 1 else c

    @contextlib.contextmanager
    def differentiable_attention(self):
        yield


class SE2Backbone(SphereBackbone):
    def __init__(self, g, cfg: dict, ckpt_path: str = "", compute_dtype=torch.float32):
        super().__init__()
        self.g = g.eval().requires_grad_(False)
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.latent_shape = tuple(g.latent_shape[1:])
        self.max_angle_deg = float(g.max_angle_deg)
        self.angle_conditioned = bool(g.use_angle_cond)
        self.num_classes = int(g.num_classes)
        self.spherify_mode = g.spherify_mode
        self.image_size = int(cfg["image_size"])
        self.compute_dtype = compute_dtype

    @classmethod
    def from_checkpoint(cls, job_dir, ckpt_fname=None, dev_dir="workspace", device="cuda",
                        compute_dtype=torch.float32, weights_dtype=torch.float32):
        from sphere.builder import build_model

        exp_dir = osp.join(dev_dir, "experiments", job_dir)
        cfg = json.load(open(osp.join(exp_dir, "cfg.json")))
        g = build_model(SimpleNamespace(**cfg), for_train=False)
        ckpt_dir = osp.join(exp_dir, "ckpt")
        if ckpt_fname is None:
            ckpt_fname = sorted(p for p in os.listdir(ckpt_dir) if p.endswith(".pth"))[-1]
        path = osp.join(ckpt_dir, ckpt_fname)
        sd = torch.load(path, map_location="cpu", weights_only=True)["ema_model"]
        g.load_state_dict(sd, strict=True)
        g.to(device=device, dtype=weights_dtype)
        logger.info(f"loaded EMA weights from {path}")
        return cls(g, cfg, ckpt_path=path, compute_dtype=compute_dtype)

    def ckpt_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.ckpt_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    def _autocast(self, device):
        on = self.compute_dtype != torch.float32
        return torch.autocast(device.type, dtype=self.compute_dtype, enabled=on)

    def _check_eval(self):
        assert not self.g.training, "SE2Backbone must stay in eval mode"

    def encode_raw(self, x, c=None):
        self._check_eval()
        self.counters["enc_fwd"] += x.shape[0]
        with self._autocast(x.device):
            z = self.g.encoder(x, None)
        return z.float()

    def decode(self, z, alpha_deg, c=None):
        self._check_eval()
        B = z.shape[0]
        self.counters["dec_fwd"] += B
        alpha = None
        if self.angle_conditioned:
            a = torch.as_tensor(alpha_deg, dtype=torch.float32, device=z.device).reshape(-1)
            alpha = torch.deg2rad(a.expand(B) if a.numel() == 1 else a).view(B, 1, 1)
        y = self.class_tensor(B, c, z.device) if self.num_classes > 0 else None
        with self._autocast(z.device):
            x = self.g.decoder(z.to(self.g.decoder.x_embedder.weight.dtype), y, alpha=alpha)
        return x.float()

    @contextlib.contextmanager
    def differentiable_attention(self):
        saved = []
        for m in self.g.decoder.modules():
            if hasattr(m, "sdpa_mode"):
                saved.append((m, m.sdpa_mode))
                m.sdpa_mode = "manual"
        hf = self.g.encoder.encoder
        hf_prev = getattr(hf.config, "_attn_implementation", "sdpa")
        hf.set_attn_implementation("eager")
        try:
            yield
        finally:
            for m, mode in saved:
                m.sdpa_mode = mode
            hf.set_attn_implementation(hf_prev)


class DummyBackbone(SphereBackbone):
    def __init__(self, H=32, patch=4, D=8, width=64, angle_conditioned=True, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.H, self.patch = H, patch
        N = (H // patch) ** 2
        self.latent_shape = (N, D)
        self.angle_conditioned = angle_conditioned
        self.num_classes = 0
        pd = 3 * patch * patch

        def lin(i, o):
            m = nn.Linear(i, o)
            with torch.no_grad():
                m.weight.copy_(torch.randn(o, i, generator=gen) / math.sqrt(i))
                m.bias.copy_(0.1 * torch.randn(o, generator=gen))
            return m

        self.e_in, self.e_qkv, self.e_out = lin(pd, width), lin(width, 3 * width), lin(width, D)
        self.d_in, self.d_qkv, self.d_out = lin(D + 1, width), lin(width, 3 * width), lin(width, pd)
        self.pos = nn.Parameter(0.1 * torch.randn(1, N, width, generator=gen))
        self.requires_grad_(False)

    @staticmethod
    def _attn(h, qkv):
        q, k, v = qkv(h).chunk(3, dim=-1)
        w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        return w @ v

    def _patchify(self, x):
        B, p = x.shape[0], self.patch
        return x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * p * p)

    def _unpatchify(self, t):
        B, p, h = t.shape[0], self.patch, self.H // self.patch
        return t.view(B, h, h, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, self.H, self.H)

    def encode_raw(self, x, c=None):
        self.counters["enc_fwd"] += x.shape[0]
        h = torch.tanh(self.e_in(self._patchify(x))) + self.pos
        h = h + self._attn(h, self.e_qkv)
        return self.e_out(h)

    def decode(self, z, alpha_deg, c=None):
        B = z.shape[0]
        self.counters["dec_fwd"] += B
        a = torch.as_tensor(alpha_deg if self.angle_conditioned else 0.0, dtype=torch.float32, device=z.device)
        a = (torch.deg2rad(a) / (math.pi / 2)).reshape(-1).expand(B)[:, None, None].expand(B, z.shape[1], 1)
        h = torch.tanh(self.d_in(torch.cat([z, a], -1))) + self.pos
        h = h + self._attn(h, self.d_qkv)
        return torch.tanh(self._unpatchify(self.d_out(h)))
