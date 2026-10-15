"""
Thin, gradient-capable API over the released Sphere Encoder 2 model (sphere.model.G1).

The repo's own G1.encode / decode / reconstruct / generate are wrapped in
@torch.no_grad(), and sphere.utils.load_ckpt needs torch.distributed. This
wrapper builds G1 from a checkpoint folder's cfg.json, loads the EMA weights
without torch.distributed, and exposes the primitives the research plan uses:

    tokens = se2.encode_tokens(x)          # raw DINOv3-S+ patch tokens [B, 256, 384]
    z      = se2.spherify(tokens)          # on the sphere of radius sqrt(d), d = 98,304
    e      = se2.sample_noise(B)           # F(randn), uniform on the sphere
    zr     = se2.rotate(z, e, alpha_deg)   # cos(a) z + sin(a) e
    x_hat  = se2.decode(zr, y, alpha_deg)  # [-1, 1], unclamped

Images are [B, 3, H, W] in [-1, 1] (the repo convention). All functions keep
the autograd graph unless called under torch.no_grad().
"""

import json
import math
import os
import os.path as osp
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sphere.builder import build_model
from sphere.layers import vector_rms_norm, grid_rms_norm
from sphere.model import rotate_away_from as _repo_rotate_away_from

DEFAULT_WORKSPACE = "/fs/nexus-scratch/vinayakg/se3d/workspace"
MODELS = {
    "large-fdlite-256": "sphere2-large-imagenet-fd-lite-256px",
    "base-256": "sphere2-base-imagenet-256px",
    "large-256": "sphere2-large-imagenet-256px",
    "large-fdlite-512": "sphere2-large-imagenet-fd-lite-512px",
}


def load_class_map(workspace=DEFAULT_WORKSPACE):
    """The released models use their own class order. Returns
    (imagenet_idx -> model class_id, model class_id -> label)."""
    path = osp.join(workspace, "imagenet.json", "folder_to_id_to_label.json")
    to_model, labels = {}, {}
    with open(path) as fio:
        for line in fio:
            if line.strip():
                r = json.loads(line)
                to_model[int(r["ori_class_id"])] = int(r["class_id"])
                labels[int(r["class_id"])] = r["label"]
    return to_model, labels


def rotate_away_from(z, z_ref, deg, rel_eps=1e-4):
    """Spherical CFG step (SE2 Eq. A17): rotate z away from z_ref by `deg`
    degrees on their great circle, keeping |z|. Same math as the repo's
    sphere.model.rotate_away_from, but the degenerate case (z ~ +-z_ref) is
    detected with a RELATIVE threshold. The repo uses an absolute 1e-6 on a
    vector of norm ~313, so rotate_away_from(z, z, g) shrinks z by
    (cos g - sin g) in fp32 instead of returning z."""
    zc, zu = z.float(), z_ref.float()
    dims = tuple(range(1, z.ndim))
    r = zc.norm(dim=dims, keepdim=True)
    zu = zu * (r / zu.norm(dim=dims, keepdim=True).clamp_min(1e-12))
    t = -(zu - ((zu * zc).sum(dims, keepdim=True) / (r * r)) * zc)  # tangent of -z_ref at z
    tn = t.norm(dim=dims, keepdim=True)
    beta = math.radians(deg)
    out = math.cos(beta) * zc + math.sin(beta) * r * t / tn.clamp_min(1e-12)
    return torch.where(tn > rel_eps * r, out, zc).to(z.dtype)


def resolve_exp_dir(name_or_dir, workspace=DEFAULT_WORKSPACE):
    if osp.isdir(name_or_dir):
        return name_or_dir
    name = MODELS.get(name_or_dir, name_or_dir)
    path = osp.join(workspace, "experiments", name)
    assert osp.isdir(path), f"checkpoint folder not found: {path}"
    return path


def _latest_ckpt(exp_dir):
    ckpt_dir = osp.join(exp_dir, "ckpt")
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pth"))
    assert ckpts, f"no .pth in {ckpt_dir}"
    return osp.join(ckpt_dir, ckpts[-1])


def _normalize_name(name):
    return (
        name.replace("module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("._orig_mod", "")
    )


class SE2:
    """Sphere Encoder 2 built from <exp_dir>/cfg.json with EMA weights."""

    def __init__(
        self,
        name_or_dir="large-fdlite-256",
        device="cuda",
        use_ema=True,
        amp_dtype=torch.bfloat16,
        workspace=DEFAULT_WORKSPACE,
        decoder_ckpt=None,
    ):
        self.exp_dir = resolve_exp_dir(name_or_dir, workspace)
        with open(osp.join(self.exp_dir, "cfg.json")) as fio:
            self.cfg = json.load(fio)
        args = SimpleNamespace(**self.cfg)
        args.use_qk_norm = self.cfg.get("use_qk_norm", False)
        model = build_model(args, for_train=False)

        self.ckpt_path = _latest_ckpt(self.exp_dir)
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        key = "ema_model" if (use_ema and "ema_model" in ckpt) else "model"
        state = {_normalize_name(k): v for k, v in ckpt[key].items()}
        msg = model.load_state_dict(state, strict=True)
        assert not msg.missing_keys and not msg.unexpected_keys, msg
        self.weights = key
        del ckpt, state
        self.decoder_ckpt = decoder_ckpt
        if decoder_ckpt:  # fine-tuned decoder (se3d/train/ft_decoder.py)
            dsd = torch.load(decoder_ckpt, map_location="cpu", weights_only=True)
            dsd = {k: v.float() for k, v in dsd.items()}
            msg = model.decoder.load_state_dict(dsd, strict=True)
            assert not msg.missing_keys and not msg.unexpected_keys, msg

        self.model = model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        self.num_classes = self.model.num_classes
        self.null_class = self.model.num_classes  # learned null embedding index
        self.latent_shape = tuple(self.model.latent_shape[1:])  # (256, 384)
        self.d = math.prod(self.latent_shape)
        self.radius = math.sqrt(self.d)
        self.image_size = self.cfg["image_size"]
        self.spherify_mode = self.model.spherify_mode
        self.use_angle_cond = self.model.use_angle_cond

    # ------------------------------------------------------------------ utils
    def autocast(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def labels(self, y, batch_size):
        """None -> null class; int -> broadcast; tensor -> as is."""
        if self.num_classes == 0:
            return None
        if y is None:
            y = self.null_class
        if isinstance(y, int):
            y = torch.full((batch_size,), y, dtype=torch.long)
        return y.to(device=self.device, dtype=torch.long)

    def _alpha(self, batch_size, alpha_deg):
        if not self.use_angle_cond:
            return None
        a = torch.as_tensor(alpha_deg, dtype=torch.float32, device=self.device)
        a = torch.deg2rad(a).reshape(-1, 1, 1)
        return a.expand(batch_size, 1, 1).contiguous()

    # ------------------------------------------------------------ primitives
    def encode_tokens(self, x, y=None):
        """x [-1,1] -> raw encoder tokens [B, N, D] (float32, not spherified).
        The released encoder is unconditional, y is accepted for API symmetry."""
        with self.autocast():
            z = self.model.encoder(x, self.labels(y, x.shape[0]))
        return z.float()

    def spherify(self, z, mode=None):
        mode = mode or self.spherify_mode
        fn = vector_rms_norm if mode == "global" else grid_rms_norm
        return fn(z.float(), zero_mean=False)

    def encode(self, x, y=None):
        """F(E(x)) on the sphere of radius sqrt(d)."""
        return self.spherify(self.encode_tokens(x, y))

    def sample_noise(self, batch_size, generator=None):
        e = torch.randn(
            (batch_size, *self.latent_shape), generator=generator, device=self.device
        )
        return self.spherify(e)

    @staticmethod
    def rotate(z, e, alpha_deg):
        """cos(a) z + sin(a) e. alpha_deg: float or [B] tensor (degrees)."""
        a = torch.as_tensor(alpha_deg, dtype=torch.float32, device=z.device)
        a = torch.deg2rad(a).reshape(-1, *([1] * (z.ndim - 1)))
        return torch.cos(a) * z + torch.sin(a) * e

    def decode(self, z, y=None, alpha_deg=0.0):
        """z on the sphere -> image in [-1, 1] (unclamped, like the repo)."""
        B = z.shape[0]
        with self.autocast():
            x = self.model.decoder(
                z, self.labels(y, B), alpha=self._alpha(B, alpha_deg)
            )
        return x.float()

    @staticmethod
    def rotate_away_from(z, z_ref, deg):
        return rotate_away_from(z, z_ref, deg)

    def imagenet_to_model_class(self, idx):
        """standard ImageNet-1k index (int or LongTensor) -> this model's class id"""
        if not hasattr(self, "_cls_map"):
            self._cls_map, self.class_labels = load_class_map()
            self._cls_lut = torch.tensor([self._cls_map[i] for i in range(1000)])
        if isinstance(idx, int):
            return self._cls_map[idx]
        return self._cls_lut.to(idx.device)[idx]

    # ------------------------------------------------------------ composites
    def reconstruct(self, x, y=None, alpha_deg=0.0, e=None):
        """D(rotate(F(E(x)), e, a), y, a). alpha 0 = plain reconstruction."""
        z = self.encode(x)
        if alpha_deg != 0:
            e = self.sample_noise(x.shape[0]) if e is None else e
            z = self.rotate(z, e, alpha_deg)
        return self.decode(z, y, alpha_deg)

    def project(self, x, y=None, alpha_deg=60.0, e=None, steps=1):
        """SP3-style projector P(x) = D(rotate(F(E(x)), e, a)), applied `steps`
        times with a shared noise e."""
        e = self.sample_noise(x.shape[0]) if e is None else e
        for _ in range(steps):
            x = self.reconstruct(x, y, alpha_deg, e)
        return x

    def generate(self, batch_size, y=None, init_angle=None, loop_angle=None, steps=1,
                 e=None, generator=None):
        """Alg. 1 of SE2 (no CFG), returns image in [-1, 1]."""
        init_angle = self.default_sampling_angle() if init_angle is None else init_angle
        loop_angle = init_angle if loop_angle is None else loop_angle
        e = self.sample_noise(batch_size, generator) if e is None else e
        x = self.decode(e, y, init_angle)
        for _ in range(steps - 1):
            x = self.decode(self.rotate(self.encode(x), e, loop_angle), y, loop_angle)
        return x

    def default_sampling_angle(self):
        return 84.0 if self.cfg["vit_dec_model_size"] == "large" else 85.5

    @staticmethod
    def to01(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    @staticmethod
    def from01(x):
        return x * 2.0 - 1.0
