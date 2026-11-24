"""Text-conditioned SE2 (sphere encoder 2) for mel images.

The released class-conditional SE2 decoder takes a [B, 1, D] conditioning vector c (adaLN).
We replace the class embedding with c = base + MLP(CLAP_text_emb), base initialised to the
released null-class embedding and the MLP's last layer zero-initialised, so at step 0 the
model is exactly the released model run with the null class. A learned `null` vector is used
for dropped captions (classifier-free guidance).
"""
import argparse, json, os, os.path as osp, sys

import torch
import torch.nn as nn

SE2_REPO = "/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/sphere2-test"
if SE2_REPO not in sys.path:
    sys.path.insert(0, SE2_REPO)
from sphere.builder import build_model  # noqa: E402
from sphere.model import rotate_away_from  # noqa: E402

ROOT = "/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text"
RELEASED = f"{ROOT}/workspace/experiments/sphere2-large-imagenet-fd-lite-256px"


def released_args(**over):
    cfg = json.load(open(f"{RELEASED}/cfg.json"))
    cfg.update(over)
    return argparse.Namespace(**cfg)


class TextCond(nn.Module):
    def __init__(self, null_init: torch.Tensor, text_dim: int = 512):
        super().__init__()
        D = null_init.numel()
        self.base = nn.Parameter(null_init.clone().float())
        self.null = nn.Parameter(null_init.clone().float())
        self.mlp = nn.Sequential(nn.Linear(text_dim, D), nn.SiLU(), nn.Linear(D, D))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, temb: torch.Tensor, drop: torch.Tensor | None = None):
        c = self.base + self.mlp(temb.float())
        if drop is not None:
            c = torch.where(drop[:, None], self.null.expand_as(c), c)
        return c[:, None, :]  # [B, 1, D]


def _patch_decoder(dec):
    """decoder(..., cond_embed=None) -> uses dec._cond when set (so G1.forward can stay unchanged)."""
    orig = dec.forward
    dec._cond = None

    def fwd(x, y=None, alpha=None, cond_embed=None, **kw):
        if cond_embed is None and dec._cond is not None:
            cond_embed = dec._cond.to(x.dtype) if torch.is_floating_point(x) else dec._cond
        return orig(x, y, alpha=alpha, cond_embed=cond_embed, **kw)

    dec.forward = fwd


class TextSE2(nn.Module):
    def __init__(self, g1, text_dim=512):
        super().__init__()
        self.g1 = g1
        _patch_decoder(g1.decoder)
        null_id = g1.num_classes
        null_init = g1.decoder.y_embedder.embedding_table.weight[null_id].detach()
        self.tc = TextCond(null_init, text_dim)
        g1.decoder.y_embedder.requires_grad_(False)  # overridden by cond_embed -> never gets grads

    def null_y(self, B, device):
        return torch.full((B,), self.g1.num_classes, dtype=torch.long, device=device)

    def forward(self, x, temb, drop):
        self.g1.decoder._cond = self.tc(temb, drop)
        try:
            return self.g1(x, self.null_y(x.shape[0], x.device), ema_model=None)
        finally:
            self.g1.decoder._cond = None

    # ---------------- inference ----------------
    @torch.no_grad()
    def reconstruct(self, x, temb=None):
        """x in [-1, 1] -> [0, 1] (angle 0). temb None -> null caption."""
        g1, B = self.g1, x.shape[0]
        y = self.null_y(B, x.device)
        c = self.tc(temb, None) if temb is not None else self.tc.null.expand(B, -1)[:, None]
        z, alpha = g1.spherify(g1.encoder(x, y), 0.0)
        return (g1.decoder(z, y, alpha=alpha, cond_embed=c.to(z.dtype)).float() * 0.5 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def generate(self, temb, steps=2, init_angle=85.0, loop_angle=None, cfg_deg=0.0, noise=None):
        """Text -> image in [0, 1]. SE2 sampler: decode noise, then (encode, re-noise, decode) steps-1 times.
        cfg_deg > 0: angle CFG (rotate the cond latent away from an unconditional stream), as in G1.generate."""
        g1 = self.g1
        B, dev = temb.shape[0], temb.device
        y = self.null_y(B, dev)
        c = self.tc(temb, None)
        cu = self.tc.null.expand(B, -1)[:, None]
        e = noise if noise is not None else torch.randn(B, *g1.latent_shape[1:], device=dev)
        z = g1.f(e)
        alpha = g1._dec_angle(B, min(g1.max_angle_deg, init_angle), dev)
        dt = next(g1.decoder.parameters()).dtype
        x = g1.decoder(z, y, alpha=alpha, cond_embed=c)
        xu = g1.decoder(z, y, alpha=alpha, cond_embed=cu) if cfg_deg > 0 else None
        loop_angle = init_angle if loop_angle is None else loop_angle
        for s in range(1, steps):
            z = g1.encoder(x, y)
            if cfg_deg > 0:
                zu = g1.f(g1.encoder(xu, y))
                z = rotate_away_from(g1.f(z), zu, cfg_deg, g1.f)
            z, alpha = g1.spherify(z, min(g1.max_angle_deg, loop_angle))
            x = g1.decoder(z, y, alpha=alpha, cond_embed=c)
            if cfg_deg > 0 and s < steps - 1:
                zu, _ = g1.spherify(zu, min(g1.max_angle_deg, loop_angle))
                xu = g1.decoder(zu, y, alpha=alpha, cond_embed=cu)
        return (x.float() * 0.5 + 0.5).clamp(0, 1)


def build(for_train=True, load_released=True, **over):
    args = released_args(freeze_encoder=False, **over)
    g1 = build_model(args, for_train=for_train)
    if load_released:
        ck = torch.load(f"{RELEASED}/ckpt/ep0299.pth", map_location="cpu", weights_only=False)
        sd = ck.get("ema_model", ck["model"])  # start from the inference (EMA) weights
        sd = {k.replace("module.", "").replace("_checkpoint_wrapped_module.", "").replace("._orig_mod", ""): v
              for k, v in sd.items()}
        msg = g1.load_state_dict(sd, strict=False)
        print("[build] loaded released EMA weights; missing", len(msg.missing_keys), msg.missing_keys[:5],
              "unexpected", len(msg.unexpected_keys), msg.unexpected_keys[:5], flush=True)
        del ck
    return TextSE2(g1), args


def load_run(path, use_ema=True, device="cuda"):
    """Load a trained TextSE2 from a checkpoint written by train.py."""
    m, _ = build(for_train=False, load_released=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    msg = m.load_state_dict(sd, strict=False)
    print("[load_run]", path, "ema" if use_ema else "raw", "missing", len(msg.missing_keys), msg.missing_keys[:3],
          "unexpected", len(msg.unexpected_keys), flush=True)
    return m.to(device).eval(), ck
