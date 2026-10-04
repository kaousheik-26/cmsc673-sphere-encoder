from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


"""
functions
"""


def is_dinov3_convnext(model_name: str):
    return str(model_name).startswith("facebook/dinov3-convnext")


def build_convnext_featurizer(
    model_name: str = "timm/convnextv2_tiny.fcmae",
    stages: list | None = None,
    img_size: int = 256,
    device: torch.device = "cpu",
    return_logits: bool = False,
    return_final: bool = False,
):
    if is_dinov3_convnext(model_name):
        assert not return_logits, (
            f"{model_name} carries no classifier head, so it has no logits to "
            "return; the logits alignment needs a timm ConvNeXt-V2 classifier "
            "tag (e.g. timm/convnextv2_nano.fcmae_ft_in1k)"
        )
        return DinoV3ConvNext(
            model_name=model_name,
            stages=stages,
            img_size=img_size,
            device=device,
            return_final=return_final,
        )
    return ConvNextV2(
        model_name=model_name,
        stages=stages,
        img_size=img_size,
        device=device,
        return_logits=return_logits,
        return_final=return_final,
    )


"""
classes
"""


class _ConvNextBase(nn.Module):
    """
    Stage-token featurizer over a frozen ConvNeXt backbone.

        forward -> [ [B, N_j, D_j] ] per stage in `stages`, ascending.

    With `return_logits` (ConvNextV2 only) and/or `return_final`, forward
    hands back a tuple instead, the second entry being a dict of head-side
    tensors:

        forward -> ([ [B, N_j, D_j] ], aux)
        aux["logits"]       : [B, C]      classifier output   (return_logits)
        aux["final_tokens"] : [B, N_3, D] last stage's map     (return_final)
        aux["final_pooled"] : [B, D]      what the head reads  (return_final)

    "final_pooled" is the global vector right before the classifier: pooled +
    normed for the timm ConvNeXt-V2 tags (its `pre_logits`), pooled + the
    trunk LayerNorm for the DINOv3 ConvNeXts (its `pooler_output`).

    Widths and token counts DIFFER per stage (stride 4/8/16/32), so
    `embed_dims` / `num_tokens_list` carry them per entry. Without the head
    outputs the backbone runs only as deep as the deepest selected stage; with
    them it has to run to the end, since that is where they come from.
    """

    return_logits = False
    return_final = False
    num_logits = None  # C, set when return_logits is on
    final_dim = None  # D of the last stage, set when return_final is on

    def _resolve_stages(self, stages: list | None, model_name=""):
        if stages is None:
            stages = [1, 3]
        out = sorted({int(s) for s in stages})
        assert out, f"stages={stages} resolved to nothing for {model_name}"
        return out

    @staticmethod
    def _tokens(t: torch.Tensor):
        """[B, D, H, W] -> [B, N, D], the layout every consumer expects."""
        return t.float().flatten(2).transpose(1, 2).contiguous()

    def _finalize(
        self,
        model_name,
        stages,
        img_size,
        backbone,
        device,
        mean,
        std,
        reductions,
        embed_dims,
    ):
        for r in reductions:
            assert img_size % r == 0, (
                f"img_size {img_size} not divisible by stage stride {r} "
                f"({model_name} stages {stages})"
            )
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

        self.proxy = (backbone.to(device, memory_format=torch.channels_last),)
        self.model_name = model_name
        self.layers = stages
        self.img_size = img_size

        # per-stage shapes (ascending stage order, matching forward's output)
        self.embed_dims = list(embed_dims)
        self.num_tokens_list = [(img_size // r) ** 2 for r in reductions]

    def _stage_feats(self, x: torch.Tensor):
        """Returns (list of [B, D, H, W] stage maps ascending, aux dict of
        head-side tensors -- empty when neither return_* flag is on)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        x = x.float() * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(
            device_type=x.device.type,
            dtype=torch.bfloat16,
            enabled=x.device.type != "cpu",
        ):
            stage_feats, aux = self._stage_feats(x)
        feats = [self._tokens(f) for f in stage_feats]  # [B,D,H,W] -> [B,N,D]
        if not (self.return_logits or self.return_final):
            return feats
        return feats, {k: v.float() for k, v in aux.items()}


class ConvNextV2(_ConvNextBase):

    def __init__(
        self,
        model_name: str = "timm/convnextv2_nano.fcmae_ft_in1k",
        stages: list | None = None,
        img_size: int = 256,
        device: torch.device = "cpu",
        return_logits: bool = False,
        return_final: bool = False,
    ):
        super().__init__()
        import timm

        stages = self._resolve_stages(stages, model_name)
        if return_logits or return_final:
            # The classifier head only exists on the FULL model: features_only
            # prunes every stage past max(out_indices) and drops the head. The
            # full model's forward_intermediates returns the same raw stage
            # outputs features_only does, plus the final map the head reads.
            model = timm.create_model(model_name, pretrained=True)
            assert not return_logits or model.num_classes > 0, (
                f"{model_name} has no classifier (num_classes=0), so there are "
                "no logits to align; use a classifier tag such as "
                "timm/convnextv2_nano.fcmae_ft_in1k"
            )
            info = model.feature_info  # per-stage dicts on a full model
            assert stages[-1] < len(info), (
                f"stages={stages} out of range for {model_name} "
                f"({len(info)} stages)"
            )
            reductions = [info[s]["reduction"] for s in stages]
            embed_dims = [info[s]["num_chs"] for s in stages]
            if return_logits:
                self.num_logits = int(model.num_classes)
            if return_final:
                self.final_dim = int(info[-1]["num_chs"])
        else:
            model = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
                out_indices=tuple(stages),
            )
            reductions = model.feature_info.reduction()  # stride per stage
            embed_dims = model.feature_info.channels()
        self.return_logits = bool(return_logits)
        self.return_final = bool(return_final)
        model.eval().requires_grad_(False)

        cfg = getattr(model, "pretrained_cfg", None) or {}
        self._finalize(
            model_name=model_name,
            stages=stages,
            img_size=img_size,
            backbone=model,
            device=device,
            mean=cfg.get("mean", IMAGENET_MEAN),
            std=cfg.get("std", IMAGENET_STD),
            reductions=reductions,
            embed_dims=embed_dims,
        )

    def _stage_feats(self, x):
        model = self.proxy[0]
        if not (self.return_logits or self.return_final):
            return model(x), {}
        # `indices` selects which stages to keep; they come back in ascending
        # stage order, matching self.layers. `final` is the last stage's map,
        # i.e. what the head pools -- the head itself is a pool + norm + one
        # linear on a [B, D] vector, so running it is free next to the backbone.
        final, feats = model.forward_intermediates(x, indices=list(self.layers))
        aux = {}
        if self.return_final:
            aux["final_tokens"] = self._tokens(final)
            # pool + norm (+ flatten), i.e. the vector the classifier reads
            aux["final_pooled"] = model.forward_head(final, pre_logits=True)
        if self.return_logits:
            aux["logits"] = model.forward_head(final, pre_logits=False)
        return feats, aux


class DinoV3ConvNext(_ConvNextBase):

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        stages: list | None = None,
        img_size: int = 256,
        device: torch.device = "cpu",
        return_final: bool = False,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        stages = self._resolve_stages(stages, model_name)
        hf = AutoModel.from_pretrained(model_name)
        hidden_sizes = list(hf.config.hidden_sizes)
        assert stages[0] >= 0 and stages[-1] < len(hidden_sizes), (
            f"stages={stages} out of range for {model_name} "
            f"({len(hidden_sizes)} stages)"
        )
        self.return_final = bool(return_final)
        all_stages = list(hf.model.stages)
        keep = len(all_stages) if self.return_final else stages[-1] + 1
        backbone = nn.ModuleList(all_stages[:keep])
        if self.return_final:
            # the model's own global head: avg-pool the last map, then the
            # trunk LayerNorm shared with the patch tokens (`pooler_output`)
            backbone.append(nn.ModuleDict({"pool": hf.pool, "norm": hf.layer_norm}))
            self.final_dim = int(hidden_sizes[-1])
        backbone.eval().requires_grad_(False)

        try:
            proc = AutoImageProcessor.from_pretrained(model_name)
            mean_v, std_v = tuple(proc.image_mean), tuple(proc.image_std)
        except Exception:
            mean_v, std_v = IMAGENET_MEAN, IMAGENET_STD

        self._finalize(
            model_name=model_name,
            stages=stages,
            img_size=img_size,
            backbone=backbone,
            device=device,
            mean=mean_v,
            std=std_v,
            # ConvNeXt geometry: stride-4 patchify stem inside stage 0, then /2
            # per stage. Verified against measured token counts.
            reductions=[4 * 2**s for s in stages],
            embed_dims=[hidden_sizes[s] for s in stages],
        )

    def _stage_feats(self, x):
        """
        https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py
        """
        mods = list(self.proxy[0])
        head = mods.pop() if self.return_final else None
        got = {}
        want = set(self.layers)
        for i, stage in enumerate(mods):
            x = stage(x)
            if i in want:
                got[i] = x  # [B, D_i, H_i, W_i]
        aux = {}
        if head is not None:
            # HF applies the trunk LayerNorm to [pooled; patches] together, so
            # both head-side tensors go through it here
            pooled = head["pool"](x).flatten(1)  # [B, D]
            aux["final_pooled"] = head["norm"](pooled)
            aux["final_tokens"] = head["norm"](self._tokens(x))
        return [got[i] for i in self.layers], aux
