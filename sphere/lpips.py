import hashlib
import os
import logging
from collections import namedtuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

logger = logging.getLogger(__name__)

"""
variables
"""

# fmt:off
_IMAGENET_MEAN = [ 0.485,  0.456,  0.406]
_IMAGENET_STD  = [ 0.229,  0.224,  0.225]
_LPIPS_MEAN    = [-0.030, -0.088, -0.188]
_LPIPS_STD     = [ 0.458,  0.448,  0.450]

URL_MAP  = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP  = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}
# fmt:on


"""
functions
"""


def download(url: str, local_path: str, chunk_size: int = 1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        logger.info(
            "downloading {} model from {} to {}".format(name, URL_MAP[name], path)
        )
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


"""
classes
"""


class vgg16(nn.Module):
    def __init__(
        self,
        requires_grad: bool = False,
        pretrained: bool = True,
    ):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(
            pretrained=pretrained
        ).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        # build feature slices
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "vgg_outputs",
            ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"],
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):
    def __init__(
        self,
        ckpt_pth="work_dirs/ckpts/lpips",
        use_dropout=True,
        chns_range=(0, 5),
        image_size: int | None = None,
        ptdtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        assert 0 <= chns_range[0] < chns_range[1] <= len(self.chns)
        self.image_size = image_size
        self.lidx, self.ridx = chns_range
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.load_from_pretrained(ckpt_pth=ckpt_pth)
        for param in self.parameters():
            param.requires_grad = False

        self._data_range_checked = False
        self._ptdtype = ptdtype

    def load_from_pretrained(self, ckpt_pth="work_dirs/ckpts/lpips", name="vgg_lpips"):
        if ckpt_pth is None:
            raise ValueError("no pretrained weights found for LPIPS loss.")
        ckpt = get_ckpt_path(name, ckpt_pth, check=True)
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        logger.info("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input: torch.Tensor, target: torch.Tensor, tile_target: int = 1):
        assert target.min() >= -1.0 and target.max() <= 1.0

        if self.image_size is not None:
            input = F.interpolate(
                input,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            target = F.interpolate(
                target,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        with torch.amp.autocast(
            device_type=input.device.type,
            dtype=self._ptdtype,
            enabled=self._ptdtype in [torch.bfloat16],
        ):
            inps = self.net(self.scaling_layer(input))
            tgts = self.net(self.scaling_layer(target))

        inps = [t.float() for t in inps]
        tgts = [t.float() for t in tgts]

        lins = self.lins[self.lidx : self.ridx]
        chns = self.chns[self.lidx : self.ridx]
        inps = inps[self.lidx : self.ridx]
        tgts = tgts[self.lidx : self.ridx]

        vals = []
        for k in range(len(chns)):
            f0 = inps[k]
            f1 = tgts[k]

            f0 = f0 * torch.rsqrt(torch.sum(f0**2, dim=1, keepdim=True) + 1e-10)
            f1 = f1 * torch.rsqrt(torch.sum(f1**2, dim=1, keepdim=True) + 1e-10)

            if tile_target > 1:
                # matches torch.cat([target] * tile_target, dim=0)
                f1 = f1.repeat(tile_target, 1, 1, 1)

            diff = lins[k].model((f0 - f1) ** 2)  # [B, 1, H, W]
            diff = diff.mean([2, 3], keepdim=True)

            vals.append(diff)
        return torch.stack(vals, dim=0).sum(dim=0)


class ConvNextAug(nn.Module):

    def __init__(
        self,
        flip_prob: float = 0.5,
        translation_ratio: float = 0.0625,
        scale_min: float = 0.85,
        scale_max: float = 1.0,
    ):
        super().__init__()
        assert 0.0 <= flip_prob <= 1.0, f"{flip_prob=}"
        assert 0.0 <= translation_ratio < 0.5, f"{translation_ratio=}"
        # capped at 1.0: scale > 1 would read outside the frame and pull
        # reflected border content in, which adds nothing here (zoom-in crops
        # already break the fixed-kernel overfit) and would make the two sides
        # depend on the padding mode.
        assert 0.0 < scale_min <= scale_max <= 1.0, f"{scale_min=} {scale_max=}"
        self.flip_prob = float(flip_prob)
        self.translation_ratio = float(translation_ratio)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    @classmethod
    def from_spec(cls, spec: str):
        """
        Parse "<flip_prob>-<translation_ratio>-<scale_min>-<scale_max>", e.g.
        "0.5-0.0625-0.85-1.0". Same dash-separated-floats shape as the
        `model_name` weights and `weight_anchor_cutoff_params`.
        """
        parts = str(spec).split("-")
        assert len(parts) == 4, (
            f"convnext_aug spec {spec!r} needs 4 dash-separated fields: "
            f"<flip_prob>-<translation_ratio>-<scale_min>-<scale_max>"
        )
        f, t, s0, s1 = (float(v) for v in parts)
        return cls(flip_prob=f, translation_ratio=t, scale_min=s0, scale_max=s1)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"flip_prob={self.flip_prob:g}, "
            f"translation_ratio={self.translation_ratio:g}, "
            f"scale=[{self.scale_min:g}, {self.scale_max:g}])"
        )

    __repr__ = __str__

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, dtype: torch.dtype):
        """Per-sample affine params as a [n, 2, 3] theta."""
        # F.affine_grid's theta maps OUTPUT normalized coords -> INPUT
        # normalized coords, so `s` is the size of the sampled WINDOW: s < 1
        # reads a sub-window, i.e. zooms in / crops.
        s = torch.empty(n, device=device, dtype=dtype).uniform_(
            self.scale_min, self.scale_max
        )
        # normalized coords span 2.0 across the image, so shifting by `ratio`
        # of the image dimension is 2 * ratio in these units.
        r = 2.0 * self.translation_ratio
        tx = torch.empty(n, device=device, dtype=dtype).uniform_(-r, r)
        ty = torch.empty(n, device=device, dtype=dtype).uniform_(-r, r)
        flip = torch.where(torch.rand(n, device=device) < self.flip_prob, -1.0, 1.0).to(
            dtype
        )

        theta = torch.zeros(n, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = s * flip  # x scale; negated => horizontal flip
        theta[:, 1, 1] = s  # y scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        return theta

    def apply(self, x: torch.Tensor, theta: torch.Tensor, tile: int = 1):
        """Resample `x` under `theta`. Differentiable w.r.t. `x`."""
        if tile > 1:
            # matches tgt_logits.repeat(tile_target, 1): row i of the tiled
            # batch pairs with unique row i % n, so repeating theta the same way
            # gives every generated row its own target's transform.
            theta = theta.repeat(tile, 1, 1)
        assert theta.shape[0] == x.shape[0], (
            f"theta batch {theta.shape[0]} != x batch {x.shape[0]}; the draws "
            f"are per unique target and must be tiled to match the input"
        )
        grid = F.affine_grid(theta.to(x.dtype), list(x.shape), align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_s",
        ckpt_path: str = "",
        perceptual_loss_chns_range: tuple[int, int] = (0, 5),
        image_size: int | None = None,
        ptdtype: torch.dtype = torch.float32,
        convnext_aug: str | None = None,
        convnext_loss_type: str = "mse",
        convnext_loss_reduction: str = "mean",
    ):
        super().__init__()
        assert "lpips" in model_name or "convnext_s" in model_name
        convnext_loss_type = str(convnext_loss_type).lower()
        assert convnext_loss_type in (
            "mse",
            "cosine",
        ), f"unknown ConvNeXt loss type: {convnext_loss_type!r}"
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None
        self._data_range_checked = False
        self._ptdtype = ptdtype
        self.convnext_loss_type = convnext_loss_type
        assert convnext_loss_reduction in ["mean", "none"]
        self.convnext_loss_reduction = convnext_loss_reduction

        # per-component (unweighted, detached) scalars of the last forward, so
        # the caller can log lpips and convnext separately.
        self.log_dict = {}

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS(
                ckpt_pth=ckpt_path,
                chns_range=perceptual_loss_chns_range,
                ptdtype=ptdtype,
                image_size=image_size,
            ).eval()

        if "convnext_s" in model_name:
            self.convnext = torchvision.models.convnext_small(
                weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            ).eval()

        # Randomized input transform for the logits branch ONLY -- the LPIPS
        # branch is a dense spatial comparison and does not have the
        # fixed-kernel overfit this guards against. None/"" keeps the previous
        # behaviour (one deterministic resize) bit-for-bit.
        self.convnext_aug = None
        if self.convnext is not None and convnext_aug:
            self.convnext_aug = ConvNextAug.from_spec(convnext_aug)
            logger.info(f"convnext logits branch aug - {self.convnext_aug}")

        if self.convnext is not None:
            logger.info(f"convnext logits loss - type: {self.convnext_loss_type}")

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split("-")[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(
                loss_config[0]
            ), float(loss_config[1])
            logger.info(
                f"loss weights - lpips: {self.loss_weight_lpips}, convnext: {self.loss_weight_convnext}"
            )

        if self.loss_weight_convnext is None:
            self.loss_weight_convnext = 1.0

        if self.loss_weight_lpips is None:
            self.loss_weight_lpips = 1.0

        self.register_buffer(
            "imagenet_mean",
            torch.Tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.Tensor(_IMAGENET_STD).reshape(1, 3, 1, 1),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor, tile_target: int = 1):
        """
        tile_target > 1 means `target` holds the unique images of a batch the
        caller would otherwise have tiled to match `input` (see LPIPS.forward);
        both backbones then run on the unique targets only.
        """
        assert (
            input.shape[0] == target.shape[0] * tile_target
            and input.shape[1:] == target.shape[1:]
        ), f"{input.shape=} != {target.shape=} x {tile_target=}"

        if not self._data_range_checked:
            assert (
                target.min() >= 0.0 and target.max() <= 1.0
            ), f"{target.min()=} ~ {target.max()=}. reminder to normalize input and target to [0, 1]."
            self._data_range_checked = True

        self.eval()
        inp, tgt = input, target
        loss = 0.0
        num_losses = 0.0
        self.log_dict = {}

        if self.lpips is not None:
            # [0, 1] -> [-1, 1]
            lpips_loss = self.lpips(inp * 2 - 1, tgt * 2 - 1, tile_target=tile_target)
            loss += self.loss_weight_lpips * lpips_loss
            num_losses += self.loss_weight_lpips
            self.log_dict[f"lpips-{self.loss_weight_lpips:.1f}"] = (
                lpips_loss.detach().mean()
            )

        if self.convnext is not None:
            inp = F.interpolate(inp, size=224, mode="bilinear", antialias=True)
            tgt = F.interpolate(tgt, size=224, mode="bilinear", antialias=True)

            if self.convnext_aug is not None:
                theta = self.convnext_aug.sample(tgt.shape[0], inp.device, inp.dtype)
                inp = self.convnext_aug.apply(inp, theta, tile=tile_target)
                tgt = self.convnext_aug.apply(tgt, theta)

            with torch.amp.autocast(
                device_type=input.device.type,
                dtype=self._ptdtype,
                enabled=self._ptdtype in [torch.bfloat16],
            ):
                inp_logits = self.convnext(
                    (inp - self.imagenet_mean) / self.imagenet_std
                )
                tgt_logits = self.convnext(
                    (tgt - self.imagenet_mean) / self.imagenet_std
                )
            inp_logits, tgt_logits = inp_logits.float(), tgt_logits.float()

            if tile_target > 1:
                tgt_logits = tgt_logits.repeat(tile_target, 1)

            if self.convnext_loss_type == "mse":
                convnext_loss = F.mse_loss(inp_logits, tgt_logits, reduction="none")
                convnext_loss = convnext_loss.mean(dim=1)  # [B, 1000] -> [B]

            if self.convnext_loss_type == "cosine":
                inp_centered = inp_logits - inp_logits.mean(dim=1, keepdim=True)
                tgt_centered = tgt_logits - tgt_logits.mean(dim=1, keepdim=True)
                convnext_loss = 1.0 - F.cosine_similarity(
                    inp_centered, tgt_centered, dim=1, eps=1e-6
                )

            if self.convnext_loss_reduction == "none":
                convnext_loss = convnext_loss.view_as(
                    lpips_loss
                )  # [B] -> [B, 1, 1, 1] to match LPIPS output shape
            if self.convnext_loss_reduction == "mean":
                convnext_loss = convnext_loss.mean()  # [B] -> scalar

            loss += self.loss_weight_convnext * convnext_loss
            num_losses += self.loss_weight_convnext
            self.log_dict[f"convnext-{self.loss_weight_convnext:.1f}"] = (
                convnext_loss.detach().mean()
            )

        return loss / num_losses


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor(_LPIPS_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("scale", torch.Tensor(_LPIPS_STD).reshape(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)
