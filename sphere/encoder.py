import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoConfig, AutoModel, SiglipModel

logger = logging.getLogger(__name__)


"""
classes
"""


class DinoV(nn.Module):
    """
    https://huggingface.co/collections/facebook/dinov2
    https://huggingface.co/collections/facebook/dinov3
    """

    def __init__(
        self,
        model_name: str = "dinov2-base",
        normalize: bool = True,
        freeze: bool = True,
        train_last_n_blocks: int = 0,
        train_norm_ls: bool = False,
        train_final_norm: bool = False,
        interpolate_num_tokens_to: int | None = None,
        interpolate_pos_embed: bool = False,
    ):
        super().__init__()
        if "/" not in model_name:
            model_name = f"facebook/{model_name}"
        model_version = "dinov3" if "dinov3" in model_name else "dinov2"
        self.encoder = AutoModel.from_pretrained(model_name)

        # turn off pos embed augmentation in case the .train() is called
        self.encoder.config.pos_embed_rescale = None
        self.encoder.config.pos_embed_shift = None
        self.encoder.config.pos_embed_jitter = None
        self.encoder.embeddings.mask_token = None

        # trainable adapter applied to the final patch tokens (the latent),
        # set from outside.
        self.out = nn.Identity()

        if model_version == "dinov2":
            blocks = self.encoder.encoder.layer
            final_norm = self.encoder.layernorm
            # last_hidden_state is [CLS, registers... (if any), patches...]
            self.num_prefix_tokens = 1 + getattr(
                self.encoder.config, "num_register_tokens", 0
            )
            self.encoder.embeddings.position_embeddings.requires_grad_(False)
        else:
            blocks = self.encoder.model.layer
            final_norm = self.encoder.norm
            # last_hidden_state is [CLS, registers..., patches...]
            self.num_prefix_tokens = 1 + self.encoder.config.num_register_tokens

        # submodules that stay in train mode when frozen=True; see train()
        self.frozen = freeze
        self.trainable_modules = []

        if freeze:
            self.encoder.eval().requires_grad_(False)

        # selectively unfreeze the last N blocks
        if train_last_n_blocks > 0:
            num_blocks = len(blocks)
            assert train_last_n_blocks <= num_blocks, (
                f"train_last_n_blocks={train_last_n_blocks} > "
                f"num_blocks={num_blocks}"
            )
            for block in blocks[-train_last_n_blocks:]:
                block.train().requires_grad_(True)
                self.trainable_modules.append(block)
            logger.info(
                f"{model_version}: training last {train_last_n_blocks}/{num_blocks} blocks"
            )

        # selectively unfreeze the norm + layer-scale params in every block
        if train_norm_ls:
            for block in blocks:
                for m in (
                    block.norm1,
                    block.norm2,
                    block.layer_scale1,
                    block.layer_scale2,
                ):
                    m.train().requires_grad_(True)
                    self.trainable_modules.append(m)
            logger.info(
                f"{model_version}: training norm1/norm2/layer_scale1/layer_scale2 "
                f"in all {len(blocks)} blocks"
            )

        if normalize:
            final_norm.elementwise_affine = False
            final_norm.weight = None
            final_norm.bias = None

        # selectively unfreeze the final norm
        if train_final_norm:
            assert not normalize, (
                "train_final_norm requires normalize=False so the final norm "
                "still has affine weight/bias to train"
            )
            final_norm.train().requires_grad_(True)
            self.trainable_modules.append(final_norm)
            logger.info(f"{model_version}: training the final norm")

        self.token_chns = self.encoder.config.hidden_size
        self.num_tokens = interpolate_num_tokens_to
        self.input_size = {
            "dinov2": 224,  # patch size = 14
            "dinov3": 256,  # patch size = 16
        }[model_version]
        self.interpolate_pos_embed = interpolate_pos_embed
        self.model_version = model_version

        self.register_buffer(
            "NORM_MEAN",
            torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "NORM_STD",
            torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        """
        keep the frozen backbone in eval mode.
        """
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
            for m in self.trainable_modules:
                m.train(mode)
        return self

    def _post(self, z: Tensor):
        z = z[:, self.num_prefix_tokens :]  # drop CLS (+ register) tokens
        if self.num_tokens is not None and z.shape[1] != self.num_tokens:
            z = self._interpolate_tokens(z, self.num_tokens)
        return z

    def _forward(self, x: torch.Tensor, extract_layer_idxs: list = []):
        """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py
        """
        res = self.encoder(
            pixel_values=x,
            output_hidden_states=len(extract_layer_idxs) > 0,
            interpolate_pos_encoding=self.interpolate_pos_embed,
        )
        self.last_cls = res.last_hidden_state[:, 0]
        z = self._post(res.last_hidden_state)  # [B, N, D]
        z = self.out(z)
        if len(extract_layer_idxs) > 0:
            # res.hidden_states = (embeddings, block_0, ..., block_{L-1}); shift
            # by +1 so idx i selects the output of block i.
            hidden_states = {
                i: self._post(res.hidden_states[i + 1]) for i in extract_layer_idxs
            }
            return z, hidden_states
        return z

    @staticmethod
    def _interpolate_tokens(z: Tensor, num_tokens: int):
        B, N, D = z.shape
        src = int(N**0.5)
        dst = int(num_tokens**0.5)
        assert src * src == N, f"{N} tokens is not a square grid"
        assert dst * dst == num_tokens, f"{num_tokens} is not a square grid"

        z = z.reshape(B, src, src, D).permute(0, 3, 1, 2)  # [B, D, src, src]
        z = F.interpolate(z, size=(dst, dst), mode="bicubic", align_corners=False)
        z = z.permute(0, 2, 3, 1).reshape(B, dst * dst, D)  # [B, num_tokens, D]
        return z

    def forward(
        self,
        x: Tensor,
        y: Tensor = None,
        *args,
        extract_layer_idxs: list = [],
        **kwargs,
    ):
        x = x.float() * 0.5 + 0.5  # [-1, 1] to [0, 1]
        if (x.shape[2] != self.input_size or x.shape[3] != self.input_size) and (
            not self.interpolate_pos_embed
        ):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear" if self.model_version == "dinov3" else "bicubic",
                antialias=True,
                align_corners=False,
            )
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.NORM_MEAN) / self.NORM_STD
        return self._forward(x, extract_layer_idxs=extract_layer_idxs)


class SigLIP2(nn.Module):
    """
    https://huggingface.co/collections/google/siglip2
    """

    def __init__(
        self,
        model_name: str = "siglip2-base-patch16-256",
        normalize: bool = True,
        freeze: bool = True,
        train_last_n_blocks: int = 0,
        train_norm_ls: bool = False,
        train_final_norm: bool = False,
        interpolate_num_tokens_to: int | None = None,
        interpolate_pos_embed: bool = False,
    ):
        super().__init__()
        if "/" not in model_name:
            model_name = f"google/{model_name}"

        self.encoder = SiglipModel.from_pretrained(model_name).vision_model
        self.encoder.head = nn.Identity()  # drop the attention-pooling head

        # trainable adapter applied to the final patch tokens (the latent),
        # set from outside.
        self.out = nn.Identity()

        blocks = self.encoder.encoder.layers
        self.encoder.embeddings.position_embedding.requires_grad_(False)

        # submodules that stay in train mode when frozen=True; see train()
        self.frozen = freeze
        self.trainable_modules = []

        if freeze:
            self.encoder.eval().requires_grad_(False)

        # selectively unfreeze the last N blocks
        if train_last_n_blocks > 0:
            num_blocks = len(blocks)
            assert train_last_n_blocks <= num_blocks, (
                f"train_last_n_blocks={train_last_n_blocks} > "
                f"num_blocks={num_blocks}"
            )
            for block in blocks[-train_last_n_blocks:]:
                block.train().requires_grad_(True)
                self.trainable_modules.append(block)
            logger.info(
                f"siglip2: training last {train_last_n_blocks}/{num_blocks} blocks"
            )

        # selectively unfreeze the norm params in every block
        if train_norm_ls:
            for block in blocks:
                for m in (block.layer_norm1, block.layer_norm2):
                    m.train().requires_grad_(True)
                    self.trainable_modules.append(m)
            logger.info(
                f"siglip2: training norm1/norm2 in all " f"{len(blocks)} blocks"
            )

        if normalize:
            self.encoder.post_layernorm.elementwise_affine = False
            self.encoder.post_layernorm.weight = None
            self.encoder.post_layernorm.bias = None

        # selectively unfreeze the final norm (applied to the patch tokens)
        if train_final_norm:
            assert not normalize, (
                "train_final_norm requires normalize=False so the final norm "
                "still has affine weight/bias to train"
            )
            self.encoder.post_layernorm.train().requires_grad_(True)
            self.trainable_modules.append(self.encoder.post_layernorm)
            logger.info("siglip2: training the final norm (encoder.post_layernorm)")

        self.token_chns = self.encoder.config.hidden_size
        self.num_tokens = interpolate_num_tokens_to
        self.input_size = self.encoder.config.image_size
        self.interpolate_pos_embed = interpolate_pos_embed

        self.register_buffer(
            "NORM_MEAN", torch.full((1, 3, 1, 1), 0.5), persistent=False
        )
        self.register_buffer(
            "NORM_STD", torch.full((1, 3, 1, 1), 0.5), persistent=False
        )

    def train(self, mode: bool = True):
        """
        keep the frozen backbone in eval mode
        """
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
            for m in self.trainable_modules:
                m.train(mode)
        return self

    def _post(self, z: Tensor):
        if self.num_tokens is not None and z.shape[1] != self.num_tokens:
            z = DinoV._interpolate_tokens(z, self.num_tokens)
        return z

    def _forward(self, x: torch.Tensor, extract_layer_idxs: list = []):
        """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py#L524
        """
        res = self.encoder(
            pixel_values=x,
            output_hidden_states=len(extract_layer_idxs) > 0,
            interpolate_pos_encoding=self.interpolate_pos_embed,
        )
        self.last_cls = None  # siglip2 has no CLS token
        z = self._post(res.last_hidden_state)  # [B, N, D]
        z = self.out(z)
        if len(extract_layer_idxs) > 0:
            # res.hidden_states = (embeddings, block_0, ..., block_{L-1}); shift
            # by +1 so idx i selects the output of block i (sphere convention).
            hidden_states = {
                i: self._post(res.hidden_states[i + 1]) for i in extract_layer_idxs
            }
            return z, hidden_states
        return z

    def forward(
        self,
        x: Tensor,
        y: Tensor = None,
        *args,
        extract_layer_idxs: list = [],
        **kwargs,
    ):
        x = x.float() * 0.5 + 0.5  # [-1, 1] to [0, 1]
        if (x.shape[2] != self.input_size or x.shape[3] != self.input_size) and (
            not self.interpolate_pos_embed
        ):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.NORM_MEAN) / self.NORM_STD
        return self._forward(x, extract_layer_idxs=extract_layer_idxs)


"""
functions
"""


def get_token_channels(model_name: str):
    if "/" not in model_name:
        prefix = "google" if "siglip" in model_name else "facebook"
        model_name = f"{prefix}/{model_name}"
    config = AutoConfig.from_pretrained(model_name)
    # siglip configs nest the vision tower under vision_config
    config = getattr(config, "vision_config", config)
    return config.hidden_size
