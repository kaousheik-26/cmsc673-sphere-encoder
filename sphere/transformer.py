from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from sphere.layers import (
    AngleEmbedder,
    LabelEmbedder,
    ModulatedLinear,
    Block,
    Contiguous,
    ScaledTanh,
    get_2d_sincos_pos_embed,
    get_rope_tensor,
    vector_rms_norm,
)

# fmt: off
SIZE_DICT = {
    "small"  :  {"width": 512,  "layers": 8,  "heads": 8,  "in_context_start": 2 },
    "base"   :  {"width": 768,  "layers": 12, "heads": 12, "in_context_start": 4 },
    "large"  :  {"width": 1024, "layers": 24, "heads": 16, "in_context_start": 8 },
    "xlarge" :  {"width": 1152, "layers": 28, "heads": 16, "in_context_start": 8 },
    "huge"   :  {"width": 1280, "layers": 32, "heads": 16, "in_context_start": 10},
    "giant"  :  {"width": 1664, "layers": 40, "heads": 16, "in_context_start": 12},
}
# fmt: on


class Transformer(nn.Module):

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base",
        model_type: str = "encoder",
        token_chns: int = 16,
        num_classes: int = 0,
        in_context_size: int = 0,
        pixel_head_type: str = "linear",
        pixel_head_use_tanh: bool = False,
        use_angle_cond: bool = False,
        angle_cond_max_deg: float = 90.0,
        angle_cond_mode: str = "adaln",
        halve_model_size: bool = False,
        spherify_model: bool = False,
        drop_residual_path_prob: float = 0.0,
        sdpa_mode: Optional[str] = None,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        assert model_type in ["encoder", "decoder"]
        assert model_size in SIZE_DICT

        self.model_type = model_type
        self.model_size = model_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.token_chns = token_chns
        self.grid_size = self.input_size // self.patch_size

        # make true to spherify the output of each transformer block
        self.spherify_model = spherify_model
        spherify_fn = vector_rms_norm
        self.f = partial(spherify_fn, zero_mean=False)

        self.num_tokens = self.grid_size**2

        params = SIZE_DICT[self.model_size]
        self.hidden_size = params["width"]
        self.num_layers = params["layers"]
        if halve_model_size:
            self.num_layers = self.num_layers // 2
        self.num_heads = params["heads"]
        self.inctx_start = params["in_context_start"]
        self.inctx_size = in_context_size

        self.latent_shape = (1, self.num_tokens, self.token_chns)
        self.num_classes = num_classes

        # input embed
        if model_type == "encoder":
            self.x_embedder = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.hidden_size,
                    self.patch_size,
                    self.patch_size,
                    bias=False,
                ),
                Rearrange("b c h w -> b (h w) c", h=self.grid_size, w=self.grid_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                Contiguous(),
            )

        else:
            # decoder input embed
            self.x_embedder = nn.Linear(self.token_chns, self.hidden_size)

        # pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=False
        )

        # rope
        rope = get_rope_tensor(
            dim=self.hidden_size // self.num_heads,
            seq_h=self.grid_size,
            seq_w=self.grid_size,
        ).unsqueeze(0)
        self.register_buffer("rope", rope, persistent=False)

        # class embed
        self.y_embedder = (
            LabelEmbedder(self.num_classes, self.hidden_size, 0.1)
            if self.num_classes > 0
            else None
        )

        # noise-angle embed (decoder only)
        self.use_angle_cond = use_angle_cond
        # How the angle reaches the trunk:
        #   "adaln" : as a term in `c`, i.e. per-block scale/shift (DiT-style).
        #   "token" : added straight onto the token sequence next to pos_embed,
        #             broadcast over the token axis.
        #   "both"  : both paths.
        assert angle_cond_mode in ("adaln", "token", "both"), angle_cond_mode
        self.angle_cond_mode = angle_cond_mode
        self.angle_cond_adaln = use_angle_cond and angle_cond_mode in ("adaln", "both")
        self.angle_cond_token = use_angle_cond and angle_cond_mode in ("token", "both")
        self.alpha_embedder = None
        if self.use_angle_cond:
            assert model_type == "decoder", (
                "angle conditioning is decoder-only; the encoder's inputs "
                "(real images, re-encoded generations) have no angle"
            )
            self.alpha_embedder = AngleEmbedder(
                self.hidden_size,
                max_angle_deg=angle_cond_max_deg,
            )

        # use adaln
        self.use_modulation = self.y_embedder is not None or self.angle_cond_adaln

        # in-context embed (can be used for unconditional generation)
        self.use_inctx = self.inctx_size > 0
        if self.use_inctx:
            inctx_rope = get_rope_tensor(
                dim=self.hidden_size // self.num_heads,
                seq_h=self.grid_size,
                seq_w=self.grid_size,
                pad_size=self.inctx_size,
            ).unsqueeze(0)
            self.register_buffer("inctx_rope", inctx_rope, persistent=False)

            self.inctx_pos_embed = nn.Parameter(
                torch.zeros(1, self.inctx_size, self.hidden_size), requires_grad=True
            )

            if not self.use_modulation:
                # for unconditional generation like register tokens
                self.inctx_embed = nn.Parameter(
                    torch.zeros(1, self.inctx_size, self.hidden_size),
                    requires_grad=True,
                )

        # transformer
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    use_modulation=self.use_modulation,
                    qk_norm=use_qk_norm,
                    drop_residual_path_prob=drop_residual_path_prob,
                    sdpa_mode=sdpa_mode,
                )
                for _ in range(self.num_layers)
            ]
        )

        # pred head
        if model_type == "encoder":
            # encoder latent head
            self.ffn = ModulatedLinear(
                self.hidden_size, self.token_chns, use_modulation=self.use_modulation
            )
            self.out = nn.Identity()

        elif model_type == "decoder":
            # decoder pixel head
            self.pixel_head_use_tanh = pixel_head_use_tanh
            self.pixel_head_type = pixel_head_type.lower()
            assert self.pixel_head_type in ["linear", "linear+conv", "conv"]

            if self.pixel_head_type == "conv":
                intermediate_chns = 32
            else:
                intermediate_chns = self.patch_size**2 * 3
            self.ffn = ModulatedLinear(
                self.hidden_size, intermediate_chns, use_modulation=self.use_modulation
            )

            if self.pixel_head_type == "linear":
                layers = [
                    Rearrange(
                        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                        h=self.grid_size,
                        w=self.grid_size,
                        p1=self.patch_size,
                        p2=self.patch_size,
                    ),
                    Contiguous(),
                ]

            elif self.pixel_head_type == "linear+conv":
                layers = [
                    Rearrange(
                        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                        h=self.grid_size,
                        w=self.grid_size,
                        p1=self.patch_size,
                        p2=self.patch_size,
                    ),
                    Contiguous(),
                    nn.Conv2d(
                        3, 3, stride=1, kernel_size=3, padding=1, padding_mode="reflect"
                    ),
                ]

            elif self.pixel_head_type == "conv":
                layers = [
                    Rearrange(
                        "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
                    ),
                    Contiguous(),
                    nn.Conv2d(
                        intermediate_chns,
                        intermediate_chns,
                        kernel_size=3,
                        padding=1,
                        padding_mode="reflect",
                    ),
                    nn.SiLU(),
                    nn.ConvTranspose2d(
                        in_channels=intermediate_chns,
                        out_channels=3,
                        kernel_size=self.patch_size + 2,
                        stride=self.patch_size,
                        padding=1,
                    ),
                ]

            if self.pixel_head_use_tanh:
                layers.append(ScaledTanh())
            self.out = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        # init transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # init pos embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_tokens**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # init input embed
        if self.model_type == "encoder":
            w1 = self.x_embedder[0].weight.data
            nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))

            w2 = self.x_embedder[2].weight.data
            nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
            nn.init.constant_(self.x_embedder[2].bias, 0)

        if self.model_type == "decoder":
            proj = self.x_embedder
            nn.init.normal_(proj.weight, std=proj.in_features**-0.5)
            nn.init.constant_(proj.bias, 0)

        # init class embed
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # init angle embed (must run after the blanket _basic_init above)
        if self.alpha_embedder is not None:
            self.alpha_embedder.initialize_weights(
                zero_out=self.y_embedder is not None or self.angle_cond_token
            )

        # zero-out adaLN modulation layers in transformer
        for block in self.blocks:
            if not block.use_modulation:
                continue
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out adaLN modulation layers in ffn
        if self.use_modulation:
            nn.init.constant_(self.ffn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.ffn.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.ffn.linear.weight, 0)
        nn.init.constant_(self.ffn.linear.bias, 0)

        if self.model_type == "decoder":
            # indices are relative to the end; the optional final tanh shifts them
            off = 1 if self.pixel_head_use_tanh else 0
            if self.pixel_head_type == "conv":
                first_conv = self.out[-3 - off]
                w = first_conv.weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1), gain=0.01)
                nn.init.constant_(first_conv.bias, 0)

            # pure linear head has no conv in `out`; ffn (already zero-init'd) is the head
            if self.pixel_head_type in ["linear+conv", "conv"]:
                last_conv = self.out[-1 - off]
                w = last_conv.weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1), gain=0.01)
                nn.init.constant_(last_conv.bias, 0)

        # init in-context embed
        if self.use_inctx:
            nn.init.normal_(self.inctx_pos_embed, std=0.02)
            if not self.use_modulation:
                nn.init.normal_(self.inctx_embed, std=0.02)

    """
    module forward
    """

    def forward(self, x, y=None, alpha=None, cond_embed=None, extract_layer_idxs=[]):
        """
        x: input images [B, C, H, W] or latents [B, L, D]
        y: class tokens [B] for condition
        alpha: per-sample noise angle in RADIANS (decoder, angle cond only)
        """
        extracted_hidden_states = {}

        B = x.shape[0]
        c = self.y_embedder(y, self.training) if self.y_embedder else None  # [B, 1, D]

        if cond_embed is not None:
            c = cond_embed

        a = None
        if self.use_angle_cond:
            assert alpha is not None, (
                "angle conditioning is enabled but no alpha was passed to the "
                "decoder; every decode site must state its noise angle"
            )
            a = self.alpha_embedder(alpha)  # [B, 1, D]
            if self.angle_cond_adaln:
                c = a if c is None else c + a.to(c.dtype)

        x = self.x_embedder(x)
        x = x + self.pos_embed
        if self.angle_cond_token:
            x = x + a
        x = x.float()

        # rope is [1, N, 2 * head_dim] and identical for every sample, so it is
        # broadcast rather than expanded to the batch
        rope = self.rope
        if self.use_inctx:
            inctx_rope = self.inctx_rope
        inctx_size = 0

        for i, block in enumerate(self.blocks):
            if self.use_inctx and i == self.inctx_start:
                if c is not None:
                    inctx_embed = c.repeat(1, self.inctx_size, 1)
                else:
                    inctx_embed = self.inctx_embed.repeat(B, 1, 1)

                inctx_embed = inctx_embed + self.inctx_pos_embed
                inctx_embed = inctx_embed.to(x.dtype)
                x = torch.cat([inctx_embed, x], dim=1)
                inctx_size = self.inctx_size

            if self.use_inctx and i >= self.inctx_start:
                rope = inctx_rope

            x = block(x, cond=c, rope=rope)

            if i in extract_layer_idxs:
                extracted_hidden_states[i] = x

            if self.spherify_model and i != self.num_layers - 1:
                xl = x[:, :inctx_size, :]
                xr = x[:, inctx_size:, :]
                xr = self.f(xr)
                x = torch.cat([xl, xr], dim=1)

        h = x[:, inctx_size:, :]

        x = self.ffn(h, cond=c)
        x = self.out(x)

        if len(extract_layer_idxs) > 0:
            return x, extracted_hidden_states

        return x
