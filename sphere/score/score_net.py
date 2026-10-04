
import torch
import torch.nn as nn
from torch import Tensor

from sphere.layers import Block, TimestepEmbedder, get_rope_tensor


class ScoreNet(nn.Module):
    """
    Preconditioning EDM
        https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
    """

    def __init__(
        self,
        latent_dim: int,
        num_tokens: int,
        width: int = 512,
        use_qk_norm: bool = False,
        width_margin: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        drop_residual_path_prob: float = 0.0,
        sigma_data: float = 0.5,
        sigma_cond: bool = True,
        class_cond: bool = False,
        num_classes: int = 0,
        sdpa_mode: str = "sdpa",
        adaln_single: bool = False,
    ):
        super().__init__()
        if int(latent_dim) > int(width):
            width = int(latent_dim) + int(width_margin)
        quant = 4 * num_heads
        width = -(-int(width) // quant) * quant

        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.width = width
        self.sigma_data = sigma_data
        self.sigma_cond = sigma_cond
        self.class_cond = bool(class_cond)
        self.num_classes = int(num_classes)
        assert not self.class_cond or self.num_classes > 0, (
            f"class-conditioned score net needs num_classes > 0, got "
            f"{self.num_classes}"
        )
        self.adaln_single = bool(adaln_single)

        grid = int(round(num_tokens**0.5))
        assert grid * grid == num_tokens, f"num_tokens {num_tokens} not square"

        self.in_proj = nn.Linear(latent_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, width))
        self.sigma_embed = TimestepEmbedder(hidden_size=width)
        self.class_embed = (
            nn.Embedding(self.num_classes, width) if self.class_cond else None
        )

        # adaLN-single from PIXART-α
        if self.adaln_single:
            self.adaLN_shared = nn.Sequential(
                nn.SiLU(),
                nn.Linear(width, 6 * width, bias=True),
            )
            self.adaLN_offset = nn.Parameter(torch.zeros(depth, 6 * width))
        else:
            self.adaLN_shared = None
            self.adaLN_offset = None

        self.blocks = nn.ModuleList(
            [
                Block(
                    width,
                    num_heads,
                    use_modulation=not self.adaln_single,
                    qk_norm=use_qk_norm,
                    drop_residual_path_prob=drop_residual_path_prob,
                    sdpa_mode=sdpa_mode,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(width, eps=1e-5)
        self.out = nn.Linear(width, latent_dim)

        rope = get_rope_tensor(
            dim=width // num_heads, seq_h=grid, seq_w=grid
        ).unsqueeze(0)
        self.register_buffer("rope", rope, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.class_embed is not None:
            nn.init.normal_(self.class_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero-out adaLN + output for a near-identity (c_skip) start
        if self.adaln_single:
            nn.init.zeros_(self.adaLN_shared[-1].weight)
            nn.init.zeros_(self.adaLN_shared[-1].bias)
            nn.init.zeros_(self.adaLN_offset)
        else:
            for block in self.blocks:
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _precond(self, sigma: Tensor):
        sd = self.sigma_data
        s2 = sigma**2
        c_skip = sd**2 / (s2 + sd**2)
        c_out = sigma * sd / (s2 + sd**2).sqrt()
        c_in = 1.0 / (s2 + sd**2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise

    def forward(
        self,
        x_t: Tensor,
        sigma: Tensor,
        class_labels: Tensor | None = None,
    ):
        """
        Inputs:
            x_t: [B, N, D]
            sigma: [B, 1, 1]

        Returns:
            x_0: [B, N, D], predicted clean latent

        Eq.(7) in the EDM paper:
            D = c_skip(σ) * x_t + c_out(σ) * F(c_in(σ) * x_t, c_noise(σ))
        """
        assert x_t.dtype == torch.float32
        sigma = sigma.to(x_t.dtype)

        c_skip, c_out, c_in, c_noise = self._precond(sigma)
        if not self.sigma_cond:
            c_noise = torch.zeros_like(c_noise)

        c = self.sigma_embed(c_noise.reshape(-1))  # [B, D]
        if self.class_cond:
            assert class_labels is not None, "class-conditioned score net needs labels"
            class_labels = class_labels.reshape(-1).to(
                device=x_t.device, dtype=torch.long
            )
            assert class_labels.shape[0] == x_t.shape[0]
            c = c + self.class_embed(class_labels)
        c = c.unsqueeze(1)  # [B, 1, D]
        rope = self.rope  # [1, N, 2 * head_dim], broadcast over the batch

        h = self.in_proj(c_in * x_t) + self.pos_embed
        h = h.to(x_t.dtype)

        if self.adaln_single:
            shared = self.adaLN_shared(c)  # [B, 1, 6w]
            for i, block in enumerate(self.blocks):
                h = block(h, rope=rope, mod=shared + self.adaLN_offset[i])
        else:
            for block in self.blocks:
                h = block(h, cond=c, rope=rope)
        h = self.norm_out(h)
        f = self.out(h)

        pred = c_skip * x_t + c_out * f.to(x_t.dtype)
        return pred
