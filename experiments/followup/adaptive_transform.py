"""RGB-conditioned spherical edits with a small learned direction dictionary."""
import math

import torch
from torch import nn

from experiments.intrinsics.geometry import geodesic_move, normalize, normalized_addition, project_tangent


class AdaptiveTransform(nn.Module):
    """Predict direction mixtures and/or size from spatially pooled RGB tokens.

    A factorized dictionary optionally holds total direction parameters roughly
    constant as the number of directions increases. No target enters forward().
    """

    def __init__(self, initial, rank=1, method="geodesic", conditional_size=True,
                 conditional_direction=True, hidden=32, factorized=False):
        super().__init__()
        if initial.ndim != 3 or initial.shape[0] != 1 or rank < 1:
            raise ValueError("Expected initial direction [1,N,D] and positive rank")
        if method not in ("geodesic", "addition", "tangent"):
            raise ValueError("Unknown movement method")
        if conditional_direction and rank == 1:
            conditional_direction = False
        self.method, self.rank = method, rank
        self.conditional_size = conditional_size
        self.conditional_direction = conditional_direction
        self.factorized = factorized
        self.tokens, self.channels = initial.shape[1:]
        if factorized:
            # K * (N*r + r*D) is bounded by the original full direction size.
            width = max(1, self.tokens * self.channels // (rank * (self.tokens + self.channels)))
            u, s, vh = torch.linalg.svd(initial[0].float(), full_matrices=False)
            width = min(width, len(s))
            left = (u[:, :width] * s[:width].sqrt()).unsqueeze(0).repeat(rank, 1, 1)
            right = (s[:width].sqrt()[:, None] * vh[:width]).unsqueeze(0).repeat(rank, 1, 1)
            self.left = nn.Parameter(left + .01 * torch.randn_like(left))
            self.right = nn.Parameter(right + .01 * torch.randn_like(right))
        else:
            dictionary = initial.repeat(rank, 1, 1)
            self.dictionary = nn.Parameter(dictionary + .05 * torch.randn_like(dictionary))
        outputs = (rank if conditional_direction else 0) + int(conditional_size)
        if outputs:
            self.conditioner = nn.Sequential(nn.LayerNorm(self.channels),
                                             nn.Linear(self.channels, hidden), nn.Tanh(),
                                             nn.Linear(hidden, outputs))
            nn.init.normal_(self.conditioner[-1].weight, std=.001)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.conditioner = None
        self.raw_size = nn.Parameter(torch.tensor(-.3, device=initial.device))

    def forward(self, z):
        if z.shape[1:] != (self.tokens, self.channels):
            raise ValueError("Input latent shape differs from the direction dictionary")
        features = self.conditioner(z.mean(1)) if self.conditioner is not None else None
        dictionary = self.left @ self.right if self.factorized else self.dictionary
        if self.conditional_direction:
            weights = features[:, :self.rank].softmax(-1)
            direction = torch.einsum("bk,knd->bnd", weights, dictionary)
        else:
            direction = dictionary.mean(0, keepdim=True)
        raw_size = self.raw_size + (features[:, -1] if self.conditional_size else 0)
        if self.method == "addition":
            size = (raw_size.sigmoid() * 3)
            if size.ndim:
                size = size[:, None, None]
            return normalized_addition(z, normalize(direction), size)
        angle = raw_size.sigmoid() * 75
        if self.method == "geodesic":
            return geodesic_move(z, direction, angle)
        # Matched tangent addition is a numerical control, not a new hypothesis.
        tangent = project_tangent(z, direction)
        norm = tangent.norm(dim=(1, 2), keepdim=True)
        threshold = 32 * torch.finfo(z.dtype).eps * direction.norm(dim=(1, 2), keepdim=True)
        valid = norm > threshold
        tangent = tangent / norm.clamp_min(torch.finfo(z.dtype).tiny) * math.sqrt(self.tokens * self.channels)
        scale = torch.tan(torch.deg2rad(angle))
        if scale.ndim:
            scale = scale[:, None, None]
        return torch.where(valid, normalized_addition(z, tangent, scale), z)


class PropertyProbe(nn.Module):
    """Linear or small nonlinear token predictor, without the Sphere decoder."""

    def __init__(self, channels, outputs, hidden=0):
        super().__init__()
        self.layers = (nn.Conv2d(channels, outputs, 1) if not hidden else
                       nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                     nn.Conv2d(hidden, outputs, 1)))

    def forward(self, z, output_size):
        side = math.isqrt(z.shape[1])
        if side * side != z.shape[1]:
            raise ValueError("Probe requires a square token grid")
        x = z.transpose(1, 2).reshape(len(z), z.shape[2], side, side)
        return torch.nn.functional.interpolate(self.layers(x), size=output_size,
                                               mode="bilinear", align_corners=False)
