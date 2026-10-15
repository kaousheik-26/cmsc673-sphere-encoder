import torch
from torch import nn

from .geometry import geodesic_move, normalize, normalized_addition


class SharedTransform(nn.Module):
    def __init__(self, initial_direction, method="geodesic", mode="global"):
        super().__init__()
        if method not in ("geodesic", "addition"):
            raise ValueError("Unknown method")
        self.direction = nn.Parameter(initial_direction.clone())
        self.raw_size = nn.Parameter(torch.tensor(-0.3, device=initial_direction.device))
        self.method = method
        self.mode = mode

    def size(self):
        # Shared scalar for every image; bounded to avoid antipodal edits.
        return torch.sigmoid(self.raw_size) * (75 if self.method == "geodesic" else 3)

    def forward(self, z):
        if self.method == "geodesic":
            return geodesic_move(z, self.direction, self.size(), self.mode)
        return normalized_addition(z, normalize(self.direction, self.mode), self.size(), self.mode)
