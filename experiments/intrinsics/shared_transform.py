# Learn one latent direction and movement strength shared across input images.
import torch
from torch import nn

from .geometry import geodesic_move, normalize, normalized_addition


# Keep the edit direction and strength trainable while the encoder and decoder stay frozen.
class SharedTransform(nn.Module):
    # Create shared parameters from an initial direction and select the movement rule.
    def __init__(self, initial_direction, method="geodesic", mode="global"):
        super().__init__()
        if method not in ("geodesic", "addition"):
            raise ValueError("Unknown method")
        self.direction = nn.Parameter(initial_direction.clone())
        self.raw_size = nn.Parameter(torch.tensor(-0.3, device=initial_direction.device))
        self.method = method
        self.mode = mode

    # Convert the unconstrained parameter to a bounded angle in degrees or addition scale.
    def size(self):
        # Shared scalar for every image; bounded to avoid antipodal edits.
        return torch.sigmoid(self.raw_size) * (75 if self.method == "geodesic" else 3)

    # Apply the same learned edit parameters to every latent in the batch.
    def forward(self, z):
        if self.method == "geodesic":
            return geodesic_move(z, self.direction, self.size(), self.mode)
        return normalized_addition(z, normalize(self.direction, self.mode), self.size(), self.mode)
