"""Small image helpers."""
import numpy as np
from PIL import Image


def load_rgba_white(path):
    """RGBA file -> (RGB composited on white [H,W,3] in [0,1], alpha [H,W])."""
    a = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    return a[..., :3] * a[..., 3:] + (1 - a[..., 3:]), a[..., 3]
