"""Running colored point cloud for autoregressive warp-and-fill (R1)."""
import torch
import torch.nn.functional as F

from se3d.geometry.cameras import unproject_depth


def edge_filter(depth, mask, rel_thresh=0.04, erode=2):
    """drop 'flying pixels': erode the mask by `erode` px and remove pixels whose
    depth differs from any 3x3 neighbour by more than rel_thresh * depth."""
    m = mask.float()[None, None]
    if erode > 0:
        m = -F.max_pool2d(-m, 2 * erode + 1, 1, erode)
    d = depth[None, None]
    big = torch.where(mask[None, None], d, torch.zeros_like(d))
    dmax = F.max_pool2d(big, 3, 1, 1)
    dmin = -F.max_pool2d(-torch.where(mask[None, None], d, torch.full_like(d, 1e9)), 3, 1, 1)
    jump = torch.maximum(dmax - d, d - dmin) / d.clamp_min(1e-6)
    keep = (m > 0.5) & (jump < rel_thresh)
    return keep[0, 0] & mask
from se3d.geometry.warp import splat_points


class PointCloud:
    def __init__(self, device="cuda"):
        self.xyz = torch.zeros(0, 3, device=device)
        self.rgb = torch.zeros(0, 3, device=device)
        self.view_id = torch.zeros(0, dtype=torch.long, device=device)

    @property
    def n(self):
        return self.xyz.shape[0]

    def add_view(self, rgb, depth, K, w2c, mask, view_id=0, edge_thresh=0.0, erode=0):
        """rgb [H,W,3], depth [H,W], mask [H,W] bool: which pixels to add.
        edge_thresh > 0 removes depth-discontinuity ('flying') pixels first."""
        mask = mask & (depth > 0)
        if edge_thresh > 0 or erode > 0:
            mask = edge_filter(depth, mask, edge_thresh if edge_thresh > 0 else 1e9, erode)
        pw, valid = unproject_depth(depth, K, w2c, mask)
        self.xyz = torch.cat([self.xyz, pw])
        self.rgb = torch.cat([self.rgb, rgb[valid]])
        self.view_id = torch.cat([self.view_id, torch.full((pw.shape[0],), view_id, device=pw.device)])
        return pw.shape[0]

    def render(self, K, w2c, H, W, splat_radius=1):
        """-> rgb [H,W,3], known [H,W], depth [H,W]"""
        return splat_points(self.xyz, self.rgb, K, w2c, H, W, splat_radius)
