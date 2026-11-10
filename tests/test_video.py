import math

import torch
from PIL import Image

from halfsphere.geometry import angle_deg, exp_map, f, tangent
from spherevideo.clips import camera_clip, write_mp4
from spherevideo.predict import VelocityMap, extrapolate, log_map, turning_deg


def _geodesic(n, L=64, step=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    z0 = f(torch.randn(1, 8, L // 8, generator=g))
    d = tangent(z0, torch.randn(1, 8, L // 8, generator=g))
    d = d / d.norm() * math.radians(step) * math.sqrt(L)
    return torch.cat([exp_map(z0, k * d) for k in range(n)])


def test_log_exp_roundtrip():
    z = _geodesic(2)
    back = exp_map(z[:1], log_map(z[:1], z[1:]))
    assert float(angle_deg(back, z[1:])) < 1e-2


def test_extrapolation_exact_on_geodesic():
    z = _geodesic(6)
    assert float(angle_deg(extrapolate(z[:-2], z[1:-1]), z[2:]).max()) < 0.05
    assert float(turning_deg(z).max()) < 0.5


def test_velocity_map_gain_one_on_geodesic():
    Z = torch.stack([_geodesic(6, seed=s) for s in range(4)])
    m = VelocityMap.fit(Z, rank=0)
    assert abs(m.gain - 1.0) < 1e-2
    z = Z[0]
    assert float(angle_deg(m.step(z[:1], z[1:2]), z[2:3])) < 0.05


def test_camera_clip_moves(tmp_path):
    im = Image.fromarray((torch.rand(120, 200, 3) * 255).byte().numpy())
    for kind in ("pan", "zoom", "panzoom"):
        x = camera_clip(im, 5, size=32, kind=kind)
        assert x.shape == (5, 3, 32, 32)
        assert float((x[0] - x[-1]).abs().mean()) > 0.01
    write_mp4(x, str(tmp_path / "a.mp4"), fps=4)
    assert (tmp_path / "a.mp4").stat().st_size > 0
