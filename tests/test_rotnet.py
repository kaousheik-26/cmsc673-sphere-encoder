import torch

from halfsphere.geometry import angle_deg, exp_map, f, tangent
from spherevideo.rotnet import RotNet, compose, dir_cos, split_target


def test_rotation_split_roundtrip():
    g = torch.Generator().manual_seed(1)
    z0 = f(torch.randn(3, 16, 8, generator=g))
    zt = exp_map(z0, 0.3 * tangent(z0, torch.randn(3, 16, 8, generator=g)))
    th, d = split_target(z0, zt)
    assert torch.allclose(th, angle_deg(z0, zt), atol=1e-3)
    assert float(dir_cos(compose(z0, th, d), zt).min()) > 1 - 1e-6
    net = RotNet(token_dim=8, n_tokens=16, width=32, depth=1, heads=2, hint="two")
    th_p, d_p = net(z0, torch.tensor([0.0, 0.5, 1.0]), zt)
    assert th_p[0] == 0 and (th_p[1:] > 0).all() and d_p.shape == z0.shape
