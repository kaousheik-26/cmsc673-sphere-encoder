import pytest
import torch
from torch.func import jacrev

from halfsphere.backbone import DummyBackbone
from halfsphere.masks import half_mask

H = 32


@pytest.fixture(scope="session")
def setup():
    torch.manual_seed(0)
    bb = DummyBackbone(H=H)
    x = torch.rand(2, 3, H, H) * 2 - 1
    mA = half_mask(2, H, H)
    return bb, x, mA, mA * x


@pytest.fixture(scope="session")
def dense(setup):
    bb, x, _, _ = setup
    v = bb.encode(x[:1]).detach()
    Jfull = jacrev(lambda u: bb.decode(u, 0.0))(v).reshape(-1, bb.L)
    vf = v.flatten()
    P = torch.eye(bb.L) - vf[:, None] * vf[None] / (vf @ vf)
    return v, Jfull, P
