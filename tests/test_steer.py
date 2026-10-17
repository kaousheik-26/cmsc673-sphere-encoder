import pytest
import torch

from halfsphere.geometry import token_manifold_gap
from halfsphere.steer import SteerConfig, steer_optimize

H = 32


@pytest.mark.parametrize("mode,manifold", [("rgd", "sphere"), ("adam", "sphere"), ("rgd", "tokens")])
def test_steer_decreases_loss_and_stays_on_sphere(setup, mode, manifold):
    bb, _, mA, y = setup
    cfg = SteerConfig(steps=40, mode=mode, n_restarts=2, lam_cyc=0.1, lr=0.5 if mode == "rgd" else 0.05, manifold=manifold)
    X, V, Lo, _ = steer_optimize(bb, y, mA, cfg)
    with torch.no_grad():
        v0 = bb.encode(y)
        base = ((mA * (bb.decode(v0, 0.0) - y)) ** 2).sum((1, 2, 3)) / (mA.sum((1, 2, 3)) * 3) / 2
    assert X.shape == (2, *y.shape) and torch.all(Lo[0] <= base + 1e-6)
    assert torch.allclose(V.flatten(2).norm(dim=2) ** 2, torch.full(V.shape[:2], float(bb.L)), rtol=1e-4)
    if manifold == "tokens":
        assert (token_manifold_gap(V[0])["dist_frac"] < 1e-4).all()
