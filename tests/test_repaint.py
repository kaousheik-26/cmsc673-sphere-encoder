import torch

from halfsphere.masks import token_index_for_half
from halfsphere.repaint import RepaintConfig, repaint_complete, repaint_samples, repaint_token_complete

H = 32


def test_repaint_pins_observed_half(setup):
    bb, _, mA, y = setup
    out = repaint_complete(bb, y, mA, RepaintConfig(K=6))
    assert torch.allclose(mA * out, mA * y.clamp(-1, 1), atol=1e-6)


def test_repaint_shared_noise_is_deterministic(setup):
    bb, _, mA, y = setup
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a = repaint_complete(bb, y, mA, RepaintConfig(K=4), generator=g1)
    b = repaint_complete(bb, y, mA, RepaintConfig(K=4), generator=g2)
    assert torch.equal(a, b)


def test_repaint_samples_trajectory_and_token_variant(setup):
    bb, x, mA, y = setup
    S = repaint_samples(bb, y, mA, n_samples=3, cfg=RepaintConfig(K=4))
    assert S.shape == (3, *y.shape)
    _, traj = repaint_complete(bb, y, mA, RepaintConfig(K=4, schedule="cosine"), return_traj=True)
    assert len(traj["steps"]) == 4 and len(traj["x"]) == 6
    iA, iB = token_index_for_half(H, H, bb.patch)
    out = repaint_token_complete(bb, y, mA, iA, iB, K=4)
    assert torch.allclose(mA * out, mA * y.clamp(-1, 1), atol=1e-6)
