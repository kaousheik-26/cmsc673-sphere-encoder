# Check depth normalization and reconstruction metrics without loading a checkpoint.
import pytest
import torch

from experiments.intrinsics.depth_feasibility import depth_to_target, similarity


# Check percentile normalization without reversing depth order or accepting constant targets.
def test_depth_normalization_preserves_order_and_rejects_constant():
    raw = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    target, normalization = depth_to_target(raw)
    assert target.shape == (3, 10, 10)
    assert normalization["low"] < normalization["high"]
    assert target.min() == 0 and target.max() == 1
    assert (target[0].flatten().diff() >= 0).all()
    torch.testing.assert_close(target[0], target[1])
    with pytest.raises(ValueError):
        depth_to_target(torch.ones(10, 10))
    with pytest.raises(ValueError):
        depth_to_target(torch.full((10, 10), float("nan")))


# Check that structure correlation and RGB error reveal different prediction failures.
def test_metrics_distinguish_structure_and_color():
    target = torch.arange(100, dtype=torch.float32).reshape(1, 10, 10).expand(3, -1, -1) / 100
    perfect = similarity(target, target)
    assert perfect["rgb_mse"] == 0
    assert perfect["gray_pearson"] == pytest.approx(1)
    inverted = similarity(1 - target, target)
    assert inverted["gray_pearson"] == pytest.approx(-1)
    constant = similarity(torch.full_like(target, target.mean().item()), target)
    assert constant["gray_pearson"] is None
    colored = target.clone()
    colored[0] += 0.1
    colored[1] -= 0.1
    color_metrics = similarity(colored, target)
    assert color_metrics["rgb_mse"] > 0
    assert color_metrics["gray_mse"] < 1e-12
    assert color_metrics["channel_spread_mean"] == pytest.approx(0.2)
