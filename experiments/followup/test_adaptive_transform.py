"""Check conditional edits, matched geometry, and evaluation data isolation."""
import pytest
import torch

from experiments.followup.adaptive_transform import AdaptiveTransform, PropertyProbe
from experiments.followup.compare_adaptive import validate_pairs, task_target, loss_for
from experiments.intrinsics.geometry import normalize


@pytest.mark.parametrize("rank,factorized", [(1, False), (4, False), (8, False), (4, True), (8, True)])
@pytest.mark.parametrize("method", ["geodesic", "addition", "tangent"])
def test_edits_preserve_radius_and_propagate_gradients(rank, factorized, method):
    torch.manual_seed(673)
    z = normalize(torch.randn(3, 16, 24))
    edit = AdaptiveTransform(torch.randn(1, 16, 24), rank, method, factorized=factorized)
    moved = edit(z)
    torch.testing.assert_close(moved.norm(dim=(1, 2)), z.norm(dim=(1, 2)))
    (moved * torch.randn_like(moved)).mean().backward()
    for p in edit.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert all(p.grad.abs().sum() > 0 for p in edit.parameters())
    torch.testing.assert_close(moved, torch.cat([edit(x[None]) for x in z]))


def test_matched_geometry_and_conditional_size():
    torch.manual_seed(673)
    z = normalize(torch.randn(3, 16, 24))
    edit = AdaptiveTransform(torch.randn(1, 16, 24), 4)
    tangent = AdaptiveTransform(torch.randn(1, 16, 24), 4, "tangent")
    tangent.load_state_dict(edit.state_dict())
    torch.testing.assert_close(edit(z), tangent(z), atol=2e-6, rtol=2e-6)
    # The predictor responds to the image, without inspecting targets.
    features = edit.conditioner(z.mean(1))
    assert not torch.equal(features[0], features[1])


def test_factorized_budget():
    initial = torch.randn(1, 256, 384)
    full = AdaptiveTransform(initial, 1)
    for rank in (4, 8):
        compact = AdaptiveTransform(initial, rank, factorized=True)
        assert compact.left.numel() + compact.right.numel() <= 256 * 384
        assert sum(p.numel() for p in compact.parameters()) <= sum(p.numel() for p in full.parameters()) + 1000


def test_probe_and_property_losses():
    z = torch.randn(2, 16, 24)
    for hidden in (0, 16):
        pred = PropertyProbe(24, 3, hidden)(z, (12, 12))
        assert pred.shape == (2, 3, 12, 12)
        for task in ("depth", "normals", "binary"):
            truth = torch.randint(0, 2, pred.shape).float() if task == "binary" else torch.rand_like(pred)
            assert torch.isfinite(loss_for(pred, truth, task))


def test_cache_overlap_and_binary_mapping():
    data = {"z": torch.rand(4, 16, 24), "target": torch.rand(4, 3, 8, 8),
            "rgb": torch.rand(4, 3, 8, 8), "labels": torch.randint(0, 4, (4, 8, 8)),
            "manifest": [{"image": str(i), "split": s, "category": str(i)}
                         for i, s in enumerate(("train", "validation", "test", "unseen"))]}
    assert len(validate_pairs(data)) == 4
    binary = task_target(data, "binary", [1, 3])
    torch.testing.assert_close(binary[:, 0].bool(), (data["labels"] == 1) | (data["labels"] == 3))
    data["rgb"][3] = data["rgb"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_pairs(data)
