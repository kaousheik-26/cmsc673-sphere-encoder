# Check that shared edits preserve sphere constraints and support optimization.
import pytest
import torch
from experiments.intrinsics.geometry import normalize
from experiments.intrinsics.shared_transform import SharedTransform


# Check both movement methods preserve the sphere and give trainable parameters gradients.
@pytest.mark.parametrize("method", ["geodesic", "addition"])
def test_shared_edit_has_gradients_and_preserves_sphere(method):
    torch.manual_seed(673)
    z = normalize(torch.randn(3, 4, 6))
    edit = SharedTransform(torch.randn(1, 4, 6), method)
    moved = edit(z)
    torch.testing.assert_close(moved.square().mean((1, 2)), torch.ones(3))
    (moved * torch.randn_like(moved)).sum().backward()
    assert edit.direction.grad is not None and torch.isfinite(edit.direction.grad).all()
    assert edit.direction.grad.norm() > 0
    assert edit.raw_size.grad is not None and edit.raw_size.grad.abs() > 0


# Check that changing batch size does not create image-specific edit parameters.
def test_held_out_samples_share_parameters():
    edit = SharedTransform(torch.randn(1, 4, 6))
    assert sum(p.numel() for p in edit.parameters()) == 25
    z = normalize(torch.randn(3, 4, 6))
    torch.testing.assert_close(edit(z), torch.cat([edit(x[None]) for x in z]))
