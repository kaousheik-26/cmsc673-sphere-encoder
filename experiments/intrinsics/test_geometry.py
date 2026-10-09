import pytest
import torch

from experiments.intrinsics.geometry import (
    geodesic_move, normalize, normalized_addition, project_tangent,
)

@pytest.mark.parametrize("mode,shape,dims", [
    ("global", (3, 5, 7), (1, 2)),
    ("global", (3, 2, 4, 5), (1, 2, 3)),
    ("local", (3, 5, 7), (2,)),
])
def test_spherical_invariants(mode, shape, dims):
    generator = torch.Generator().manual_seed(673)
    raw = torch.randn(shape, generator=generator, dtype=torch.float64)
    z = normalize(raw, mode)
    torch.testing.assert_close(z.square().mean(dims), torch.ones_like(z.square().mean(dims)))
    direction = torch.randn(shape[1:], generator=generator, dtype=z.dtype)
    tangent = project_tangent(z, direction, mode)
    torch.testing.assert_close((z * tangent).sum(dims), torch.zeros_like(z.sum(dims)), atol=1e-12, rtol=0)
    torch.testing.assert_close(geodesic_move(z, direction, 0, mode), z)
    for angle in [5, 40, 90, -60, 180]:
        moved = geodesic_move(z, direction, angle, mode)
        torch.testing.assert_close(moved.square().sum(dims), z.square().sum(dims))
        cosine = (moved * z).sum(dims) / z.square().sum(dims)
        torch.testing.assert_close(cosine, torch.full_like(cosine, torch.cos(torch.deg2rad(torch.tensor(float(angle), dtype=z.dtype))).item()))
    baseline = normalized_addition(z, direction, mode=mode)
    torch.testing.assert_close(baseline.square().mean(dims), torch.ones_like(baseline.square().mean(dims)))
    short = z * 0.999
    torch.testing.assert_close(geodesic_move(short, direction, 40, mode).square().sum(dims), short.square().sum(dims))


@pytest.mark.parametrize("mode", ["global", "local"])
def test_degenerate_and_batched_angles(mode):
    z = normalize(torch.randn(3, 5, 7), mode)
    for direction in [torch.zeros_like(z), z, -2 * z]:
        torch.testing.assert_close(geodesic_move(z, direction, 60, mode), z)
    direction = torch.randn_like(z)
    angles = torch.tensor([0., 20., 60.])
    moved = geodesic_move(z, direction, angles, mode)
    for i in range(3):
        torch.testing.assert_close(moved[i:i+1], geodesic_move(z[i:i+1], direction[i:i+1], angles[i], mode))
    if mode == "local":
        torch.testing.assert_close(geodesic_move(z, direction, angles[:, None].expand(3, 5), mode), moved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_finite_outputs(dtype):
    z = normalize(torch.randn(2, 4, 8).to(dtype))
    moved = geodesic_move(z, torch.randn_like(z), 40)
    assert moved.dtype == dtype
    assert torch.isfinite(moved).all()
    torch.testing.assert_close(moved.float().norm(dim=(1, 2)), z.float().norm(dim=(1, 2)), atol=0.04, rtol=0.005)


def test_invalid_inputs():
    z = torch.ones(2, 3, 4)
    for invalid in [torch.zeros_like(z), torch.full_like(z, float("nan")), z.long()]:
        with pytest.raises(ValueError):
            normalize(invalid)
    with pytest.raises(ValueError):
        normalize(z, "unknown")
    with pytest.raises(ValueError):
        normalize(torch.ones(2, 3), "local")
    with pytest.raises(ValueError):
        normalized_addition(z, -z)
    with pytest.raises(ValueError):
        geodesic_move(z, z, float("inf"))
    with pytest.raises(ValueError):
        project_tangent(z, torch.full_like(z, float("nan")))


@pytest.mark.parametrize("mode", ["global", "local"])
def test_gradients(mode):
    torch.manual_seed(673)
    z = normalize(torch.randn(2, 3, 4, dtype=torch.float64), mode)
    direction = torch.randn_like(z, requires_grad=True)
    angle = torch.tensor(20., dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u, a: geodesic_move(z, u, a, mode), (direction, angle))
    assert torch.autograd.gradcheck(lambda u: normalized_addition(z, u, mode=mode), (direction,))


@pytest.mark.parametrize("mode", ["global", "local"])
def test_matched_tangent_addition_baseline(mode):
    """A tangent offset scaled by tan(angle) matches a geodesic below 90deg."""
    torch.manual_seed(673)
    z = normalize(torch.randn(2, 3, 4, dtype=torch.float64), mode)
    direction = torch.randn_like(z)
    tangent = normalize(project_tangent(z, direction, mode), mode)
    for angle in [0., 5., 20., 60., -40.]:
        scale = torch.tan(torch.deg2rad(torch.tensor(angle, dtype=z.dtype)))
        torch.testing.assert_close(normalized_addition(z, tangent, scale, mode),
                                   geodesic_move(z, direction, angle, mode))
