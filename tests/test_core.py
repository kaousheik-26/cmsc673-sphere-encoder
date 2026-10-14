import json
import math

import torch

from halfsphere.geometry import angle_deg, exp_map, f, project_tokens, tangent, token_manifold_gap
from halfsphere.masks import half_mask, prox_mask
from halfsphere.metrics import diversity, half_crop, kid, psnr_masked, seam_score

H = 32


def test_f_matches_repo_vector_rms_norm():
    from sphere.layers import vector_rms_norm

    z = torch.randn(3, 16, 8) * 3
    assert torch.equal(f(z), vector_rms_norm(z, zero_mean=False))


def test_exp_map_stays_on_sphere_and_moves_by_angle():
    v = f(torch.randn(2, 16, 8))
    xi = tangent(v, torch.randn_like(v))
    xi = xi / xi.flatten(1).norm(dim=1).view(-1, 1, 1) * math.sqrt(v[0].numel()) * math.radians(7)
    w = exp_map(v, xi)
    assert torch.allclose(w.flatten(1).norm(dim=1) ** 2, torch.full((2,), float(v[0].numel())), rtol=1e-4)
    assert torch.allclose(angle_deg(v, w), torch.full((2,), 7.0), atol=1e-3)


def test_token_manifold_gap_zero_on_projected_and_positive_on_random():
    z = torch.randn(2, 16, 32)
    gap_rand = token_manifold_gap(f(z))
    gap_proj = token_manifold_gap(f(project_tokens(z)))
    assert (gap_proj["dist_frac"] < 1e-5).all() and (gap_proj["tok_mean_abs"] < 1e-6).all()
    assert (gap_rand["dist_frac"] > 1e-2).all()


def test_prox_mask_matches_dense_solve():
    y = torch.randn(1, 3, 4, 4)
    mA = half_mask(1, 4, 4)
    xp = torch.randn(1, 3, 4, 4)
    lam = 0.37
    M = torch.diag((mA.expand_as(y)).flatten())
    sol = torch.linalg.solve(M + lam * torch.eye(M.shape[0]), M @ (mA * y).flatten() + lam * xp.flatten())
    assert torch.allclose(prox_mask(xp, mA * y, mA, lam).flatten(), sol, atol=1e-6)


def test_contract_shapes_and_sphere(setup):
    bb, x, _, _ = setup
    v = bb.encode(x)
    assert v.shape[1:] == bb.latent_shape and v.dtype == torch.float32
    assert torch.allclose(v.flatten(1).norm(dim=1) ** 2, torch.full((2,), float(bb.L)), rtol=1e-4)
    assert bb.decode(v, 0.0).shape == x.shape


def test_contract_rotation_angle(setup):
    bb, x, _, _ = setup
    v = bb.encode(x)
    z = bb.rotate(v, bb.noise_like(v), 60.0)
    assert torch.all((angle_deg(v, z) - 60).abs() < 5)


def test_contract_input_gradients_flow(setup):
    bb, x, _, _ = setup
    xx = x.clone().requires_grad_(True)
    bb.encode(xx).sum().backward()
    assert xx.grad is not None and xx.grad.abs().sum() > 0


def test_metrics_basic(setup):
    _, x, mA, _ = setup
    assert half_crop(x).shape[-1] == H // 2
    assert (psnr_masked(x, x, mA) > 100).all()
    smooth = torch.linspace(-1, 1, H).view(1, 1, 1, H).expand(2, 3, H, H)
    assert torch.allclose(seam_score(smooth), torch.ones(2), atol=0.05)
    assert diversity(torch.stack([x, x])).abs().max() == 0


def test_kid_zero_for_same_distribution_positive_for_shift():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(400, 64, generator=g), torch.randn(400, 64, generator=g)
    same, _ = kid(a, b, subset_size=100, n_subsets=50)
    shifted, _ = kid(a, b + 1.0, subset_size=100, n_subsets=50)
    assert abs(same) < 0.01 and shifted > 10 * abs(same) + 0.01


def test_balanced_subset_is_nested_and_balanced():
    from halfsphere.data import balanced_subset

    class FakeDS:
        list = [{"class_id": i % 5} for i in range(50)]

    a, b = balanced_subset(FakeDS, 10, seed=3), balanced_subset(FakeDS, 25, seed=3)
    assert b[:10] == a and len(set(b)) == 25
    counts = [sum(1 for i in a if FakeDS.list[i]["class_id"] == c) for c in range(5)]
    assert counts == [2] * 5


def test_records_writes_run_json_and_rows(tmp_path):
    from halfsphere.records import Run, read_rows

    with Run("unit", {"a": 1}, dev_dir=str(tmp_path), seeds={"base": 0}) as run:
        run.row(image_id="x", metrics={"p": torch.tensor(1.5)})
    meta = json.load(open(f"{run.dir}/run.json"))
    assert meta["status"] == "ok" and meta["config"] == {"a": 1} and "torch" in meta["env"]
    assert read_rows(run.dir)[0]["metrics"]["p"] == 1.5
