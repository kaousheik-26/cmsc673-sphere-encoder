"""Check fresh sampling and masked measured-depth evaluation conventions."""
import numpy as np
import pytest
import torch

from experiments.followup.evaluate_diode import aligned_sensor_crop, measured_metrics
from experiments.followup.prepare_followup_pairs import sample_manifest


def test_measured_depth_orientation_alignment_and_invalid_pixels():
    depth = torch.linspace(1, 8, 256).reshape(16, 16)
    inverse = depth.double().reciprocal()
    low, high = torch.quantile(inverse.flatten(), torch.tensor([.01, .99], dtype=torch.float64))
    prediction = ((inverse - low) / (high - low)).clamp(0, 1)[None].expand(3, -1, -1)
    mask = torch.ones_like(depth, dtype=torch.bool)
    perfect = measured_metrics(prediction, depth, mask)
    assert perfect["normalized_inverse_mse"] < 1e-12
    assert perfect["inverse_pearson"] > .999
    inverted = measured_metrics(1 - prediction, depth, mask)
    assert inverted["inverse_pearson"] < -.999
    assert inverted["oracle_affine_slope"] < 0  # Negative alignment must not conceal orientation failure.
    mask[0, 0] = False
    depth[0, 0] = float("nan")
    assert measured_metrics(prediction, depth, mask)["valid_pixels"] == 255


def test_nearest_sensor_crop_retains_mask_and_alignment():
    depth = np.arange(768 * 1024, dtype=np.float32).reshape(768, 1024) + 1
    mask = np.ones_like(depth, dtype=bool)
    a, b = aligned_sensor_crop(depth, mask)
    assert a.shape == b.shape == (256, 256)
    assert b.all()
    assert a[0, 0] < a[-1, -1]
    assert torch.isfinite(a).all()


def test_fresh_sampling_is_reproducible_and_excludes_prior_images(tmp_path):
    for category in ("a", "b", "c", "d"):
        folder = tmp_path / category
        folder.mkdir()
        for i in range(8):
            (folder / f"{i}.jpg").touch()
    old = [{"image": str(tmp_path / "a" / f"{i}.jpg"), "category": "a", "split": split}
           for i, split in enumerate(("train", "validation", "test"))]
    old.append({"image": str(tmp_path / "b" / "0.jpg"), "category": "b", "split": "unseen"})
    sampled = sample_manifest(old, tmp_path, per_category=2)
    assert sampled == sample_manifest(old, tmp_path, per_category=2)
    prior = {r["image"] for r in old}
    fresh = [r for r in sampled if r["split"] in ("test", "unseen")]
    assert len(fresh) == 4
    assert all(r["image"] not in prior for r in fresh)
    assert {r["category"] for r in fresh if r["split"] == "unseen"}.isdisjoint({"a", "b"})
