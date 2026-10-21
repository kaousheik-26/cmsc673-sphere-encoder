# Check normal angles, segmentation scoring, and differentiable palette conversion.
import pytest
import torch
from experiments.intrinsics.property_metrics import (
    segmentation_palette, palette_labels, confusion_matrix, segmentation_metrics, normal_metrics,
)


# Check unique class colors, label recovery, and known accuracy/IoU examples.
def test_segmentation_encoding_and_confusion():
    palette = segmentation_palette()
    assert len(torch.unique(palette, dim=0)) == 150
    labels = torch.tensor([[0, 1], [2, 149]])
    image = palette[labels].permute(2, 0, 1)
    torch.testing.assert_close(palette_labels(image, palette), labels)
    metrics = segmentation_metrics(confusion_matrix(image, labels, palette))
    assert metrics['miou'] == 1 and metrics['pixel_accuracy'] == 1
    wrong = palette[torch.zeros_like(labels)].permute(2, 0, 1)
    metrics = segmentation_metrics(confusion_matrix(wrong, labels, palette))
    assert metrics['pixel_accuracy'] == 0.25
    assert metrics['miou'] == pytest.approx(0.0625)


# Check matching, opposite, and undefined normals against expected angular errors.
def test_normal_angles_and_zero_vector():
    target = torch.tensor([0.5, 0.5, 1.]).reshape(3, 1, 1)
    assert normal_metrics(target, target)['mean_angle_deg'] == 0
    opposite = torch.tensor([0.5, 0.5, 0.]).reshape(3, 1, 1)
    assert normal_metrics(opposite, target)['mean_angle_deg'] == 180
    undefined = torch.full_like(target, 0.5)
    assert normal_metrics(undefined, target)['mean_angle_deg'] == 90


# Check that palette logits recover labels and remain differentiable for categorical training.
def test_categorical_logits_match_palette_and_retain_gradients():
    from experiments.intrinsics.categorical_segmentation import palette_logits
    palette = segmentation_palette()
    labels = torch.tensor([[0, 1], [2, 149]])
    image = palette[labels].permute(2, 0, 1)[None].requires_grad_(True)
    logits = palette_logits(image, palette)
    torch.testing.assert_close(logits.argmax(1)[0], labels)
    torch.nn.functional.cross_entropy(logits, labels[None]).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
