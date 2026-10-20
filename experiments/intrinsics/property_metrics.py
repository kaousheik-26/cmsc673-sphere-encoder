import torch
import torch.nn.functional as F


def segmentation_palette(classes=150):
    colors = []
    for label in range(classes):
        value, color = label, [0, 0, 0]
        for bit in range(8):
            for channel in range(3):
                color[channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
        colors.append(color)
    return torch.tensor(colors, dtype=torch.float32) / 255


def normal_metrics(prediction, target):
    a = F.normalize(prediction * 2 - 1, dim=0)
    b = F.normalize(target * 2 - 1, dim=0)
    angles = torch.rad2deg(torch.acos((a * b).sum(0).clamp(-1, 1)))
    return {"mean_angle_deg": angles.mean().item(), "median_angle_deg": angles.median().item(),
            "fraction_under_11_25_deg": (angles < 11.25).float().mean().item()}


def palette_labels(prediction, palette):
    flat = prediction.clamp(0, 1).permute(1, 2, 0).reshape(-1, 3)
    return torch.cat([torch.cdist(chunk, palette).argmin(1) for chunk in flat.split(8192)]).reshape(prediction.shape[1:])


def confusion_matrix(prediction, labels, palette):
    predicted = palette_labels(prediction, palette)
    classes = len(palette)
    return torch.bincount((labels.flatten() * classes + predicted.flatten()), minlength=classes**2).reshape(classes, classes)


def segmentation_metrics(confusion):
    matrix = confusion.double()
    union = matrix.sum(0) + matrix.sum(1) - matrix.diag()
    present = union > 0
    return {"pixel_accuracy": (matrix.diag().sum() / matrix.sum().clamp_min(1)).item(),
            "miou": (matrix.diag()[present] / union[present]).mean().item() if present.any() else None,
            "classes_with_union": present.sum().item()}
