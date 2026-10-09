import math
import torch

def _prepare(z, mode):
    if not z.is_floating_point() or z.ndim < 2 or z.numel() == 0:
        raise ValueError("Expected a nonempty floating-point batched latent")
    if mode == "global":
        dims = tuple(range(1, z.ndim))
    elif mode == "local" and z.ndim == 3:
        dims = (2,)
    else:
        raise ValueError("mode must be global, or local with shape [B,N,D]")
    work = z if z.dtype == torch.float64 else z.float()
    if not torch.isfinite(work).all():
        raise ValueError("Latents must be finite")
    radius = math.sqrt(math.prod(z.shape[d] for d in dims))
    return work, dims, radius


def normalize(z, mode="global"):

    work, dims, radius = _prepare(z, mode)
    norm = torch.linalg.vector_norm(work, dim=dims, keepdim=True)
    if (norm == 0).any():
        raise ValueError("A zero vector has no spherical normalization")
    return (work * (radius / norm)).to(z.dtype)


def project_tangent(z, direction, mode="global"):

    work, dims, _ = _prepare(z, mode)
    direction = torch.as_tensor(direction, device=z.device, dtype=work.dtype)
    direction = torch.broadcast_to(direction, z.shape)
    if not torch.isfinite(direction).all():
        raise ValueError("Directions must be finite")
    squared_norm = work.square().sum(dim=dims, keepdim=True)
    if (squared_norm == 0).any():
        raise ValueError("A zero latent has no tangent plane")
    tangent = direction - (work * direction).sum(dim=dims, keepdim=True) / squared_norm * work
    return tangent.to(z.dtype)


def geodesic_move(z, direction, angle_deg, mode="global"):

    work, dims, _ = _prepare(z, mode)
    tangent = project_tangent(work, direction, mode)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=dims, keepdim=True)
    radius = torch.linalg.vector_norm(work, dim=dims, keepdim=True)
    angle = torch.as_tensor(angle_deg, device=z.device, dtype=work.dtype)
    target_shape = (z.shape[0],) if mode == "global" else z.shape[:2]
    if angle.ndim == 1 and mode == "local":
        angle = angle[:, None]
    angle = torch.broadcast_to(angle, target_shape)
    if not torch.isfinite(angle).all():
        raise ValueError("Angles must be finite")
    angle = torch.deg2rad(angle).reshape(*target_shape, *([1] * len(dims)))
    ambient = torch.broadcast_to(torch.as_tensor(direction, device=z.device, dtype=work.dtype), z.shape)
    threshold = 32 * torch.finfo(work.dtype).eps * torch.linalg.vector_norm(ambient, dim=dims, keepdim=True)
    valid = tangent_norm > threshold
    unit = tangent / torch.where(valid, tangent_norm, torch.ones_like(tangent_norm))
    moved = torch.cos(angle) * work + torch.sin(angle) * radius * unit
    return torch.where(valid, moved, work).to(z.dtype)


def normalized_addition(z, offset, scale=1.0, mode="global"):

    work, _, _ = _prepare(z, mode)
    offset = torch.broadcast_to(torch.as_tensor(offset, device=z.device, dtype=work.dtype), z.shape)
    return normalize(work + scale * offset, mode).to(z.dtype)
