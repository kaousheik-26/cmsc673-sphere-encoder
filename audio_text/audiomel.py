"""Mel spectrogram <-> SE2 image, for BigVGAN v2 22 kHz / 80 band / fmax 8 kHz / hop 256.

Image layout (256 x 256, grey copied to RGB):
  3 strips of 80 mel bands x 256 frames, stacked top to bottom (frames 0-255, 256-511, 512-767),
  low frequency at the bottom of each strip; rows 240-255 = floor (black).
  768 frames = 8.92 s of audio.
Scale is GLOBAL (fixed LO..HI over the whole dataset), not per clip, so loudness is
part of what the model learns and generated images map back to absolute log-mel.
"""
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

SR, N_FFT, N_MELS, HOP, WIN, FMIN, FMAX = 22050, 1024, 80, 256, 1024, 0, 8000
LO = float(np.log(1e-5))  # BigVGAN log-mel floor
STRIP, N_STRIPS = 256, 3
FRAMES = STRIP * N_STRIPS  # 768 frames
DB = 20 / np.log(10)  # natural log magnitude -> dB

_cache = {}


def mel_spectrogram(y: torch.Tensor) -> torch.Tensor:
    """Same as BigVGAN meldataset.mel_spectrogram (center=False). y [B, T] in [-1, 1] -> [B, 80, frames]."""
    key = str(y.device)
    if key not in _cache:
        mb = librosa_mel_fn(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        _cache[key] = (torch.from_numpy(mb).float().to(y.device), torch.hann_window(WIN).to(y.device))
    mb, win = _cache[key]
    pad = (N_FFT - HOP) // 2
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, N_FFT, hop_length=HOP, win_length=WIN, window=win, center=False,
                      pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(mb @ spec, min=1e-5))


def to_img(mel: torch.Tensor, hi: float) -> torch.Tensor:
    """mel [..., 80, 768] (natural-log) -> image [..., 3, 256, 256] in [0, 1]."""
    assert mel.shape[-1] == FRAMES and mel.shape[-2] == N_MELS, mel.shape
    v = ((mel - LO) / (hi - LO)).clamp(0, 1).flip(-2)  # low freq at bottom
    strips = [v[..., i * STRIP:(i + 1) * STRIP] for i in range(N_STRIPS)]
    pad = v.new_zeros(*v.shape[:-2], 256 - N_STRIPS * N_MELS, STRIP)
    g = torch.cat(strips + [pad], -2)  # [..., 256, 256]
    return g.unsqueeze(-3).expand(*g.shape[:-2], 3, 256, 256).contiguous()


def from_img(img: torch.Tensor, hi: float) -> torch.Tensor:
    """image [..., 3, 256, 256] in [0, 1] -> mel [..., 80, 768] (natural-log)."""
    g = img.float().mean(-3).clamp(0, 1)
    strips = [g[..., i * N_MELS:(i + 1) * N_MELS, :] for i in range(N_STRIPS)]
    v = torch.cat(strips, -1).flip(-2)
    return v * (hi - LO) + LO


def colour(mel: torch.Tensor, hi: float) -> torch.Tensor:
    """mel [80, T] -> RGB magma picture [3, 80, T] for figures."""
    import matplotlib
    v = ((mel - LO) / (hi - LO)).clamp(0, 1).flip(0).cpu().numpy()
    return torch.from_numpy(matplotlib.colormaps["magma"](v)[..., :3]).permute(2, 0, 1).float()
