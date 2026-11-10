import glob
import os
import os.path as osp
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image


def to_tensor(im: Image.Image) -> torch.Tensor:
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def _window(W, H, kind, u):
    s = min(W, H)
    if kind == "pan":
        c = 0.7 * s
        if W >= H:
            return (W - c) * u, (H - c) / 2, c
        return (W - c) / 2, (H - c) * u, c
    if kind == "zoom":
        c = s * (1.0 - 0.5 * u)
        return (W - c) / 2, (H - c) / 2, c
    if kind == "panzoom":
        c = s * (0.9 - 0.3 * u)
        return (W - c) * u, (H - c) * (0.5 + 0.3 * (u - 0.5)), c
    raise ValueError(kind)


def camera_clip(im: Image.Image, n: int, size: int = 256, kind: str = "pan") -> torch.Tensor:
    im = im.convert("RGB")
    W, H = im.size
    frames = []
    for t in range(n):
        u = t / max(n - 1, 1)
        x0, y0, c = _window(W, H, kind, u)
        box = (x0, y0, x0 + c, y0 + c)
        frames.append(to_tensor(im.resize((size, size), Image.BICUBIC, box=box)))
    return torch.stack(frames)


def synthetic_clips(image_glob: str, n_clips: int, n_frames: int, size: int = 256, kinds=("pan", "zoom", "panzoom"), seed: int = 0):
    paths = sorted(glob.glob(image_glob))
    if not paths:
        raise FileNotFoundError(image_glob)
    g = np.random.default_rng(seed)
    pick = g.choice(len(paths), size=min(n_clips, len(paths)), replace=False)
    clips, meta = [], []
    for i, p in enumerate(pick):
        kind = kinds[i % len(kinds)]
        clips.append(camera_clip(Image.open(paths[p]), n_frames, size, kind))
        meta.append({"source": paths[p], "kind": kind})
    return torch.stack(clips), meta


def video_frames(path: str, n_frames: int, fps: float, size: int = 256, start_s: float = 0.0, src_fps: float = 16.0) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as d:
        vf = f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase:flags=bicubic,crop={size}:{size}"
        src = ["-framerate", str(src_fps), "-i", osp.join(path, "%05d.png")] if osp.isdir(path) else ["-ss", str(start_s), "-i", path]
        cmd = ["ffmpeg", "-v", "error", *src, "-vf", vf, "-frames:v", str(n_frames), osp.join(d, "%05d.png")]
        subprocess.run(cmd, check=True)
        files = sorted(glob.glob(osp.join(d, "*.png")))
        return torch.stack([to_tensor(Image.open(f)) for f in files])


def video_clips(video_glob: str, n_frames: int, fps: float, size: int = 256, src_fps: float = 16.0):
    clips, meta = [], []
    for p in sorted(glob.glob(video_glob)):
        x = video_frames(p, n_frames, fps, size, src_fps=src_fps)
        if x.shape[0] == n_frames:
            clips.append(x)
            meta.append({"source": p, "kind": osp.basename(p.rstrip("/")).split("_", 1)[-1].rsplit(".", 1)[0]})
    if not clips:
        raise FileNotFoundError(video_glob)
    return torch.stack(clips), meta


def write_mp4(frames: torch.Tensor, path: str, fps: float = 8.0):
    x = ((frames.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).contiguous().numpy()
    T, H, W, _ = x.shape
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    subprocess.run(cmd, input=x.tobytes(), check=True)
