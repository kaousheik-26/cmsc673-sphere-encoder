import dataclasses
import hashlib
import json
import os
import os.path as osp
import platform
import subprocess
import time

import torch


def _git(*args, cwd=None):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def config_hash(cfg) -> str:
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _file_sha256(path):
    if not path or not osp.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Run:
    def __init__(self, experiment: str, config: dict, dev_dir="workspace", ckpt_path="", seeds=None, repo_dir="."):
        sha = _git("rev-parse", "--short", "HEAD", cwd=repo_dir) or "nogit"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = osp.join(dev_dir, "runs", experiment, f"{stamp}-{sha}")
        os.makedirs(self.dir, exist_ok=True)
        diff = _git("diff", "HEAD", cwd=repo_dir)
        untracked = _git("ls-files", "--others", "--exclude-standard", cwd=repo_dir)
        if diff:
            with open(osp.join(self.dir, "git.diff"), "w") as fh:
                fh.write(diff)
        lock = osp.join(os.environ.get("SPHERE_HOME", ""), "requirements.lock.txt")
        self.meta = {
            "experiment": experiment,
            "config": config,
            "seeds": seeds or {},
            "git": {
                "sha": _git("rev-parse", "HEAD", cwd=repo_dir),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_dir),
                "dirty": bool(diff),
                "untracked": untracked.splitlines(),
            },
            "checkpoint": {"path": ckpt_path, "sha256": _file_sha256(ckpt_path)},
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "lock_sha256": _file_sha256(lock),
                "host": platform.node(),
            },
            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
        }
        self._write_meta()
        self.rows_path = osp.join(self.dir, "rows.jsonl")
        self._t0 = time.time()

    def _write_meta(self):
        with open(osp.join(self.dir, "run.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2, default=str)

    def row(self, **kw):
        with open(self.rows_path, "a") as fh:
            fh.write(json.dumps(kw, default=_jsonable) + "\n")

    def path(self, *parts):
        p = osp.join(self.dir, *parts)
        os.makedirs(osp.dirname(p), exist_ok=True)
        return p

    def finish(self, status="ok", **extra):
        self.meta.update(status=status, end=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - self._t0, 1), **extra)
        self._write_meta()

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.finish("ok" if et is None else f"error: {et.__name__}: {ev}")
        return False


def _jsonable(o):
    if torch.is_tensor(o):
        return o.item() if o.numel() == 1 else o.tolist()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return str(o)


LEGACY_METHOD = {"s2": "repaint", "s2_tok": "repaint_tok", "s1": "steer"}


def _from_legacy(r):
    if "method" in r:
        r["method"] = LEGACY_METHOD.get(r["method"], r["method"])
    if "s2_config" in r:
        r["repaint_config"] = r.pop("s2_config")
    for k in ("modes", "lm"):
        if isinstance(r.get(k), dict) and "s2" in r[k]:
            r[k]["repaint"] = r[k].pop("s2")
    return r


def read_rows(run_dir):
    with open(osp.join(run_dir, "rows.jsonl")) as fh:
        return [_from_legacy(json.loads(l)) for l in fh]


def save_grid(tensors, path, nrow=None, pad=2):
    from torchvision.utils import make_grid, save_image

    if torch.is_tensor(tensors):
        tensors = [tensors]
    x = torch.stack([t.detach().float().cpu() for t in tensors], 1).flatten(0, 1)
    x = (x.clamp(-1, 1) * 0.5 + 0.5)
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    save_image(make_grid(x, nrow=nrow or len(tensors), padding=pad, pad_value=1.0), path)
