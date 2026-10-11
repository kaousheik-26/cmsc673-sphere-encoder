import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .geometry import geodesic_move, normalize, normalized_addition, project_tangent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to cfg.json beside checkpoint or above ckpt/")
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist; no outputs are overwritten")
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 5, 10, 20, 40, 60])
    parser.add_argument("--seed", type=int, default=673)
    parser.add_argument("--class-id", type=int, help="Omit for model's null class conditioning")
    parser.add_argument("--decoder-angle-deg", type=float, default=0,
                        help="Fixed decoder conditioning, independent of traversal angle")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--encoder-from-checkpoint", action="store_true",
                        help="Offline construction for DINOv3 ViT-S/16+; strict checkpoint restores all weights")
    args = parser.parse_args()
    if any(not math.isfinite(a) or abs(a) >= 90 for a in args.angles):
        parser.error("Matched normalized-addition comparison requires finite angles in (-90,90)")
    return args


def build_checkpoint_model(config, checkpoint, weights, offline_encoder):
    """Reuse builder and load_ckpt, including the repository's EMA handling."""
    from sphere.builder import build_model
    from sphere.ema import ModuleEMA
    from sphere.utils import load_ckpt

    config = dict(config)

    config.setdefault("use_qk_norm", False)
    context = nullcontext()
    if offline_encoder:
        from transformers import AutoModel, DINOv3ViTConfig

        name = config.get("load_pretrained_encoder")
        if name not in ("dinov3-vits16plus-pretrain-lvd1689m", "facebook/dinov3-vits16plus-pretrain-lvd1689m"):
            raise ValueError("Offline encoder construction supports only DINOv3 ViT-S/16+")

        encoder_config = DINOv3ViTConfig(
            hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
            intermediate_size=1536, use_gated_mlp=True, hidden_act="silu",
            num_register_tokens=4, patch_size=16, image_size=224,
            rope_theta=100.0, key_bias=False, layer_norm_eps=1e-5,
        )

        def construct_encoder(requested_name, *unused, **kwargs):
            if requested_name != f"facebook/{name.removeprefix('facebook/')}":
                raise ValueError(f"Unexpected encoder requested: {requested_name}")
            return AutoModel.from_config(encoder_config)

        context = patch("sphere.encoder.AutoModel.from_pretrained", side_effect=construct_encoder)
    with context:
        model = build_model(SimpleNamespace(**config), for_train=False)
    ema = ModuleEMA(model) if weights == "ema" else None

    load_ckpt(model, ckpt_path=str(checkpoint), ema_model=ema,
              strict=True, override_model_with_ema=weights == "ema")
    return model.eval().requires_grad_(False)


def latent_stats(z, moved, mode):
    dims = tuple(range(1, z.ndim)) if mode == "global" else (2,)
    a, b = z.double(), moved.double()
    ra = torch.linalg.vector_norm(a, dim=dims)
    rb = torch.linalg.vector_norm(b, dim=dims)
    cosine = (a * b).sum(dims) / (ra * rb)
    return {
        "norm_min": rb.min().item(), "norm_max": rb.max().item(),
        "max_relative_radius_error": ((rb - ra).abs() / ra).max().item(),
        "measured_angle_deg_mean": torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean().item(),
    }


def save_grid(rows, labels, path):
    from PIL import Image, ImageDraw
    import numpy as np

    size = rows[0][0].shape[-1]
    header = 36
    canvas = Image.new("RGB", (len(rows[0]) * size, len(rows) * (size + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, tensor in enumerate(row):
            array = (tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array), (c * size, r * (size + header) + header))
            draw.text((c * size + 4, r * (size + header) + 6), labels[r][c], fill="black")
    canvas.save(path)


@torch.no_grad()
def main():
    args = parse_args()
    config_path = args.config or args.checkpoint.parent / "cfg.json"
    if args.config is None and not config_path.is_file() and args.checkpoint.parent.name == "ckpt":
        config_path = args.checkpoint.parent.parent / "cfg.json"
    config = json.loads(config_path.read_text())
    missing = [str(p) for p in [args.checkpoint, *args.image] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use an allocated GPU node")
    if args.class_id is not None and not (config["cond_generator"] and 0 <= args.class_id < config["num_classes"]):
        raise ValueError("class-id outside the checkpoint's conditioning range")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=(args.output_dir.resolve() / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        print("Building and strictly loading checkpoint", flush=True)
        model = build_checkpoint_model(config, args.checkpoint, args.weights, args.encoder_from_checkpoint)
        model = model.to(args.device)  # keep float32 parameters; optional autocast below
        print("Checkpoint loaded", flush=True)
        from PIL import Image
        import numpy as np
        from sphere.loader import center_crop_arr

        mode = model.spherify_mode
        dtype = getattr(torch, args.dtype)
        autocast = lambda: torch.autocast(args.device, dtype=dtype, enabled=args.dtype != "float32")
        metadata = {
            "checkpoint": str(args.checkpoint.resolve()), "config": str(config_path.resolve()),
            "seed": args.seed, "weights": args.weights, "mode": mode,
            "class_id": args.class_id, "class_conditioning": "null" if args.class_id is None else "explicit",
            "decoder_angle_deg": args.decoder_angle_deg, "dtype": args.dtype,
            "torch": torch.__version__, "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "encoder_from_checkpoint": args.encoder_from_checkpoint,
            "interpretation": "Traversal smoke test, not an intrinsic-property result",
            "images": [],
        }

        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        direction = None
        for index, path in enumerate(args.image):
            with Image.open(path) as image:
                array = np.array(center_crop_arr(image.convert("RGB"), model.decoder.input_size))
            original = torch.from_numpy(array.copy()).permute(2, 0, 1).float().to(args.device) / 255
            x = original.unsqueeze(0) * 2 - 1
            label = None if args.class_id is None else torch.tensor([args.class_id], device=args.device)
            with autocast():
                features = model.encode(x, label)

            z_model = model.f(features.float())
            z = normalize(z_model, mode)
            if direction is None:
                direction = torch.randn(z.shape, generator=generator, device=z.device)
                torch.save(direction.cpu(), args.output_dir / "shared_direction.pt")
            if direction.shape != z.shape:
                raise ValueError("All images must produce the same latent shape")
            tangent = normalize(project_tangent(z, direction, mode), mode)
            ambient = normalize(direction, mode)
            with autocast():
                reconstruction = model.decode(z_model, label, alpha_deg=args.decoder_angle_deg)[0].float()
            rows = [[original, reconstruction], [original, reconstruction]]
            labels = [["Input RGB", "Model reconstruction"], ["Input RGB", "Model reconstruction"]]
            record = {"image": str(path), "latent_shape": list(z.shape),
                      "model_latent": latent_stats(z, z_model, mode),
                      "reconstruction_mse": (reconstruction - original).square().mean().item(),
                      "traversals": []}
            for angle in args.angles:
                geodesic = geodesic_move(z, direction, angle, mode)

                matched = normalized_addition(z, tangent, math.tan(math.radians(angle)), mode)

                baseline = normalized_addition(z, ambient, math.tan(math.radians(angle)), mode)
                entry = {"requested_angle_deg": angle,
                         "geodesic": latent_stats(z, geodesic, mode),
                         "normalized_addition": latent_stats(z, baseline, mode),
                         "ambient_offset_scale": math.tan(math.radians(angle)),
                         "matched_tangent_baseline_max_difference": (geodesic - matched).abs().max().item()}
                for row, edited, method in [(0, geodesic, "Geodesic"), (1, baseline, "Ambient add+norm")]:
                    if not torch.isfinite(edited).all():
                        raise RuntimeError("Nonfinite latent")
                    with autocast():
                        decoded = model.decode(edited, label, alpha_deg=args.decoder_angle_deg)[0].float()
                    if not torch.isfinite(decoded).all():
                        raise RuntimeError("Nonfinite decoder output")
                    entry["geodesic" if row == 0 else "normalized_addition"]["output_min_max"] = [decoded.min().item(), decoded.max().item()]
                    rows[row].append(decoded)
                    labels[row].append(f"{method} {angle:g} deg" if row == 0 else f"{method} s={math.tan(math.radians(angle)):.3f}")
                record["traversals"].append(entry)
            grid_name = f"{index:02d}_{path.stem}_traversal.png"
            save_grid(rows, labels, args.output_dir / grid_name)
            record["grid"] = grid_name
            metadata["images"].append(record)
            print(f"Saved {grid_name}; reconstruction MSE={record['reconstruction_mse']:.6f}", flush=True)
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
