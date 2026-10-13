"""Inspect the sphere2 checkpoint to decide how to handle the gated DINOv3 encoder."""
import sys, torch, collections

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("top-level keys:", list(ckpt.keys()))
for mk in ["model", "ema_model"]:
    if mk not in ckpt:
        continue
    sd = ckpt[mk]
    print(f"\n=== {mk}: {len(sd)} tensors ===")
    # group by top two name components
    groups = collections.Counter()
    for k in sd:
        parts = k.split(".")
        groups["/".join(parts[:2])] += 1
    for g, n in sorted(groups.items()):
        print(f"  {g:40s} {n}")
    # does it carry the DINOv3 backbone?
    enc_keys = [k for k in sd if k.startswith("encoder.")]
    print(f"  encoder.* count: {len(enc_keys)}")
    for k in enc_keys[:12]:
        print(f"    {k:60s} {tuple(sd[k].shape)}")
    # look for the backbone sub-module name
    bb = [k for k in enc_keys if "encoder.encoder" in k or "layer" in k or "block" in k]
    print(f"  backbone-like keys: {len(bb)}  e.g. {bb[:4]}")
    break
