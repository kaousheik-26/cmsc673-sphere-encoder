# Reconciliation: the solution plan versus the authors' SE2 code

What `halfsphere/` assumes about Sphere Encoder 2, checked against this repo and the 12
`tomg-group-umd/sphere2` checkpoints on 2026-10-07. Every row cites code (`file:line`) or a
measurement. Checkpoint configs referenced are `workspace/experiments/<job>/cfg.json`.
The plan's solvers S2, S4 and S1 are named Repaint, Ridge Map and Steer here (Solutions 1 to 3).

## R1 to R16

| # | Item | Plan assumed | Found | Consequence for halfsphere |
|---|---|---|---|---|
| R1 | model class | `G1` via `build_model` | `G1` (`sphere/model.py:89`), `sphere.builder.build_model(args, for_train=False)` (`sphere/builder.py:50`) builds from `cfg.json` | `SE2Backbone.from_checkpoint` |
| R2 | checkpoint | `ckpt['model']`, EMA in `ckpt['ema_model']` | keys `model, epoch, ema_model, score_match, score_optimizer`; fp32, no prefixes; `ema_model` is a plain G1 state dict (`sphere/ema.py:98-102`) | load `ema_model` with `strict=True` (all keys match) |
| R3 | distributed at load | `load_ckpt` barriers | `dist.barrier()` unconditionally (`sphere/utils.py:445,503`) | bypassed: single process, no `torch.distributed` |
| R4 | encoder | frozen DINOv3 + trainable linear adapter | frozen DINOv3 ViT-S/16+, **no adapter**: `encoder.out` stays `nn.Identity` (`sphere/encoder.py:48`) because `pretrained_encoder_fix_compression_ratio=False` (`model.py:172-186`); no block unfrozen; 0 trainable params | E is fixed DINOv3 features; H1 should hold strongly |
| R5 | encoder input | maps range, normalises, may resize | `[-1,1]->[0,1]`, resize to 256 (bilinear, antialias) if needed, **clamp [0,1]**, ImageNet mean/std (`encoder.py:196-208`); final LayerNorm with affine removed (`encoder.py:101-104`); CLS + 4 registers dropped (`encoder.py:147-151`) | saturated pixels get zero gradient through E; never clamp outside, E already does |
| R6 | decoder call | `g.decoder(z, y, alpha)`, radians `[B,1,1]` | confirmed (`transformer.py:344-372`, `model.py:265-270`); angle embedded twice in mode `both` (adaLN and token) | adapter converts degrees to radians `[B,1,1]` |
| R7 | decoder output | `[-1,1]`; helpers add `*0.5+0.5` and clamp | head is linear + rearrange to `[B,3,H,W]`, **no tanh, no clamp** (`transformer.py:207-217`, `pixel_head_use_tanh=False`); `G1.decode` returns `x*0.5+0.5` (`model.py:605`) | adapter returns raw unclamped `[-1,1]`-ish (smoke: range -1.48 to 1.48) |
| R8 | angle conditioning | `use_angle_cond`, `max_angle_deg=90` | `use_angle_condition=True`, mode `both`, `noise_sigma_max_angle=90` in all 12 cfgs; `AngleEmbedder` clamps `alpha/90deg` to [0,1] (`layers.py:585-597`) | angles above 90 behave as 90 |
| R9 | spherify | `g.f`, global or local | `partial(vector_rms_norm, zero_mean=False)`, global in all 12 cfgs (`model.py:224-228`); formula `z * rsqrt(mean(z^2) + 1e-5)` in fp32 (`layers.py:36-46`) | `geometry.f` uses the same formula: **bitwise equal** (smoke) |
| R10 | latent shape | `(256, 384)` at 256px | `(256, 384)` for **all 12** checkpoints, including 512px (patch 32, encoder sees a 256px resize) | L = 98,304 everywhere; at 512px n_A/L = 4 |
| R11 | class conditioning | null = `num_classes` | null index `num_classes` (`layers.py:517`, `model.py:502,593`); 1001 / 103 embedding rows; **`y=None` crashes** a class-conditional decoder (`transformer.py:353`); all 12 cfgs `cond_generator=True` | adapter always passes a long tensor |
| R12 | no_grad helpers | `encode, decode, reconstruct, generate, interpolate` | also `edit` and `_log_angle_embed` (`model.py:279,439,590,598,607,652,748`) | adapter calls `g.encoder` / `g.decoder` only |
| R13 | attention | `sdpa_mode` around SDPA | all cfgs `sdpa_fp32`: `_sdpa` opens its **own** `sdpa_kernel([EFFICIENT, MATH])` (`layers.py:375-386`), which overrides an outer `sdpa_kernel(MATH)`; efficient kernel has no forward AD and no double backward (verified). HF DINOv3 defaults to `sdpa`, same problem | `SE2Backbone.differentiable_attention()`: decoder blocks `sdpa_mode="manual"` (`layers.py:387-393`), DINOv3 `set_attn_implementation("eager")`; agreement: decoder 2.4e-6, encoder 4.5e-5 max abs (smoke) |
| R14 | training noise | rotation, stratified alpha | `cos(a) z + sin(a) f(e)`, not re-normalised; radii stratified across the batch plus one sample pinned at 0 and one at 90 deg (`model.py:315-331`, `layers.py:62-119`) | `geometry.rotate` matches |
| R15 | latent consistency | encode decoded image, match clean latent | `1 - cos(f(E(D(z_noisy))), z_clean)` weighted by the **high-angle** weight `w_hi = 1 - sigmoid(-(a - 70)/1.25)` (`loss.py:287-307`, `476-495`); weight **0.1 on ImageNet, 0.0 on Flowers** | R_cyc and the projector are trained properties only for ImageNet, and only above ~70 deg |
| R16 | sampling settings | init 84 / 85.5, cached noise | `sample.py` default 85 (`sample.py:102`); scripts use 84 with cached noise, steps {1,2,4}; reconstruction sweep uses the true class | Repaint schedules start from 80 to 85 deg |

## Further findings

- **Data latents live on a thinner set than the sphere.** The non-affine final LayerNorm makes
  every data token mean 0 and norm sqrt(384) (smoke: token-manifold distance 7e-8). That set
  has dimension N(D-2) = 97,792 inside the sphere's 98,303. Random latents are off it (token
  norms 0.91 to 1.09 sqrt(D)). `geometry.token_manifold_gap` is logged by every solver; Steer has
  an optional `manifold="tokens"` ablation (ours).
- **Reconstruction is semantic, not pixel-exact.** `D(v(x), 0)` keeps content but shifts pose
  slightly: PSNR 18 to 25 dB, LPIPS 0.12 to 0.24 on 4 Flowers-256 val images (smoke run
  `workspace/runs/smoke_se2/20261007-232328-f99069a`). So the observed-half residual at the true
  latent is far from zero: the noiseless fiber at small epsilon may be empty or far from v(x).
- **Training-mode randomness.** Label dropout 0.1 and stochastic depth 0.1 are live in train mode
  (`layers.py:522`, `:707`); the adapter keeps eval mode and asserts it on every call.
- **Equivalence (rule 2.1).** Through adapter primitives only, the flowers-512 seed-99 class-17
  4-step 84-degree sample is **pixel-identical** to the PNG `sample.py` saved, provided the
  harness matches sample.py: weights in bf16 with `channels_last` (`sample.py:326`; without it
  DINOv3's patch-embedding conv differs, max 42/255) and the final `x*0.5+0.5` quantised in bf16
  (in fp32 a third of pixels move by one level). The adapter and the model's own code path are
  bitwise identical in fp32 and in bf16. Run the gate in a fresh process
  (`--only_equivalence True`): after other models ran in the same process, cuDNN picked a
  different bf16 conv algorithm and the PNG match drifted (max 32/255) while adapter and G1 path
  stayed identical to each other.
- **`save_tensors_to_images` writes nothing without torch.distributed** (`sphere/utils.py:172`);
  `halfsphere.records.save_grid` is our replacement.
- **LPIPS loads with `strict=False`** (`sphere/lpips.py:157`): `HalfLPIPS` asserts the five heads
  equal the file.

## Deviations from the reference code in the solution plan

| Where | Reference | Ours | Why |
|---|---|---|---|
| `geometry.f` | `sqrt(L) z / norm(z)` | `z * rsqrt(mean(z^2) + 1e-5)` | bitwise equal to the model's `g.f` |
| `geometry.tangent` | divide by L | divide by `<v,v>` | exact when the eps makes `norm^2` differ from L |
| Ridge Map attention | `sdpa_kernel(MATH)` | `bb.differentiable_attention()` | R13: the decoder overrides the outer context |
| `DummyBackbone` | `nn.MultiheadAttention` | attention written as matmuls | forward-mode AD on CPU through the dummy |
| `SE2Backbone` load | `load_ckpt` with a gloo group | `load_state_dict(ema, strict=True)` | R3 |
| null class | `c=None` passed through | explicit index tensor | R11 |
| Steer `R_cyc` | `encode(x.clamp(-1,1))` | `encode(x)` | E clamps internally; ground rule: no clamp in a differentiable path |
| Steer modes | rgd, adam, geoopt | rgd, adam (+ `manifold="tokens"`) | geoopt not approved; the manifold option is our ablation |
| metrics | pip `lpips`, `torchmetrics` | `sphere/lpips.py`, `sphere/metric.py` features + own KID | no new dependencies |
| outputs | `results/` | `workspace/runs/<exp>/<stamp>-<sha>/` | `results/` is not gitignored |
| completion modes | gamma in {10, 100, 1000}, k=48 | default gamma=10, k=80, plus Ritz residuals | C's spectrum reaches -gamma lambda_max(G_A); at gamma=1000, k=40 Lanczos missed the top modes on the dummy (Ritz 2.1 vs dense 6.9), at gamma=10, k=80 it matched a dense solve to 3 decimals with leakage ~0.02; on real weights residuals <= 6e-3 |
| CPU threads | (unspecified) | `OMP_NUM_THREADS=4` in env.sh | on this box (load ~60) default threading made tiny CPU ops cost ~30 ms each; the repo's own `run.sh` sets 1 |
| Repaint pass counts | counters | solver cost reported as 2(K+1) passes per image | the run counters also include the encodes made for metrics |
