"""Build the results page for a run: page/<run>/index.html + copied media. Prints the files map as JSON."""
import json, os, os.path as osp, shutil, sys

ROOT = "/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text"
run = sys.argv[1] if len(sys.argv) > 1 else "text5c_6k"
ev = f"{ROOT}/runs/{run}/eval"
out = f"{ROOT}/page/{run}"
os.makedirs(f"{out}/a", exist_ok=True); os.makedirs(f"{out}/i", exist_ok=True)
r = json.load(open(f"{ev}/metrics.json"))
meta = json.load(open(f"{ROOT}/data/meta.json"))
log = [json.loads(l) for l in open(f"{ROOT}/runs/{run}/log.jsonl")]
evals = [l for l in log if "test_rec_mel_err_db_raw" in l]

NAMES = {"chainsaw": "Chainsaw", "church_bell": "Church bell", "helicopter": "Helicopter", "rain": "Rain", "siren": "Civil defense siren"}
files = {}


def media(src, kind):
    name = osp.basename(src)
    dst = f"{kind}/{name}"
    shutil.copy(src, f"{out}/{dst}")
    files[dst] = osp.relpath(f"{out}/{dst}", ROOT)
    return dst


def pct(x):
    return f"{100 * x:.0f}%"


rec = r["recon"]
main = next(g for g in r["generation"] if (g["steps"], g["cfg_deg"]) == (r["main_setting"]["steps"], r["main_setting"]["cfg_deg"]))
best = max(r["generation"], key=lambda g: g["clap_acc_train_caption"] + g["clap_acc_unseen_caption"])
ref = r["clap_acc_reference"]

# ---------- generation cards ----------
caps = {(c, k): p for c, k, p in r["prompts"]}
gen_html = []
for c in r["classes"]:
    cols = []
    for kind, label in (("train_caption", "Training caption"), ("unseen_caption", "Unseen caption")):
        img = media(f"{ev}/img/gen_{c}_{kind}.png", "i")
        auds = "".join(
            f'<div class="aud"><span class="n">{j + 1}</span><audio controls preload="none" src="{media(f"{ev}/wav/gen_{c}_{kind}_{j}.wav", "a")}"></audio></div>'
            for j in range(3))
        cols.append(f'''<div class="gcol"><p class="kind">{label}</p><p class="prompt">&ldquo;{caps[(c, kind)]}&rdquo;</p>
<img src="{img}" alt="Four generated mel spectrograms for: {caps[(c, kind)]}" loading="lazy"><div class="auds">{auds}</div></div>''')
    acc = main["clap_acc_per_class_train_caption"][c]
    gen_html.append(f'''<article class="gen"><header><h3>{NAMES[c]}</h3><span class="chip">CLAP picks this class: {pct(acc)}</span></header>
<div class="gcols">{"".join(cols)}</div></article>''')

# ---------- reconstruction cards ----------
rec_html = []
for c in r["classes"]:
    img = media(f"{ev}/img/recon_{c}.png", "i")
    row = "".join(
        f'<div class="aud lab"><span>{lab}</span><audio controls preload="none" src="{media(f"{ev}/wav/recon_{c}_{k}.wav", "a")}"></audio></div>'
        for k, lab in (("original", "Original"), ("vocoder", "True mel → vocoder"), ("released", "Released SE2"), ("finetuned", "Fine-tuned SE2")))
    rec_html.append(f'''<article class="rec"><h3>{NAMES[c]}</h3>
<img src="{img}" alt="Mel spectrograms: true, released SE2, fine-tuned SE2" loading="lazy">
<p class="legend">Rows: true mel · released SE2 · fine-tuned SE2</p><div class="auds">{row}</div></article>''')

# ---------- sampler table ----------
sampler_rows = "".join(
    f'<tr{" class=hl" if g is main else ""}><td>{g["steps"]}</td><td>{g["cfg_deg"]:g}°</td><td>{pct(g["clap_acc_train_caption"])}</td>'
    f'<td>{pct(g["clap_acc_unseen_caption"])}</td><td>{g["nn_train_db_mean"]:.2f}</td><td>{g["within_prompt_diversity_db"]:.2f}</td></tr>'
    for g in r["generation"])
curve_rows = "".join(f'<tr><td>{e["step"]:,}</td><td>{e["test_rec_mel_err_db_raw"]:.2f}</td><td>{e["test_rec_mel_err_db_ema"]:.2f}</td></tr>' for e in evals)

html = f'''<title>SE2 Text-to-Audio</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: single reading column of findings, then wide sample grids; spectrogram (magma) palette as the accent world */
:root {{
  --bg: #f7f6f9; --surface: #ffffff; --fg: #1d1724; --muted: #645b70; --line: #e2dde8;
  --accent: #b5367a; --accent-2: #e8833a; --hl: #fbeef4;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #14111a; --surface: #1d1925; --fg: #efeaf4; --muted: #a79db3; --line: #332d3d;
  --accent: #f07ab4; --accent-2: #f4a261; --hl: #2c1d2a; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #14111a; --surface: #1d1925; --fg: #efeaf4; --muted: #a79db3; --line: #332d3d;
  --accent: #f07ab4; --accent-2: #f4a261; --hl: #2c1d2a; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 400 1.0625rem/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 2.5rem 4rem; }}
.wrap {{ max-width: 1120px; margin: 0 auto; display: grid; gap: 3rem; }}
.col {{ max-width: 760px; }}
h1, h2, h3 {{ font-family: var(--display); text-wrap: balance; line-height: 1.15; margin: 0; }}
h1 {{ font-size: clamp(2rem, 5vw, 3rem); font-weight: 800; letter-spacing: -0.02em; }}
h2 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: .75rem; }}
h3 {{ font-size: 1.15rem; font-weight: 700; }}
p {{ margin: .5rem 0; }}
.eyebrow {{ font: 600 .78rem/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin-bottom: .9rem; }}
.lede {{ font-size: 1.2rem; color: var(--muted); max-width: 62ch; }}
.band {{ height: 10px; border-radius: 5px; background: linear-gradient(90deg, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf); margin-top: 1.5rem; max-width: 760px; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
.kpi {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem; display: grid; gap: .25rem; }}
.kpi b {{ font: 800 1.9rem/1.1 var(--display); color: var(--fg); font-variant-numeric: tabular-nums; }}
.kpi span {{ color: var(--muted); font-size: .92rem; }}
ul.find {{ padding-left: 1.2rem; display: grid; gap: .45rem; margin: .5rem 0; }}
.tbl {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .95rem; }}
th, td {{ text-align: left; padding: .55rem .8rem; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th {{ font: 600 .74rem/1.2 var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
tr:last-child td {{ border-bottom: 0; }}
tr.hl td {{ background: var(--hl); }}
td.good {{ color: var(--accent); font-weight: 600; }}
.note {{ color: var(--muted); font-size: .93rem; }}
code, .mono {{ font-family: var(--mono); font-size: .88em; }}
.pipe {{ display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; font: 400 .85rem/1.3 var(--mono); }}
.pipe span {{ background: var(--surface); border: 1px solid var(--line); padding: .35rem .6rem; border-radius: 6px; }}
.pipe i {{ color: var(--accent); font-style: normal; }}
.gens, .recs {{ display: grid; gap: 1.25rem; }}
.gen, .rec {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem; display: grid; gap: .8rem; min-width: 0; }}
.gen header {{ display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; justify-content: space-between; }}
.chip {{ font: 600 .78rem/1 var(--mono); color: var(--accent); border: 1px solid var(--accent); border-radius: 99px; padding: .35rem .6rem; }}
.gcols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.2rem; }}
.gcol {{ min-width: 0; display: grid; gap: .45rem; align-content: start; }}
.kind {{ font: 600 .72rem/1 var(--mono); letter-spacing: .07em; text-transform: uppercase; color: var(--muted); margin: 0; }}
.prompt {{ font-weight: 600; margin: 0; }}
.gen img, .rec img {{ width: 100%; border-radius: 6px; display: block; }}
.auds {{ display: grid; gap: .35rem; }}
.aud {{ display: flex; align-items: center; gap: .5rem; min-width: 0; }}
.aud .n {{ font: 600 .8rem var(--mono); color: var(--muted); width: 1.2rem; }}
.aud.lab span {{ font-size: .88rem; color: var(--muted); width: 9.5rem; flex: none; }}
audio {{ width: 100%; min-width: 0; height: 34px; }}
.recs {{ grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }}
.legend {{ font-size: .85rem; color: var(--muted); margin: 0; }}
a {{ color: var(--accent); }}
:focus-visible {{ outline: 2px solid var(--accent-2); outline-offset: 2px; }}
@media (max-width: 480px) {{ .aud.lab {{ flex-wrap: wrap; }} .aud.lab span {{ width: 100%; }} }}
</style>
<div class="wrap">
<section class="col">
  <p class="eyebrow">Sphere Encoder 2 · AudioSet · run {run}</p>
  <h1>Text-conditioned sound generation in the SE2 latent space</h1>
  <p class="lede">The released SE2 image model (large, ImageNet 256px) was fine-tuned on mel spectrograms of 100 AudioSet clips from 5 classes, with a CLAP text embedding in place of the class label. It now generates sounds from a text prompt, including prompts it never saw in training.</p>
  <div class="band" role="img" aria-label="Magma colour scale used for the spectrograms"></div>
</section>

<section class="kpis">
  <div class="kpi"><b>{pct(main["clap_acc_train_caption"])}</b><span>Generated clips that CLAP assigns to the prompted class (training captions; 2 steps, no CFG)</span></div>
  <div class="kpi"><b>{pct(best["clap_acc_unseen_caption"])}</b><span>Same, for unseen captions ({best["steps"]} steps, CFG {best["cfg_deg"]:g}°)</span></div>
  <div class="kpi"><b>{rec["released"]["mel_err_db"]:.1f} → {rec["finetuned"]["mel_err_db"]:.1f} dB</b><span>Mel error on 25 held-out clips, released SE2 → fine-tuned</span></div>
  <div class="kpi"><b>{pct(ref["real_test_audio"])}</b><span>CLAP accuracy on real held-out recordings, for reference</span></div>
</section>

<section class="col">
  <h2>What we found</h2>
  <ul class="find">
    <li><b>Text conditioning works.</b> With 2 sampling steps and no guidance, CLAP classifies {pct(main["clap_acc_train_caption"])} of generated clips as the prompted class. Real recordings score {pct(ref["real_test_audio"])}.</li>
    <li><b>It generalises past the training captions.</b> Prompts such as &ldquo;a tornado warning siren&rdquo; or &ldquo;a rainstorm&rdquo; reach {pct(main["clap_acc_unseen_caption"])} with no guidance and {pct(best["clap_acc_unseen_caption"])} with {best["steps"]} steps and a {best["cfg_deg"]:g}° angle CFG.</li>
    <li><b>Reconstruction of unseen clips improved</b> from {rec["released"]["mel_err_db"]:.2f} dB (released) to {rec["finetuned"]["mel_err_db"]:.2f} dB mel error (PSNR {rec["released"]["mel_psnr"]:.1f} → {rec["finetuned"]["mel_psnr"]:.1f}). For scale, the vocoder alone adds about 2.1 dB.</li>
    <li><b>Some memorisation, no copies.</b> Generated mels are {main["nn_train_db_mean"]:.1f} dB from their nearest training crop on average, closer than new real clips of the same classes ({r["nn_train_db_real_test"]:.1f} dB) but far from an exact copy (0 dB). More sampling steps give more varied and less train-like outputs.</li>
    <li><b>Weak spot:</b> fine-tuned reconstructions lose some class identity in CLAP&rsquo;s eyes ({pct(ref["finetuned_recon"])} vs {pct(ref["vocoded_test_mel"])} for the true mel), because fine detail is smoothed.</li>
  </ul>
</section>

<section>
  <h2>Generated samples</h2>
  <p class="note col">Each prompt below is decoded from 8 fixed noise seeds (the same seeds for every prompt); the picture shows 4 of them as mel spectrograms (time left to right, 8.9 s; low frequency at the bottom). Audio is the BigVGAN vocoder on the generated mel. Setting: 2 forward steps, initial angle 85°, no CFG.</p>
  <div class="gens">{"".join(gen_html)}</div>
</section>

<section>
  <h2>Sampler settings</h2>
  <p class="note col">40 clips per row (5 classes × 8 seeds) for each caption type. CLAP accuracy = 5-way zero-shot classification of the generated audio. Distance to train = mean |dB| to the nearest 8.9 s crop of any training clip (real held-out clips: {r["nn_train_db_real_test"]:.2f} dB). Diversity = mean pairwise |dB| between samples of the same prompt. Highlighted row = the setting used for the samples above.</p>
  <div class="tbl"><table>
    <thead><tr><th>Steps</th><th>Angle CFG</th><th>CLAP acc, train caption</th><th>CLAP acc, unseen caption</th><th>Distance to train (dB)</th><th>Diversity (dB)</th></tr></thead>
    <tbody>{sampler_rows}</tbody></table></div>
</section>

<section>
  <h2>Reconstruction of held-out clips</h2>
  <div class="tbl col"><table>
    <thead><tr><th>Model (25 held-out clips)</th><th>Mel error (dB) ↓</th><th>Mel PSNR ↑</th></tr></thead>
    <tbody>
      <tr><td>Released SE2</td><td>{rec["released"]["mel_err_db"]:.2f}</td><td>{rec["released"]["mel_psnr"]:.2f}</td></tr>
      <tr class="hl"><td>Fine-tuned SE2</td><td class="good">{rec["finetuned"]["mel_err_db"]:.2f}</td><td class="good">{rec["finetuned"]["mel_psnr"]:.2f}</td></tr>
      <tr><td>Fine-tuned SE2, decoder also given the caption</td><td>{rec["finetuned_textcond"]["mel_err_db"]:.2f}</td><td>{rec["finetuned_textcond"]["mel_psnr"]:.2f}</td></tr>
    </tbody></table></div>
  <p class="note col">Held-out error during training (raw / EMA weights):</p>
  <div class="tbl col"><table><thead><tr><th>Step</th><th>Raw (dB)</th><th>EMA (dB)</th></tr></thead><tbody>{curve_rows}</tbody></table></div>
  <p class="note col">The held-out error flattens after step 1,000, so longer training on 100 clips mostly increases memorisation.</p>
  <div class="recs" style="margin-top:1.25rem">{"".join(rec_html)}</div>
</section>

<section class="col">
  <h2>Setup</h2>
  <div class="pipe"><span>AudioSet wav 16 kHz</span><i>→</i><span>mono, 22.05 kHz</span><i>→</i><span>log-mel 80 × 768 (8.9 s)</span><i>→</i><span>256×256 image</span><i>→</i><span>SE2 encoder → sphere latent 16×16×384</span><i>→</i><span>decoder + CLAP text</span><i>→</i><span>BigVGAN → audio</span></div>
  <ul class="find">
    <li><b>Data:</b> AudioSet-strong train clips where the class covers ≥ 8.5 s of the 10 s and the clip has ≤ 3 labels. 5 classes × 20 train + 5 held-out clips. Each class has 3 training captions; a 4th caption per class is used only at test time.</li>
    <li><b>Image layout:</b> 3 strips of 80 mel bands × 256 frames stacked in one 256×256 grey image (exactly invertible); one global loudness scale (log 1e-5 to 1.5) so generated images map back to absolute levels.</li>
    <li><b>Conditioning:</b> CLAP (laion/clap-htsat-unfused) text embedding → 2-layer MLP → the decoder&rsquo;s adaLN conditioning vector, initialised to the released null-class embedding; 10% caption dropout trains a null embedding for CFG.</li>
    <li><b>Training:</b> from the released large-fd-lite-256 EMA weights; encoder and decoder both trained (encoder lr × 0.1); the released losses without the score and FD terms (L1 + LPIPS + ConvNeXt at low noise angles, latent consistency at high angles). 6,000 steps, batch 64 on 4× RTX A6000, 68 min. Random 8.9 s crops and ±3 dB gain.</li>
    <li><b>Vocoder:</b> NVIDIA BigVGAN v2 22 kHz / 80 band / fmax 8 kHz. Its own round trip (true mel → audio → mel) is about 2.1 dB.</li>
  </ul>
  <p class="note">Code and outputs: <span class="mono">/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text</span> (scripts/, runs/{run}/).</p>
</section>
</div>
'''
open(f"{out}/index.html", "w").write(html)
json.dump(files, open(f"{out}/files.json", "w"), indent=0)
print(len(files), "media files")
