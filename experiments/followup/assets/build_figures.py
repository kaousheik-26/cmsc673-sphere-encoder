"""Rebuild five follow-up figures from saved grids and the findings tables.

Run from the repository root with Python, matplotlib, numpy, and Pillow installed.
Generated source grids remain in the ignored intrinsics output directory.
"""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'experiments/intrinsics/outputs/followup-20261027'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def table(section):
    text = (OUT.parent / 'FOLLOWUP_FINDINGS.md').read_text().split(section, 1)[1].split('\n## ', 1)[0]
    return [[c.strip() for c in line.strip('|').split('|')] for line in text.splitlines() if line.startswith('| ')][2:]

# Preserve source grids byte-for-byte. No model outputs are synthesized.
shutil.copyfile(SOURCE / 'sky-mask/fresh-addition-unseen.png', OUT / '02-fresh-sky-audit.png')
shutil.copyfile(SOURCE / 'diode-first-check/depth-geodesic-shared-673_shared_673.png', OUT / '03-measured-depth-transfer.png')

# The source grid has 256px tiles, with 36px caption strips before each row.
# Select one sky-positive scene, one correct empty mask, and two false positives.
source = Image.open(SOURCE / 'sky-mask/fresh-addition-unseen.png')
assert source.size == (1280, 2336), source.size
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for r, (row, label) in enumerate([(6, 'Sky present'), (0, 'Correct empty mask'), (2, 'Indoor false positive'), (4, 'Bright background false positive')]):
    for c, column in enumerate([0, 1, 3, 4]):
        tile = source.crop((column*256, row*292+36, (column+1)*256, row*292+292))
        axes[r, c].imshow(tile)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0: axes[r, c].set_title(['RGB', 'Teacher sky mask', 'Raw shared edit', 'Thresholded mask'][c])
        if c == 0: axes[r, c].set_ylabel(label, fontsize=10)
fig.suptitle('Binary sky extraction: useful structure and visible transfer failures', fontsize=15)
fig.text(.5, .015, 'Fresh new categories • addition, seed 673 • selected qualitative examples, not aggregate performance', ha='center', fontsize=10)
fig.tight_layout(rect=(.02,.04,1,.96))
fig.savefig(OUT/'01-sky-successes-and-failures.png', dpi=150); plt.close(fig)

rows = table('## Binary sky masks:')
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(3)
for row in rows:
    label = row[0].replace('geodesic: ', 'Geodesic ').replace('addition: ', 'Addition ')
    control = 'random' in label or label in ['zero edit', 'training template']
    ax.plot(x, [float(v) for v in row[1:]], marker='o', label=label, linestyle='--' if control else '-', alpha=.65 if control else 1, linewidth=1.4 if control else 2)
ax.set_xticks(x, ['Original unseen\n24 images', 'Fresh familiar\n96 images', 'Fresh new categories\n96 images'])
ax.set_ylabel('Pooled sky-class IoU ↑'); ax.set_ylim(0, .9); ax.grid(axis='y', alpha=.2)
ax.set_title('Sky masks: learned extraction improves on controls; category transfer remains limited', pad=15)
ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=9)
fig.text(.08,.015,'Source: FOLLOWUP_FINDINGS.md • trained/random methods: means over three seeds • fixed controls evaluated once', fontsize=9)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(OUT/'04-sky-performance.png', dpi=160); plt.close(fig)

rows = table('## Depth: extra complexity')
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
variants = ['shared', 'adaptive-size', 'subspace4', 'subspace8', 'budget4', 'budget8']
for ax, method in zip(axes, ['geodesic', 'addition']):
    selected = {row[1]: row for row in rows if row[0] == method}
    x = np.arange(6)
    ax.plot(x,[float(selected[v][4]) for v in variants], 'o-', label='Fresh familiar (96 images)', color='#2171b5')
    ax.plot(x,[float(selected[v][5]) for v in variants], 's-', label='Fresh new categories (96 images)', color='#cb5b24')
    ax.set_xticks(x, variants, rotation=35, ha='right'); ax.set_title(method.capitalize()); ax.grid(axis='y', alpha=.2)
    ax.set_ylim(.025,.068); ax.legend(fontsize=8, loc='upper left')
axes[0].set_ylabel('Teacher-target RGB MSE ↓')
fig.suptitle('Depth: added editor complexity gives no consistent transfer gain', fontsize=14)
fig.text(.08,.025,'Source: FOLLOWUP_FINDINGS.md • means over three seeds, rounded to five decimals • no uncertainty intervals shown', fontsize=9)
fig.tight_layout(rect=(0,.09,1,.93)); fig.savefig(OUT/'05-depth-complexity.png', dpi=160); plt.close(fig)
