"""English mean-error bars with explicit truncated axes and break marks."""
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
records = json.loads((ROOT / 'metrics.json').read_text())
metrics = {(r['candidate'], r['split']): r['mean_mm'] for r in records}
keys = ['rei_r1_k1024', 'cnn_single', 'fixed_ridge_100.0', 'svr_17']
labels = ['Adapted REI', 'CNN', 'Ridge', 'SVR']
with (ROOT / 'predictions.csv').open(newline='') as source:
    circular = [r for r in csv.DictReader(source)
                if r['geometry'] == 'circular' and int(r['strength_band']) == 2
                and r['split'] == 'test_id' and r['candidate'] in keys]
for key in keys:
    rows = [r for r in circular if r['candidate'] == key]
    assert len(rows) == 120 and len({r['event'] for r in rows}) == 60
    metrics[key, 'strong_circular'] = sum(float(r['error_mm']) for r in rows) / len(rows)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
fig.subplots_adjust(left=.055, right=.985, bottom=.16, top=.77, wspace=.32)
fig.suptitle('Mean Epicenter Localization Error', fontsize=20, y=.96)
for col, split, title, color in [
    (0, 'test_id', 'Standard test set', '#3479ac'),
    (1, 'strong_circular', 'Strong circular events', '#399579'),
    (2, 'test_ood', 'Strong elliptical events', '#db8731'),
]:
    ax = axes[col]
    values = [metrics[k, split] for k in keys]
    bars = ax.bar(range(4), values, width=.62, color=color, zorder=3)
    ax.bar_label(bars, fmt='%.3f', padding=6, fontsize=12)
    ax.set_xticks(range(4), labels)
    ax.tick_params(axis='x', length=0, pad=10)
    lower, upper = (2.97, 3.085) if col == 0 else (1.5, 2.65)
    ax.set_ylim(lower, upper)
    ax.set_yticks([2.98, 3.00, 3.02, 3.04, 3.06, 3.08] if col == 0 else [1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    ax.set_ylabel('Mean localization error (mm)', labelpad=10)
    ax.set_title(title, fontsize=14, pad=16)
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(True)
    # Mark the omitted lower range on the axis and across each bar.
    for offset in [.018, .04]:
        ax.plot([-.012, .012], [offset-.008, offset+.008],
                transform=ax.transAxes, color='black', linewidth=1.2,
                clip_on=False, zorder=6)
    for x in range(4):
        ax.plot([x-.31, x+.31], [lower+.020*(upper-lower), lower+.038*(upper-lower)],
                color='white', linewidth=3, solid_capstyle='butt', zorder=5)
for ext in ['png', 'pdf', 'svg']:
    path = ROOT / f'rei_mean_comparison_with_circular_en.{ext}'
    fig.savefig(path, dpi=220, facecolor='white')
    print(path)
plt.close(fig)
