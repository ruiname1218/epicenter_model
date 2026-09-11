"""Render the saved REI comparison; no fitting or test-set selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
metrics = {(r['candidate'], r['split']): r for r in json.loads((ROOT / 'metrics.json').read_text())}
paired = {(r['candidate'], r['split']): r for r in json.loads((ROOT / 'paired.json').read_text())}
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
names = ['REI適用版', 'CNN単体', 'Ridge', 'SVR']
keys = ['rei_r1_k1024', 'cnn_single', 'fixed_ridge_100.0', 'svr_17']
splits = [('test_id', '通常条件（300イベント）', '#3479ac'),
          ('test_ood', '強い楕円（60イベント）', '#db8731')]
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.subplots_adjust(left=.055, right=.985, top=.76, bottom=.28, wspace=.34)
fig.suptitle('REI適用版と学習モデルの比較［予備的結果］', fontsize=21, y=.98)
fig.text(.5, .88, '同一テスト集合・各イベント2ショット ｜ distance-3 surface code ｜ 観測時間 2.048 ms', ha='center')
for ax, field, title in zip(axes[:2], ['mean_mm', 'p90_mm'], ['① 平均位置誤差', '② 90パーセンタイル誤差（大外れの指標）']):
    for j, (split, label, color) in enumerate(splits):
        values = [metrics[k, split][field] for k in keys]
        bars = ax.bar(np.arange(4) + (j-.5)*.36, values, width=.36, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
    ax.set_xticks(np.arange(4), names)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel('距離誤差 [mm]（小さいほど良い）')
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)

ax = axes[2]
for j, (split, label, color) in enumerate(splits):
    records = [paired[k, split] for k in keys[1:]]
    values = np.array([-r['minus_selected_rei_mm'] for r in records])
    lows = np.array([-r['event_bootstrap_95pct_mm'][1] for r in records])
    highs = np.array([-r['event_bootstrap_95pct_mm'][0] for r in records])
    ax.errorbar(values, np.arange(3)+(j-.5)*.22, xerr=[values-lows, highs-values],
                fmt='o', capsize=4, color=color, label=label)
ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
ax.set_yticks(np.arange(3), names[1:])
ax.invert_yaxis()
ax.set_ylim(2.5, -.5)
ax.set_xlim(-.08, .72)
ax.set_xlabel('REI − 各モデルの平均誤差 [mm]\n正の値ほど改善、0をまたぐ差は不明確')
ax.set_title('③ REIからの改善量と95%信頼区間', fontsize=12, pad=12)
ax.grid(axis='x', alpha=.18)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .16), ncol=2, frameon=False)
fig.text(.055, .12, '要点：強い楕円では平均誤差が改善。一方、通常条件の大外れではREIがRidge・SVRより良い。', fontsize=12)
fig.text(.055, .067, '注：原著REIの入力対応は未確定。学習・較正資源も同一ではない。既に結果を確認したテスト集合での再分析。', fontsize=10, color='#555555')
fig.text(.055, .026, '信頼区間：イベント単位の対応付きbootstrap（10,000回）。多重比較補正・学習集合の変動は含まない。最新MDN・物理照合は未比較。', fontsize=10, color='#555555')
for suffix in ['png', 'pdf', 'svg']:
    output = ROOT / f'rei_comparison_progress.{suffix}'
    fig.savefig(output, dpi=180, facecolor='white')
    print(output)
plt.close(fig)
