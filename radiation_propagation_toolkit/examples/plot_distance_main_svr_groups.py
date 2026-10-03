"""Single-panel SVR code-distance plot with all strength groups."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
RUN = BASE / 'localization_runs/distance_main_svr_20260923'

def main(run=RUN):
    run = Path(run)
    m = pd.read_csv(run / 'metrics.csv')
    order = [('all', 'All events', '#2F5D8A', 'o'),
             ('weak', 'Weak', '#8A8F98', 's'),
             ('medium', 'Medium', '#8A8F98', '^'),
             ('strong', 'Strong', '#8A8F98', 'D')]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=.12, right=.98, bottom=.17, top=.91)
    for group, label, color, marker in order:
        s = m[m.group == group].sort_values('distance')
        ax.plot(s.distance, s.mean_mm, marker=marker, markersize=7, linewidth=2.6,
                color=color, label=label)
        for x, y in zip(s.distance, s.mean_mm):
            ax.annotate(f'{y:.2f}', (x, y), xytext=(0, 7), textcoords='offset points',
                        ha='center', fontsize=9.5, color=color)
    ax.set_title('SVR localization error vs. code distance', loc='left',
                 fontsize=16, fontweight='bold', color='#17324D', pad=12)
    ax.set_xlabel('Surface-code distance', fontsize=12)
    ax.set_ylabel('Mean localization error (mm)', fontsize=12)
    ax.set_xticks([3, 5, 7]); ax.set_xlim(2.6, 7.4); ax.set_ylim(0, 3.65)
    ax.grid(axis='y', color='#E1E7EC', linewidth=1); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AAB7C2'); ax.spines['bottom'].set_color('#AAB7C2')
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.5, -0.11),
              handlelength=1.8, columnspacing=1.2)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(run / f'distance_main_svr_groups.{ext}', dpi=300 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(run / 'distance_main_svr_groups.png')

if __name__ == '__main__':
    main()
