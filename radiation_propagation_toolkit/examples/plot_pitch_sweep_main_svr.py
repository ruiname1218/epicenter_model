"""Plot compact fixed-distance pitch/density sweep for the SVR study."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[2]
RUN=BASE/'localization_runs/pitch_sweep_main_svr_small_20260924'

def main(run=RUN):
    run=Path(run); m=pd.read_csv(run/'metrics.csv')
    levels=['ultra_low','low','standard','high','ultra_high']
    labels=['Ultra-low','Low','Standard','High','Ultra-high']
    order=[('all','All events','#2F5D8A','o'),('weak','Weak','#8A8F98','s'),
           ('medium','Medium','#8A8F98','^'),('strong','Strong','#8A8F98','D')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8.0,4.9)); fig.subplots_adjust(left=.11,right=.98,bottom=.2,top=.9)
    x=range(len(levels))
    for group,label,color,marker in order:
        s=m[m.group==group].set_index('pitch_level').reindex(levels)
        ax.plot(x,s.mean_mm,marker=marker,markersize=7,linewidth=2.5,color=color,label=label)
        for xi,y in zip(x,s.mean_mm): ax.annotate(f'{y:.2f}',(xi,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9,color=color)
    ax.set_xticks(list(x),labels); ax.set_xlabel('Qubit pitch (density condition)'); ax.set_ylabel('Mean localization error (mm)')
    ax.set_ylim(0,3.65); ax.grid(axis='y',color='#E1E7EC'); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.spines['left'].set_color('#AAB7C2');ax.spines['bottom'].set_color('#AAB7C2')
    ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.13),handlelength=1.8,columnspacing=1.2)
    for ext in ('png','pdf','svg'): fig.savefig(run/f'pitch_sweep_main_svr.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(run/'pitch_sweep_main_svr.png')
if __name__=='__main__': main()
