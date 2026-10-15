"""Create a separate English poster summary from frozen results, without fitting."""
from pathlib import Path
import hashlib,json,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/poster_matched_v3'
OUT=ROOT/'generated/leakage_poster_summary'
BLUE='#28759b';GRAY='#9aa1a8';INK='#223039';LIGHT='#edf4f7'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(SOURCE/'metrics.csv')
    def row(study,condition,model='SVR',group='all'):
        r=m[(m.study==study)&(m.condition==condition)&(m.model==model)&(m.group==group)]
        assert len(r)==1
        return r.iloc[0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':27,'axes.labelsize':27,'xtick.labelsize':25,'ytick.labelsize':25,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(48,36),facecolor='white')
    texts=[]
    def txt(x,y,s,size=30,color=INK,weight='normal',width=None):
        if width:s='\n'.join(textwrap.fill(line,width) if line else '' for line in s.split('\n'))
        artist=fig.text(x,y,s,fontsize=size,color=color,weight=weight,va='top',linespacing=1.38)
        texts.append(artist);return artist
    def heading(x,y,s):txt(x,y,s,38,BLUE,'bold')
    def axis(rect):
        ax=fig.add_axes(rect);ax.set_ylim(bottom=0);ax.tick_params(length=6,pad=8)
        ax.set_axisbelow(True);ax.grid(axis='y',color='#e3e6e8',lw=.8)
        ax.set_ylabel('Mean error (nominal mm)',labelpad=15)
        return ax
    def bars(ax,labels,rs,colors):
        vals=np.array([r['mean'] for r in rs]);xx=np.arange(len(vals))
        ax.bar(xx,vals,width=.52,color=colors)
        ax.errorbar(xx,vals,yerr=[vals-np.array([r.ci_low for r in rs]),np.array([r.ci_high for r in rs])-vals],fmt='none',ecolor=INK,capsize=6,lw=1.7)
        ax.set_xticks(xx,labels)
        for x,v,r in zip(xx,vals,rs):ax.text(x,r.ci_high+.065,f'{v:.2f}',ha='center',va='bottom',fontsize=25)
        ax.set_ylim(0,max(r.ci_high for r in rs)*1.19)

    txt(.035,.970,'From Leakage Events to QEC Syndromes',78,INK,'bold')
    txt(.035,.933,'Simulation and source localization',46,BLUE)
    txt(.035,.907,'Synthetic detector-level injection  |  Rotated surface code  |  Single-shot coordinate estimation',30)
    fig.add_artist(plt.Line2D([.035,.965],[.887,.887],transform=fig.transFigure,color=BLUE,lw=3))
    x1,x2,x3=.035,.355,.680

    heading(x1,.868,'Introduction')
    txt(x1,.841,'Can QEC detection records reveal where a\nleakage event started?',34,weight='bold')
    txt(x1,.798,'We inject localized leakage-like disturbances into\nsimulated detection records, then estimate the\nsource qubit coordinates from the records alone.',29)

    heading(x1,.730,'1  Leakage-to-syndrome pipeline')
    txt(x1,.700,'1. Sample a noisy surface-code circuit.\n2. Choose a source qubit and an onset round.\n3. Transport and decay leakage on the code graph.\n4. Randomize affected detection records.\n5. Estimate the source coordinates (x, y).',29)
    txt(x1,.608,'Input: responding checks across time.\nTarget: the source data-qubit coordinates.\nTrue location, onset and strength are not inputs.',29,weight='bold')
    txt(x1,.548,'Detector-level injection, not a qutrit circuit model.\nTransport = 0.123/round; decay = 7.63 rounds.',26)

    heading(x1,.499,'2  Longer observation')
    ax=axis([.068,.290,.245,.165])
    rs=[row('observation',f'{r:02d}_rounds') for r in [4,16,32]]
    bars(ax,['4','16','32'],rs,[BLUE]*3);ax.set_xlabel('Observation window (round prefix)',labelpad=14)
    txt(x1,.240,'Longer observation reduces mean error.\nThe gain from 16 to 32 rounds is small.',30,weight='bold')
    txt(x1,.192,'Matched events and onset times; a separately\ntrained SVR for each observation window.',27)

    heading(x1,.138,'Evaluation design')
    txt(x1,.112,'Primary: d=5, 16-round prefix (15 internal).\n864 training + 180 validation events per replica.\n540 test events; 3 replicas; 2 separate shots/event.\nError bars: 95% event-bootstrap intervals.',26)

    heading(x2,.868,'Main result')
    rei=row('models','nominal','Adapted REI')['mean'];svr=row('models','nominal')['mean']
    txt(x2,.834,f'{100*(1-svr/rei):.1f}% lower mean error',51,BLUE,'bold')
    txt(x2,.798,f'{rei:.3f} → {svr:.3f} nominal mm',40,weight='bold')
    txt(x2,.769,'Adapted REI → SVR  |  Same test records',28)
    ax=axis([.386,.568,.250,.158])
    models=['Adapted REI','SVR','Ridge','CNN'];groups=['all','weak','medium','strong']
    for j,(model,color) in enumerate(zip(models,[GRAY,BLUE,'#c8cdd2','#6ba1bf'])):
        rs=[row('models','nominal',model,g) for g in groups];v=np.array([r['mean'] for r in rs]);xx=np.arange(4)+(j-1.5)*.19
        ax.bar(xx,v,width=.18,color=color,label=model)
        ax.errorbar(xx,v,yerr=[v-np.array([r.ci_low for r in rs]),np.array([r.ci_high for r in rs])-v],fmt='none',ecolor=INK,capsize=3,lw=1)
    ax.set_xticks(range(4),['All','Weak','Medium','Strong']);ax.set_ylim(0,4.4)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.20),frameon=False,fontsize=23,handlelength=1.1,columnspacing=1.)
    txt(x2,.532,'CNN has the lowest overall mean error (1.692).\nSVR and CNN are close for strong events.',30,weight='bold')
    txt(x2,.488,'Weak / medium / strong: corruption probabilities\n0.15 / 0.30 / 0.50, not measured leakage strengths.\nREI answers 66.7%; other cases use a fixed fallback.',25)

    heading(x2,.427,'3  Which information matters?')
    ax=axis([.386,.226,.250,.155])
    variants=['F0','F1','F2','N1','N2'];rs=[row('ablation',v) for v in variants]
    bars(ax,variants,rs,[BLUE,BLUE,GRAY,GRAY,BLUE])
    ax.axhline(rei,color=GRAY,lw=2,ls='--');ax.text(4.4,rei+.04,'REI',ha='right',va='bottom',color='#68737a',fontsize=23)
    txt(x2,.198,'F0  Spatial + temporal     F1  Spatial only\nF2  Temporal only            N1  Shuffled check positions\nN2  Shuffled time order',24)
    txt(x2,.140,'Spatial information is the main localization cue.\nShuffling time order also increases error.',30,weight='bold')
    txt(x2,.093,'F1 is close to F0; their paired difference interval\nincludes zero. Each feature variant is retrained.',26)

    heading(x3,.868,'4  Noise and propagation shift')
    ax=axis([.715,.677,.242,.137])
    conditions=['nominal','double_noise','slow_transport']
    for j,(model,color) in enumerate([('Adapted REI',GRAY),('SVR',BLUE)]):
        rs=[row('models' if c=='nominal' else 'robustness',c,model) for c in conditions]
        xx=np.arange(3)+(j-.5)*.30;vals=[r['mean'] for r in rs]
        ax.bar(xx,vals,width=.28,color=color,label=model)
        for x,v in zip(xx,vals):ax.text(x,v+.06,f'{v:.2f}',ha='center',fontsize=24)
    ax.set_xticks(range(3),['Nominal','2× circuit\nnoise','Unseen slow\ntransport']);ax.set_ylim(0,3.5)
    ax.legend(ncol=2,frameon=False,loc='upper center',fontsize=23)
    txt(x3,.621,'SVR remains more accurate than adapted REI\nin both tested shifts, without retraining.',30,weight='bold')
    txt(x3,.577,'Nominal models and quiet reference are frozen.\nUnseen transport = 0.03/round (nominal: 0.123).',26)

    heading(x3,.526,'5  QEC layout as sensor geometry')
    txt(x3,.496,'A  Higher density, fixed area',30,weight='bold')
    ax=axis([.718,.349,.235,.116]);rs=[row('density',c) for c in ['d05_pitch1','d15_pitch0.333']]
    bars(ax,['d=5, pitch=1','d=15, pitch=1/3'],rs,[BLUE,BLUE])
    txt(x3,.310,'Higher density worsens SVR at fixed training size.\nSame area and 25 source positions; more checks\nand background noise. Physical spread is not fixed.',26)

    txt(x3,.252,'B  Wider coverage, fixed spacing',30,weight='bold')
    ax=axis([.718,.112,.235,.108])
    for g,ls in zip(groups,['-','--',':','-.']):
        rs=[row('distance',f'd{d}',group=g) for d in [3,5,7]]
        ax.plot([3,5,7],[r['mean'] for r in rs],marker='o',ms=9,lw=2.5,ls=ls,color=BLUE if g=='all' else GRAY,label=g.capitalize())
    ax.set_xticks([3,5,7]);ax.set_ylim(0,1.9);ax.set_xlabel('Surface-code distance',labelpad=10)
    ax.legend(ncol=4,frameon=False,fontsize=21,loc='upper center',handlelength=1.5,columnspacing=.9)
    txt(x3,.063,'All: 1.121 → 0.974 → 1.092. Wider is not always better.\nCommon nine-source region; separate from the main test.',24)

    fig.add_artist(plt.Line2D([.035,.965],[.030,.030],transform=fig.transFigure,color='#b8c3ca',lw=1))
    txt(.035,.023,'Scope: synthetic data-qubit sources; source sites shared across train/test. Nominal mm is an assumed scale. No measured-response or decoding validation. Gains are specific to the REI comparator.',22)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in texts:
        bbox=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert bbox.x0>=0 and bbox.x1<=1 and bbox.y0>=0 and bbox.y1<=1,(t.get_text(),bbox)
    fig.savefig(OUT/'leakage_results_poster_en.pdf',metadata={'Title':'From Leakage Events to QEC Syndromes: Simulation and Source Localization'})
    fig.savefig(OUT/'leakage_results_poster_en.png',dpi=75)
    plt.close(fig)
    # A local source manifest; main experiment files remain unchanged.
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'metrics.csv',SOURCE/'protocol.json',Path(__file__)]}
    (OUT/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
    print(OUT/'leakage_results_poster_en.pdf')

if __name__=='__main__':main()
