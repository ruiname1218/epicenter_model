"""Build a concise monochrome report from saved results; no experiments rerun."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/poster_matched_v3'
OUT=ROOT/'generated/leakage_research_report_complete_20261009'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(SRC/'metrics.csv')
    def r(study,condition,model='SVR',group='all'):
        s=m[(m.study==study)&(m.condition==condition)&(m.model==model)&(m.group==group)]
        assert len(s)==1
        return s.iloc[0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.5),constrained_layout=True)
    ax=axes[0,0]
    rs=[r('ablation',c) for c in ['F0','F1','F2','N1','N2']]
    ax.bar(range(5),[x['mean'] for x in rs],width=.6,color=['.3','.3','.8','.8','.3'],edgecolor='black',lw=.5)
    ax.errorbar(range(5),[x['mean'] for x in rs],yerr=[[x['mean']-x.ci_low for x in rs],[x.ci_high-x['mean'] for x in rs]],fmt='none',ecolor='black',capsize=2,lw=.8)
    ax.set_xticks(range(5));ax.set_xticklabels(['F0','F1','F2','N1','N2']);ax.set_title('(a) Feature ablation',loc='left',fontsize=10)
    ax=axes[0,1]
    for model,style,mark in [('Adapted REI','--','s'),('SVR','-','o'),('CNN',':','^')]:
        vals=[r('models' if c=='nominal' else 'robustness',c,model)['mean'] for c in ['nominal','double_noise','slow_transport']]
        ax.plot(range(3),vals,color='black',ls=style,marker=mark,ms=4,label=model)
    ax.set_xticks(range(3));ax.set_xticklabels(['Nominal','2x noise','Slow']);ax.set_ylim(0,3.6)
    ax.legend(frameon=False,fontsize=8,ncol=2,loc='upper left');ax.set_title('(b) Frozen-model evaluation',loc='left',fontsize=10)
    for ax,study,conditions,xx,title,xlabel in [
        (axes[1,0],'observation',['04_rounds','16_rounds','32_rounds'],[4,16,32],'(c) Observation length','Round prefix'),
        (axes[1,1],'distance',['d3','d5','d7'],[3,5,7],'(d) Spatial coverage','Code distance')]:
        rs=[r(study,c) for c in conditions]
        ax.errorbar(xx,[x['mean'] for x in rs],yerr=[[x['mean']-x.ci_low for x in rs],[x.ci_high-x['mean'] for x in rs]],fmt='o-',color='black',capsize=3,ms=4,lw=1)
        ax.set_xticks(xx);ax.set_xlabel(xlabel);ax.set_title(title,loc='left',fontsize=10);ax.set_ylim(bottom=0)
    ax=axes[1,1]
    ax.lines[0].set_label('All')
    for group,style,marker in [('weak','--','s'),('medium',':','^'),('strong','-.','D')]:
        ax.plot([3,5,7],[r('distance',f'd{d}',group=group)['mean'] for d in [3,5,7]],
                color='.45',ls=style,marker=marker,ms=3,lw=1,label=group.capitalize())
    ax.set_ylim(0,1.95);ax.legend(frameon=False,ncol=2,fontsize=7,loc='upper right')
    for ax in axes.flat:
        ax.set_ylabel('Mean error (nominal mm)');ax.set_ylim(bottom=0);ax.grid(axis='y',color='.88',lw=.5);ax.set_axisbelow(True)
    fig.savefig(OUT/'controlled_comparisons.png',dpi=300);fig.savefig(OUT/'controlled_comparisons.pdf');plt.close(fig)

    body=ParagraphStyle('Body',fontName='Times-Roman',fontSize=10.4,leading=13,spaceAfter=6)
    small=ParagraphStyle('Small',parent=body,fontSize=8.7,leading=11,spaceAfter=6)
    heading=ParagraphStyle('Heading',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=8,spaceAfter=6,keepWithNext=True)
    title=ParagraphStyle('Title',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=9)
    story=[];plain=[]
    def p(text,style=body):
        story.append(Paragraph(text,style));plain.append(re.sub('<[^>]+>','',text))
    def h(text):p(text,heading)
    def table(data,widths):
        t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),8.2),('LEADING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),4),('ALIGN',(1,0),(-1,-1),'RIGHT'),('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black)]))
        story.append(t);story.append(Spacer(1,7));plain.extend([' | '.join(map(str,row)) for row in data])

    p('Leakage Source Localization<br/>from Simulated QEC Detection Records',title)
    p('Research report  |  9 October 2026  |  Complete leakage-poster results',small)
    h('Summary')
    p('This study evaluated source-coordinate estimation from simulated quantum error correction (QEC) detection records. In a rotated surface-code benchmark, mean localization error was 2.695 for adapted REI, 1.843 for support vector regression (SVR), 1.892 for ridge regression and 1.692 for a convolutional neural network (CNN). Spatial information was the main localization cue. Longer observation reduced error, whereas wider coverage and higher density did not consistently improve SVR. These findings describe a synthetic detector-level injection model, not measured hardware performance.')
    h('1. Experimental design')
    p('A leakage-like disturbance was injected into noisy Stim detection records using the local ReserveQ-derived simulator. Leakage population moved on the circuit connectivity graph and decayed over time; affected detector outcomes were randomized. Nominal transport was 0.123 per round and the decay time was 7.63 rounds. The main setting used code distance d = 5 and the 15 complete internal detector layers in a 16-round prefix. Gate, measurement and reset error probabilities were 0.0005, 0.001 and 0.001, respectively.')
    p('Each event began at one of 25 data-qubit locations, with onset sampled from rounds 0, 1 and 2. Corruption probabilities of 0.15, 0.30 and 0.50 defined weak, medium and strong <i>proxies</i>, with 180 test events per group. They do not represent measured leakage amplitudes or radiation energies. Inputs contained only detection records and the known check layout; true source coordinates, onset and strength were not provided at inference.')
    p('Each of three training replicas used 864 training and 180 validation events. Models shared 540 held-out test events, each with two independently predicted shots. Splits were event-disjoint, but source locations were shared across splits. Hyperparameters and CNN checkpoints were selected on validation data. SVR used a radial-basis-function kernel; ridge used linear regression with L2 regularization. CNN used two spatial convolution layers, with time layers as channels. Adapted REI used frequency filtering, a spatial filter and squared normalized frequency weights; its history length was selected on validation data.')
    p('Error is the Euclidean distance between predicted and true coordinates, averaged over shots and training replicas, not an ensemble prediction. All distances use <b>nominal mm</b>: centered Stim coordinates multiplied by a chosen spatial scale. This scale is not calibrated to hardware. Confidence intervals use 2,000 event-level bootstrap resamples after averaging within events; they are conditional on the three fitted replicas.',small)
    h('2. Results: model comparison')
    p('<b>Table 1.</b> Localization error under nominal conditions. Lower error is better. Intervals are 95% event-bootstrap intervals. The final column is the fraction of predictions within 1 nominal mm.',small)
    data=[['Method','All','95% interval','Weak','Medium','Strong','Within 1']]
    for model in ['Adapted REI','SVR','Ridge','CNN','Constant']:
        z=r('models','nominal',model)
        data.append([model if model!='Constant' else 'Fixed location',f'{z["mean"]:.3f}',f'{z.ci_low:.3f}-{z.ci_high:.3f}',*[f'{r("models","nominal",model,g)["mean"]:.3f}' for g in ['weak','medium','strong']],f'{100*z.within_1:.1f}%'])
    table(data,[82,42,96,48,53,48,57])
    p('CNN had the lowest overall mean error among the primary models; SVR and CNN were close for strong events. The paired SVR-minus-REI difference was -0.852 nominal mm (95% interval: -0.952 to -0.755). This is a comparison with the implemented adaptation, not a claim to reproduce or outperform the original REI study.',small)
    story.append(PageBreak())

    h('2. Results (continued)')
    p('<b>REI coverage.</b> Adapted REI returned a location for 66.7% of nominal test shots. Otherwise, it used the mean training-source coordinate, as specified by the comparison protocol. On the same subset of shots where REI returned an answer, mean error was 1.824 for REI, 1.564 for SVR and 1.549 for CNN. The overall difference therefore cannot be attributed entirely to fallback predictions.')
    story.append(Image(str(OUT/'controlled_comparisons.png'),width=484,height=307))
    p('<b>Figure 1.</b> Controlled comparisons. (a) F0: spatial and temporal features; F1: spatial only; F2: temporal only; N1: event-specific check-position shuffling; N2: event-specific time shuffling. Each variant was retrained. (b) Nominal models and the quiet reference were frozen under doubled circuit noise or unseen transport of 0.03 per round. (c) SVR was retrained on matched 4-, 16- and 32-round prefixes. (d) Fixed-spacing layouts used the same nine source locations; all-event and weak, medium and strong proxy results are shown. Error bars are 95% event-bootstrap intervals, shown for all groups in (a) and for all-event means in (c,d). Coverage uses a different source region from the main comparison.',small)
    p('<b>Features.</b> Mean errors were 1.843 (F0), 1.901 (F1), 3.878 (F2), 3.875 (N1) and 2.093 (N2). Removing spatial information or shuffling check positions approached the fixed-location baseline. Time shuffling increased error by 0.249 (95% interval: 0.195 to 0.304). The spatial-only difference from F0 had an interval that included zero (-0.003 to 0.118).')
    p('<b>Noise and transport.</b> With doubled circuit noise, mean error was 2.804 for REI, 2.215 for SVR, 2.221 for ridge and 2.090 for CNN. Under unseen slow transport, the corresponding errors were 2.555, 1.657, 1.719 and 1.561. Learned models retained lower mean error in these two shifts, without retraining. This does not establish general robustness to other simulator or hardware changes.')
    p('<b>Observation and coverage.</b> SVR error decreased from 2.145 to 1.843 to 1.788 across 4-, 16- and 32-round prefixes. The final improvement was modest: -0.055 (95% interval: -0.098 to -0.013). For the common nine-source region, errors at d = 3, 5 and 7 were 1.121, 0.974 and 1.092. Wider coverage therefore did not yield a monotonic improvement.')
    group_results=[]
    for group in ['weak','medium','strong']:
        values=', '.join(f'{r("distance",f"d{d}",group=group)["mean"]:.3f}' for d in [3,5,7])
        group_results.append(f'{group}: {values}')
    p('At d = 3, 5 and 7, respectively, the strength-proxy errors were ' + '; '.join(group_results) + '. Each group improved from d = 3 to 5 and worsened from d = 5 to 7. Thus, the non-monotonic all-event result also appeared within each strength group.',small)
    story.append(PageBreak())

    h('2. Results (continued): fixed-area density')
    p('Both density settings retained the same 10 by 10 nominal-mm bounding box, 25 source coordinates and 864 training events per replica. The d = 15 layout with pitch 1/3 contains all d = 5 source coordinates exactly (Table 2).')
    p('<b>Table 2.</b> Fixed-area density comparison. Counts refer to the 15 internal rounds. Excess is the average observed detection count minus the independent quiet-reference expectation; it is not a count of labeled leakage errors.',small)
    d=pd.read_csv(SRC/'density_signal_diagnostic.csv')
    table([['Quantity','d = 5, pitch = 1','d = 15, pitch = 1/3'],
           ['Mean localization error','1.843','2.758'],['Number of checks','24','224'],['SVR features','120','1,120'],
           ['Expected background detections',f'{d.iloc[0].expected_background_detections:.2f}',f'{d.iloc[1].expected_background_detections:.2f}'],
           ['Excess over quiet expectation',f'{d.iloc[0].excess_over_quiet:.2f}',f'{d.iloc[1].excess_over_quiet:.2f}']],[232,116,138])
    h('3. Discussion')
    p('The ablations support spatial response patterns as the main cue for source localization in this benchmark. Preserving time order also helped, but spatial-only features remained competitive. These comparisons change feature dimensions and fitted regularization as well as information content, so they do not isolate a purely causal contribution of each feature.')
    p('Higher density increased background counts and input dimension much more than the excess detection count. This may explain the loss of SVR accuracy at fixed training size, but no single cause was isolated. Fixed graph transport per round also spans a different physical distance when pitch changes. This is not a pure density comparison at fixed physical propagation, nor evidence that denser hardware is intrinsically worse.')
    p('Lower error under slower transport is consistent with information remaining near the source, but this explanation was not tested separately. Observation gains diminished beyond 16 rounds; changed time-bin boundaries and separately fitted models also contribute to this comparison.')
    p('<b>Supplementary baseline check.</b> A separate, untuned evaluation of the earlier weighted-centroid method gave mean error 2.082 on the same nominal test records. It is not part of the primary comparison table. SVR reduced error by 11.5% relative to this centroid, compared with 31.6% relative to adapted REI. The latter percentage is comparator-specific and should not be presented as improvement over analytic localization methods in general.')
    h('4. Limitations and conclusion')
    p('This detector-level injection study does not validate qutrit or hardware behavior. Test events shared source locations and most generator settings with training. Unseen locations, ancilla or multiple sources, event detection, other codes and decoding benefits remain untested. The exploratory intervals do not fully capture training variability or multiple comparisons. Different observation lengths and source priors also preclude ranking radiation against leakage by absolute error.')
    p('The results support learned localization within this simulator, with spatial patterns as the main cue. Next tests should use held-out source locations, alternative injection mechanisms and learning curves for the high-density layout before drawing broader conclusions.')
    h('Data and implementation')
    p('Archives: <font name="Courier">poster_matched_v3</font> and <font name="Courier">legacy_centroid_reference_v3</font>. The simulator derives from ReserveQ-leakage commit 441718d; adapted REI uses the radiation toolkit function <font name="Courier">rei_rates</font>. Predictions, protocols, validation selections and hashes are retained. Coordinate mapping, event splits, paired targets and metrics were checked. No new training or measured-response comparison was performed for this report.',small)

    def footer(canvas,doc):
        canvas.setStrokeGray(.7);canvas.setLineWidth(.4);canvas.line(54,43,A4[0]-54,43)
        canvas.setFont('Helvetica',8);canvas.setFillGray(.3)
        canvas.drawString(54,30,'Leakage source localization | Synthetic benchmark')
        canvas.drawRightString(A4[0]-54,30,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'leakage_research_report_en.pdf'),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=46,bottomMargin=55,title='Leakage Source Localization from Simulated QEC Detection Records')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'report_text_en.txt').write_text('\n\n'.join(plain)+'\n')
    # Explicit coverage inventory: every quantitative series in the leakage poster.
    coverage=[]
    for model in ['Adapted REI','SVR','Ridge','CNN']:
        for group in ['all','weak','medium','strong']:
            coverage.append(dict(panel='Main result',location='Table 1',**r('models','nominal',model,group).to_dict()))
    for study,conditions,location in [('ablation',['F0','F1','F2','N1','N2'],'Figure 1a and text'),
                                      ('observation',['04_rounds','16_rounds','32_rounds'],'Figure 1c and text'),
                                      ('density',['d05_pitch1','d15_pitch0.333'],'Table 2')]:
        for condition in conditions:coverage.append(dict(panel=study,location=location,**r(study,condition).to_dict()))
    for condition in ['nominal','double_noise','slow_transport']:
        for model in ['Adapted REI','SVR']:
            coverage.append(dict(panel='Noise and propagation shift',location='Figure 1b and text',**r('models' if condition=='nominal' else 'robustness',condition,model).to_dict()))
    for d in [3,5,7]:
        for group in ['all','weak','medium','strong']:
            coverage.append(dict(panel='Wider coverage',location='Figure 1d and text',**r('distance',f'd{d}',group=group).to_dict()))
    assert len(coverage)==44
    pd.DataFrame(coverage).to_csv(OUT/'poster_result_coverage.csv',index=False)
    paths=[SRC/'metrics.csv',SRC/'protocol.json',SRC/'paired_svr_contrasts.csv',SRC/'density_signal_diagnostic.csv',ROOT/'results/legacy_centroid_reference_v3/metrics.csv',Path(__file__)]
    (OUT/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print(OUT/'leakage_research_report_en.pdf')

if __name__=='__main__':main()
