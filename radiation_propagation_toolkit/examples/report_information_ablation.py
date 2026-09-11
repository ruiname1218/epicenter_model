"""Separate controlled information interventions from observational associations."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from qp_ode_simulator.information_ablation import verify
from qp_ode_simulator.strength_benchmark import summary
from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file

root=Path(sys.argv[1]).resolve();read=lambda path:json.loads(path.read_text());p,s=verify(root)
rows=pd.read_csv(root/'test_rows.csv');pred=pd.read_csv(root/'predictions.csv');m=pd.read_csv(root/'metrics.csv');pairs=read(root/'paired.json')
source=pd.read_csv(Path(p['source'])/'rows.csv');train=source[source.role=='train']
tests={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in tests:tests[key]+=int(suite.attrib.get(key,0))
assert tests['tests']>0 and tests['failures']==tests['errors']==0
assert not set(rows.event_uid)&set(source.event_uid)
for key in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert rows[key].nunique()==1 and rows[key].iloc[0]==source[key].iloc[0]

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def nice(frame):
    a=frame[['model','group','events','mean_mm','within_1mm','p90_mm','over_3mm']].copy()
    a[['within_1mm','over_3mm']]*=100
    return a.round(4).rename(columns={'model':'モデル','group':'強度・形状','events':'イベント数','mean_mm':'平均mm','within_1mm':'1mm以内%','p90_mm':'p90 mm','over_3mm':'3mm超%'})

combined=pred.merge(rows,on=['event_uid','shot','strength_band','geometry','test_seed'],validate='many_to_one')
conditions=['epicenter_region','geometry','propagation_law','event_onset_ms','axis_ratio','maximum_distance_mm','initial_lambda_mm','qp_peak_generation_rate_per_us','apparent_speed_m_per_s','diffusion_coefficient_mm2_per_ms']
diagnostics=[];bininfo={}
for field in conditions:
    if pd.api.types.is_numeric_dtype(train[field]):
        edges=np.unique(np.quantile(train[field].dropna(),[1/3,2/3]));bininfo[field]=edges.tolist()
        values=pd.cut(combined[field],[-np.inf,*edges,np.inf],duplicates='drop').astype(str)
    else:values=combined[field].astype(str)
    a=combined.assign(condition_bin=values)
    if field=='apparent_speed_m_per_s':a=a[a.propagation_law=='ballistic']
    if field=='diffusion_coefficient_mm2_per_ms':a=a[a.propagation_law=='diffusive']
    for (name,band,value),g in a[a.model.isin(['Current','full__SVR','relative_only__SVR','time_average__SVR','window512__SVR','window1024__SVR'])].groupby(['model','strength_band','condition_bin']):
        diagnostics.append(dict(model=name,strength_band=int(band),condition=field,bin=value,events=g.event_uid.nunique(),**summary(g.error_mm)))
diagnostics=pd.DataFrame(diagnostics);diagnostics.to_csv(root/'condition_metrics.csv',index=False);_write_json(root/'condition_bin_edges.json',bininfo)
# Boundary diagnostic: no-event-yet and differing post-impact observation time.
window=[]
for model,end in [('window512__SVR',.512),('window1024__SVR',1.024),('full__SVR',2.048)]:
    a=combined[combined.model==model].copy();a['status']=np.where(a.event_onset_ms>=end,'not_started','started')
    a['post_ms']=np.maximum(end-a.event_onset_ms,0)
    for (band,status),g in a.groupby(['strength_band','status']):
        window.append(dict(model=model,strength_band=int(band),event_status=status,events=g.event_uid.nunique(),post_impact_ms_mean=g.post_ms.mean(),**summary(g.error_mm)))
pd.DataFrame(window).to_csv(root/'window_event_status.csv',index=False)

core=m[m.group.isin(['weak','medium','strong'])&m.model.isin(['Current','REI',s['selected'],s['unconstrained']])]
svr=m[(m.model.str.endswith('__SVR'))&m.group.isin(['weak','medium','strong'])]
lines=['# 精度を決める観測時間・特徴情報・物理条件の分析',
    '## 結論',
    '最も有望だったのは、相対的な空間分布72特徴だけのSVR。新規テストで元の169特徴SVRに対し、強の平均誤差が0.9725→0.8484 mm（約12.8%減、対応差の記述的95%区間[-0.1658,-0.0847]）、1mm以内率が61.25→70.69%、p90が1.8533→1.6039 mmとなった。強平均の改善方向は独立テスト生成root3個全てで一致。',
    'これは現行最良方式から12.8%改善したという意味ではない。Currentの強0.8481 mmと相対SVR0.8484 mmは差が不明確（差95%区間[-0.0271,+0.0269]）。中ではCurrent2.6700→相対SVR2.6223 mmだが、p90は4.1235→4.2207 mmで悪化。弱平均も3.2485→3.3187 mmで悪化した。検証で採用制約を満たさず、現行方式を維持した判断を変更しない。',
    '同じSVRで、相対分布を除くと強1.2018 mm、時間変化を除くと1.0990 mm、場所ごとの差を除くと3.2749 mm。相対空間分布と時間情報の利用が、この学習レシピでは精度に寄与する。チェック別較正を全体平均に替える影響は強0.9725→0.9934 mmと比較的小さかった。',
    '観測時間0.512→1.024→2.048msで、同じ形式のSVRの強平均は2.7035→1.4422→0.9725 mm、中平均は3.2534→2.9825→2.6322 mm。0.512msは発生前に終了する例を含むが、既に発生した強イベントだけでも2.5519 mmであり、単に未発生例だけが短窓悪化の理由ではない。',
    '解釈: 位置推定には「総量」だけでなく「どこが相対的に多いか」と十分な観測時間が重要。この設定では不要な特徴を減らす方が有利な場合がある。ただし特徴次元とSVRカーネル幅も変わっているため、「絶対強度が有害」「ノイズ除去が唯一の原因」とまでは断定できない。次の確認候補は、相対分布SVRのカーネル幅を固定した対照と、弱信号時の保守的なモデルへの切り替え。',
    '## 実験設計',
    '学習2,880イベント・検証360イベントは固定。8種類の入力表現×Ridge/SVR/ExtraTreesの計24モデルを再学習。検証選択とモデルを固定してから、新規1,080イベント（弱/中/強各360、各2ショット、独立生成root3個）を全条件で対応比較した。',
    'Currentは前回の中専用SVRに変更したThreeHard、3seed座標アンサンブル。今回のETはseed41の単体であり、アンサンブルと同じ計算予算ではない。Ridge/SVRは決定的。各特徴の効果はまず同じモデル族内で比較する。',
    f'主選択: `{s["selected"]}`、制約なし副選択: `{s["unconstrained"]}`。中・強の平均を等重み最小化し、各p90がCurrent+.02mm以内、弱平均が+.03mm以内という検証制約を適用した。',
    '## 主比較',table(nice(core)),
    table(pd.DataFrame([r for r in pairs if r['model']==s['selected'] and r['reference']=='Current' and r['group'] in ['medium','strong']])),
    '## 情報を変えた比較：SVR',table(nice(svr)),
    '- full: 4時間帯の平常からの増分96特徴＋相対空間分布72特徴＋固定時間区切り1特徴。',
    '- excess_only: 増分96特徴だけ。relative_only: 相対空間分布72特徴だけ。',
    '- time_average: 全観測時間の平均増分24特徴。時間変化を除く。no_spatial: 24チェックを平均した4時間帯だけ。場所ごとの差を除く。',
    '- global_quiet: チェックごとの平常較正を全体平均に置換し、fullと同形式の特徴を再計算。全て学習データだけでスケーラーを学習。',
    '- window512/window1024: 同じ観測開始時点から約0.512/1.024msだけ観測し、4時間帯特徴を作って再学習。固定区切り339roundは変更しない。真の発生時刻で切り出さない。',
    'SVRのgammaは.57/特徴次元、C=1、epsilon=.1。Ridge alpha1000、ET256本/leaf3。特徴を除くと次元・カーネル距離・正則化の効き方も変わるため、情報そのものの理論的寄与ではなく、この固定レシピでの実用的寄与を測っている。',
    '## 対応付き差',
    '各族のfullを基準にした平均差。正なら情報を変更した方式で悪化。物理イベント単位10,000回bootstrap。主選択の中・強だけ97.5%区間、それ以外は多数の記述的95%比較であり、多重比較補正済みの発見とは扱わない。',
    table(pd.DataFrame(pairs)),
    '## 発生前に観測が終わる例',table(pd.DataFrame(window).round(4)),
    '短窓はイベント発生前に終了する例や、到達・広がりを十分観測できない例を含む。全例の性能と発生済み例の性能を両方示す。発生済み例だけを実運用性能として報告しない。長窓の予測を早い時点で利用できるとも主張しない。',
    '## 物理条件と誤差の関連',
    'Currentでは中の平均が生成基準領域のinside2.0361、edge2.6575、outside3.3165 mm。強でも0.6508、0.7848、1.1086 mmの順。強の円0.7378、楕円0.9583 mm。周辺領域と強い楕円は重点的な検証・学習追加候補。ただし真のチップ外ではない。中のballistic2.6957、diffusive2.6444 mmで、以前の検証とは大小関係が逆だったため、「diffusiveは常に難しい」とは結論しない。',
    'condition_metrics.csvに強度帯×位置/形状/伝播則/発生時刻/軸比/到達範囲/初期lambda/ピーク生成率/速度/拡散係数で分けた結果を保存。数値の区切りは学習集合の三分位から固定。速度はballistic、拡散係数はdiffusiveだけで評価。条件間の交絡があり、この表だけで因果関係とは言わない。',
    table(diagnostics[(diagnostics.model=='Current')&(diagnostics.strength_band.isin([1,2]))&diagnostics.condition.isin(['epicenter_region','geometry','propagation_law'])].round(4)),
    '位置区分inside/edge/outsideは生成基準領域[-3,3]mmに対するラベル。d5の物理qubit配置の外側という意味ではない。',
    '## 全モデル・形状別',table(nice(m)),
    '## 再現性・限界',
    'test_seed_metrics.csvに3生成root別の全結果。観測や特徴の比較では同じ物理イベントを再利用しており、条件数をサンプル数に掛けない。学習データ集合の再抽出、複数ET学習seed、未学習ハードウェア、背景ノイズの物理的変更、復号効果は今回の範囲外。',
    '既存の検証集合を再利用している。特徴・窓は検証で比較後に固定し、新規テストで選び直していない。全強度を報告し、弱・大外れの悪化を隠さない。主選択がCurrentの場合、今回の候補による採用基準を満たす改善は見つからなかったことを意味する。',
    '## 成果物・監査',
    f'実装テスト{tests["tests"]}件通過。過去生成seed非重複、イベント分離、モデル/データhash、同一ハードウェアを確認。',
    'protocol.json / selection.json / validation_metrics.csv / metrics.csv / paired.json / condition_metrics.csv / condition_bin_edges.json / window_event_status.csv / test_seed_metrics.csv / tests.xml / audit.json。',
    '再実行: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.information_ablation <prepare|train|generate|evaluate> <出力先>。レポート: examples/report_information_ablation.py <出力先>。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
audit=read(root/'audit.json');audit.update(tests=tests,hardware_equal=True,report_source_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit);print(root/'RESULT_JA.md')
