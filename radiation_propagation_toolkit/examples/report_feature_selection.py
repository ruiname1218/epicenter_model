"""Feature x kernel-width factorial report; no test-based adoption."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from qp_ode_simulator.feature_selection_study import verify,BASELINE,GAMMAS
from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file

root=Path(sys.argv[1]).resolve();read=lambda path:json.loads(path.read_text());p,s=verify(root)
m=pd.read_csv(root/'metrics.csv');pred=pd.read_csv(root/'predictions.csv');pairs=read(root/'paired.json')
fit=pd.read_csv(root/'fit_rows.csv');test=pd.read_csv(root/'test_rows.csv')
assert fit.groupby('role').event_uid.nunique().to_dict()=={'train':1080,'validation':180}
assert not set(fit[fit.role=='train'].event_uid)&set(fit[fit.role=='validation'].event_uid)
assert not set(fit.event_uid)&set(test.event_uid)
assert not ((fit.geometry=='elliptical')&(fit.strength_band==2)).any()
for key in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert fit[key].nunique()==test[key].nunique()==1 and fit[key].iloc[0]==test[key].iloc[0]
unique=test.drop_duplicates('event_uid');assert len(unique)==540 and unique.generation_seed.nunique()==540
for seed in p['test_seeds']:
    count=unique[unique.test_seed==seed].groupby(['geometry','propagation_law','epicenter_region','strength_band']).size()
    assert len(count)==36 and (count==5).all()
assert pred.groupby('model').event_uid.nunique().eq(540).all() and pred.groupby('model').size().eq(1080).all()
tests={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in tests:tests[key]+=int(suite.attrib.get(key,0))
assert tests['tests']>0 and tests['failures']==tests['errors']==0

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def nice(df):
    a=df[['model','group','events','mean_mm','within_1mm','p90_mm','p95_mm','over_3mm']].copy()
    a[['within_1mm','over_3mm']]*=100
    return a.round(4).rename(columns={'model':'モデル','group':'強度・形状','events':'イベント数','mean_mm':'平均mm','within_1mm':'1mm以内%','p90_mm':'p90 mm','p95_mm':'p95 mm','over_3mm':'3mm超%'})

tails=[]
for band in [0,1,2]:
    b=pred[(pred.strength_band==band)&(pred.model==BASELINE)].sort_values(['event_uid','shot']);be=b.error_mm.to_numpy().reshape(-1,2)
    for name in sorted(set([s['selected'],s['unconstrained']])):
        a=pred[(pred.strength_band==band)&(pred.model==name)].sort_values(['event_uid','shot'])
        assert a.event_uid.to_list()==b.event_uid.to_list() and a.shot.to_list()==b.shot.to_list()
        ae=a.error_mm.to_numpy().reshape(-1,2);idx=np.random.default_rng(2026091129).integers(0,len(ae),(10000,len(ae)))
        for metric,fn,factor in [('within1',lambda z,ax:np.mean(z<=1,axis=ax),100),('over3',lambda z,ax:np.mean(z>3,axis=ax),100),('p90',lambda z,ax:np.quantile(z,.9,axis=ax),1)]:
            boot=[]
            for start in range(0,10000,250):
                ids=idx[start:start+250];boot.extend(factor*(fn(ae[ids].reshape(len(ids),-1),1)-fn(be[ids].reshape(len(ids),-1),1)))
            tails.append(dict(model=name,strength_band=band,metric=metric,difference=factor*float(fn(ae,None)-fn(be,None)),ci95=np.quantile(boot,[.025,.975]).tolist(),unit='percentage points' if factor==100 else 'mm'))
_write_json(root/'tail_intervals.json',tails)

factorial=m[m.model.str.contains('__g')&m.group.isin(['weak','medium','strong'])].copy()
factorial[['features','gamma_key']]=factorial.model.str.split('__',expand=True)
factorial.to_csv(root/'factorial_metrics.csv',index=False)
main_names=list(dict.fromkeys([BASELINE,s['selected'],s['unconstrained'],'REI','Prior']))
lines=['# 4ms特徴選択とカーネル幅の分離実験',
    '## 結論',
    '主選択はlog変換＋gamma=.57/72。新規テストで基準の相対分布SVRから中平均2.4137→2.3889 mm（約1.0%減、97.5%差区間[-0.0464,-0.0031]）。強平均0.8650→0.8505 mmは区間[-0.0432,+0.0115]が0を含み、明確な改善とは言えない。中・強とも3テスト生成rootで平均の改善方向は一致した。',
    '中p90は4.0710→3.9802 mmと点推定で改善したが、差95%区間[-0.1626,+0.0706]で不確実。強p90は1.6830→1.7051 mm、3mm超率は0→0.56%と点推定で悪化。平均の小幅改善を全指標の改善としない。弱平均3.2760→3.2731 mmで、弱の問題を解決していない。既定モデルは置き換えていない。',
    '副選択は負の増分を残すsigned正規化＋gamma=.57/120。中2.3514 mm（基準比約2.6%減、記述的95%差区間[-0.1066,-0.0166]）で有望だが、弱3.3052 mmに悪化し主選択の制約を満たさなかった。強は0.8520 mmで改善不明確。さらにテスト最良の別gammaへ差し替えることはしない。',
    'カーネル幅を固定しても、強では全特徴より相対分布のみが3gamma全てで良かった。gamma .57/265では全特徴1.0135対相対分布0.9160 mm、.57/120では1.3197対0.8650、.57/72では1.6392対0.8746 mm。したがって、以前の相対分布の利益はgamma変更だけでは説明できない。ただし次元による距離分布やモデル容量の違いは残る。',
    '同じgamma=.57/120で、relative_level（発火量5特徴を追加）は中2.4065・強0.8553 mmと小差。時間差分だけは中3.1745・強2.8454 mmと大幅に悪化。前半だけは中2.6897・強0.9244、後半だけは中2.5461・強1.0293 mmで、両方を保持する基準が良かった。top60は中2.4484・強0.9180、PCA32は中2.5946・強1.0240 mmで、単純な特徴削減も中・強を改善しなかった。',
    '解釈: 相対的な空間分布を保持したまま値の表し方を調整するのは有望。一方、時間差分だけにすると持続する位置情報を失う可能性があり、相関上位特徴や分散上位成分だけでは位置推定に必要な情報を残せない場合がある。これは今回のモデル/設定での結果で、唯一の因果説明や全ての特徴選択法の否定ではない。',
    '## 条件と事前選択',
    '約4.096ms、d5、同一背景/デバイス。前回の8ms研究から同じ学習1,080・検証180物理イベントの先頭4msを再利用。特徴選択を固定後、新規540イベント（各強度180、3独立生成root、各2ショット）で共通比較した。',
    f'主選択 `{s["selected"]}`、制約なし副選択 `{s["unconstrained"]}`。基準 `{BASELINE}`。中・強の検証平均を等重みで最小化し、各p90が基準+.05mm以内、弱平均が基準+.02mm以内という制約を主選択に適用した。テストを見て選択を変更していない。',
    '## 新規未使用テストの主結果',table(nice(m[m.model.isin(main_names)&m.group.isin(['weak','medium','strong'])])),
    '## 主比較の不確実性',
    '主選択−基準の中・強平均に97.5%イベントbootstrap区間（2主比較のBonferroni対応、10,000回）。副候補、特徴別・gamma別の多数の差は記述的95%で、多重比較補正済みの発見とは扱わない。',
    table(pd.DataFrame([r for r in pairs if r['model'] in [s['selected'],s['unconstrained']] and r['reference']==BASELINE])),
    '## 比較した特徴',
    '- full: 6時間帯の増分＋相対分布＋固定区切り、265特徴。relative: 発生前を除く5時間帯の相対分布、120特徴。',
    '- relative_level: 相対分布120＋時間帯ごとの正の増分平均をlog1p変換した5特徴。',
    '- early: 最初の2msの相対分布72。late: 2〜4msの相対分布48。delta: 相対分布の隣接時間差分96。earlyは4ms記録から抽出するが未来情報を使わない。',
    '- signed: 負の増分も残し、平均絶対値+.003で正規化。sqrt: 正の増分を平方根化して正規化。log: 従来相対分布をlog1p変換。いずれも120特徴。',
    '- top60: 学習データでx/y座標との単変量Fスコアを平均し、上位60相対特徴だけ採用。pca32: 学習データで標準化・PCAを学習し32成分に圧縮。検証/テストの正解を特徴選択に使わない。',
    '## カーネル幅を固定した比較',
    f'全11特徴に同じ3つのgamma値を交差させた: {GAMMAS}。C=1、epsilon=.1、各特徴は学習データで標準化。別途Ridge alpha1000も全特徴で評価。計44モデル。特徴数ごとに自動でgammaを変えて比較するのではなく、同じgamma列内で特徴を比較し、同じ特徴行内でgammaを比較する。',
    '同じgammaでも次元・表現によってサンプル間距離分布が変わる。従って全てのモデル容量や正則化効果を完全に等しくしたという意味ではなく、「gamma数値の変更だけが理由か」を調べる対照。特徴ごとの無制限なハイパーパラメータ最適化でもない。']
for group in ['weak','medium','strong']:
    lines += [f'### {group}平均誤差mm',table(factorial[factorial.group==group].pivot(index='features',columns='gamma_key',values='mean_mm').reset_index().round(5))]
lines += ['## 対応付き特徴差・gamma差',table(pd.DataFrame(pairs)),
    '## 大外れと1mm以内率の対応差',table(pd.DataFrame(tails).round(5)),
    '## 全モデル・形状別',table(nice(m)),
    '## 再現性・限界',
    'test_seed_metrics.csvに3独立生成root別の結果。学習データ集合の再抽出や複数学習seedは未検証。SVR/Ridge/PCAは今回の設定では決定的。強い楕円は学習・検証に含めず、テストの強の半分を占める。',
    '過去の2,880学習イベントのモデルと学習量が異なる。検証集合は以前にも利用したものなので過適応リスクを明記し、新規テストで確認した。真の座標・強度・時刻・形状は推論に渡さない。',
    'REIは適用版、末尾1,024round保持で4ms終了時に評価。学習方式は4msの複数時間帯と正解付き学習、独立平常較正を使う。原著実装/条件の優越性、イベント検出、復号効果や実機汎化は未検証。',
    '## 監査・成果物',
    f'実装テスト{tests["tests"]}件通過。基準特徴の以前の4ms特徴との一致、学習限定の特徴削減、イベント分離、同一ハードウェア、3root×36層均等、モデル/データhashを確認。',
    'protocol.json / selection.json / validation_metrics.csv / metrics.csv / factorial_metrics.csv / paired.json / tail_intervals.json / test_seed_metrics.csv / reduction.joblib / audit.json / tests.xml。既定モデルは自動置換していない。',
    '実行: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.feature_selection_study <prepare|train|generate|evaluate> <出力先>。レポート: examples/report_feature_selection.py <出力先>。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
audit=read(root/'audit.json');assert audit['status']=='passed';audit.update(tests=tests,hardware_equal=True,balanced_strata=True,split_audit='passed',report_source_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit);print(root/'RESULT_JA.md')
