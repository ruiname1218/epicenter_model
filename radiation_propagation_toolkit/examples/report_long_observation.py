"""Report paired duration comparisons, tails, elapsed observation cost and audit."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from qp_ode_simulator.long_observation import verify
from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file

root=Path(sys.argv[1]).resolve();read=lambda path:json.loads(path.read_text());p=verify(root);s=read(root/'selection.json')
m=pd.read_csv(root/'metrics.csv');pred=pd.read_csv(root/'predictions.csv');pairs=read(root/'paired.json')
fit=pd.read_csv(root/'fit_rows.csv');test=pd.read_csv(root/'test_rows.csv')
assert fit.groupby('role').event_uid.nunique().to_dict()=={'train':1080,'validation':180}
assert test.event_uid.nunique()==540 and not set(test.event_uid)&set(fit.event_uid)
assert not set(fit[fit.role=='train'].event_uid)&set(fit[fit.role=='validation'].event_uid)
assert not ((fit.geometry=='elliptical')&(fit.strength_band==2)).any()
assert test.groupby('strength_band').event_uid.nunique().eq(180).all()
for field in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert fit[field].nunique()==test[field].nunique()==1 and fit[field].iloc[0]==test[field].iloc[0]
tests={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in tests:tests[key]+=int(suite.attrib.get(key,0))
assert tests['tests']>0 and tests['failures']==tests['errors']==0

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def nice(a):
    a=a[['model','group','events','mean_mm','within_1mm','p90_mm','p95_mm','over_3mm']].copy()
    a[['within_1mm','over_3mm']]*=100
    return a.round(4).rename(columns={'model':'モデル','group':'強度・形状','events':'イベント数','mean_mm':'平均mm','within_1mm':'1mm以内%','p90_mm':'p90 mm','p95_mm':'p95 mm','over_3mm':'3mm超%'})

tails=[]
for band in [0,1,2]:
    b=pred[(pred.strength_band==band)&(pred.model=='2_relative_SVR1')].sort_values(['event_uid','shot'])
    be=b.error_mm.to_numpy().reshape(-1,2)
    for h in [4,8]:
        a=pred[(pred.strength_band==band)&(pred.model==f'{h}_relative_SVR1')].sort_values(['event_uid','shot'])
        assert a.event_uid.to_list()==b.event_uid.to_list() and a.shot.to_list()==b.shot.to_list()
        ae=a.error_mm.to_numpy().reshape(-1,2)
        idx=np.random.default_rng(2026091105).integers(0,len(ae),(10000,len(ae)))
        for metric,fn,factor in [('within1',lambda z,ax:np.mean(z<=1,axis=ax),100),('p90',lambda z,ax:np.quantile(z,.9,axis=ax),1),('over3',lambda z,ax:np.mean(z>3,axis=ax),100)]:
            boot=[]
            for start in range(0,10000,250):
                ids=idx[start:start+250];boot.extend(factor*(fn(ae[ids].reshape(len(ids),-1),1)-fn(be[ids].reshape(len(ids),-1),1)))
            tails.append(dict(strength_band=band,horizon=h,metric=metric,difference=factor*float(fn(ae,None)-fn(be,None)),ci95=np.quantile(boot,[.025,.975]).tolist(),unit='percentage points' if factor==100 else 'mm'))
_write_json(root/'tail_intervals.json',tails)
incremental=[]
for band in [0,1,2]:
    a=pred[(pred.strength_band==band)&(pred.model=='8_relative_SVR1')].sort_values(['event_uid','shot'])
    b=pred[(pred.strength_band==band)&(pred.model=='4_relative_SVR1')].sort_values(['event_uid','shot'])
    assert a.event_uid.to_list()==b.event_uid.to_list()
    delta=(a.error_mm.to_numpy()-b.error_mm.to_numpy()).reshape(-1,2).mean(1)
    bootstrap=np.random.default_rng(2026091107).choice(delta,(10000,len(delta))).mean(1)
    incremental.append(dict(strength_band=band,difference_8minus4_mm=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist()))
_write_json(root/'incremental_8vs4.json',incremental)
primary=m[m.model.isin([f'{h}_relative_SVR1' for h in [2,4,8]])&m.group.isin(['weak','medium','strong'])]
selected=m[m.model.isin(s['selected'].values())&m.group.isin(['weak','medium','strong'])]
audit=read(root/'audit.json');assert audit['status']=='passed'
center=fit[fit.role=='train'][['epicenter_row','epicenter_col']].mean().to_numpy()
prior_error=np.linalg.norm(test[['epicenter_row','epicenter_col']].to_numpy()-center,axis=1)
prior_weak=float(prior_error[test.strength_band==0].mean())
lines=['# 同一イベントの2・4・8ms観測比較',
    '## 結論',
    '同じ相対空間分布SVR1で、2.048/4.096/8.192msの中平均誤差は2.6511/2.3971/2.2361 mm、強は0.8973/0.8072/0.8042 mm。2→4msで中約9.6%、強約10.0%、2→8msで中約15.7%、強約10.4%低減。中・強の4主比較の98.75%対応区間は全て改善側。',
    '4→8msの追加平均改善は中-0.1610 mm（記述的95%区間[-0.2131,-0.1089]）、強-0.0030 mm（[-0.0345,+0.0276]）。この条件では中は8msまで利益があり、強の平均は4ms付近で頭打ち。3テストブロックすべてで2→4/8msの中・強平均は改善したが、学習データ再抽出や複数独立生成rootの検証ではない。',
    '中の1mm以内率は8.3→11.7→15.3%、p90は4.2628→3.9254→3.7815 mm、3mm超率は36.1→28.9→24.2%。強の1mm以内率は66.7→69.7→69.7%、p90は1.7818→1.5482→1.5203 mm。副指標の区間は以下に示し、点推定の改善をすべて有意と呼ばない。',
    f'弱平均は3.2878→3.2630→3.2241 mmだが、2msとの差95%区間はいずれも0を含み改善不明確。学習座標の固定平均を常に返す基準の弱平均は{prior_weak:.4f} mmで、8msでもこの基準より平均が小さくなっていない。弱の位置推定を解決したとは言えない。',
    '精度と待ち時間の折衷として4msが候補。中を重視し約6msの追加待ちを許容できるなら8msも候補。ただし観測終了後に初めて予測できるため、復号での実用性は別途遅延込みで検証する。既定モデルは変更していない。',
    '## 条件',
    'd5、通常背景ノイズ、同一ハードウェア。新規学習1,080・検証180・未使用テスト540物理イベント、各2ショット。テストは弱/中/強各180。強い楕円は学習・検証から除外し、テストでは強の半分を占める。',
    '1つの8.192msシンドローム記録から先頭2.048/4.096/8.192msを切り出した対応実験。各窓は同じイベント・同じショット・同じ先頭データを共有する。末尾の未来情報を短窓モデルに渡さない。最初の2msの時間区切りを固定し、後半に広い時間帯を追加。',
    '過去の学習2,880イベントの結果とは学習量が異なるため直接比較しない。今回の3観測時間の間では学習イベント集合・モデル候補数は等しい。',
    '## 固定モデル族による主比較：相対空間分布SVR1',table(nice(primary)),
    'SVR C1,epsilon.1,gamma .57/特徴次元。相対空間分布の特徴数は2/4/8msで72/120/168。モデル設定のレシピは共通だが、次元に応じてgammaは変わる。これは固定された実装レシピの観測時間比較であり、追加観測の理論的情報量を測るものではない。',
    '## 主比較の不確実性',
    '4ms−2msと8ms−2msの中・強という4主比較に、98.75%の物理イベント単位bootstrap区間（10,000回、Bonferroni対応の近似区間）。弱および他候補は記述的95%比較。2ショットを独立イベントとして数えない。',
    table(pd.DataFrame([r for r in pairs if r['model'].endswith('relative_SVR1') and r['group'] in ['medium','strong']])),
    '## 各観測時間で検証選択したモデル',
    str(s['selected']),table(nice(selected)),
    '各時間にfull/relative × Ridge/SVR1/SVR10/ETの8候補、計24学習。選択は中・強の検証平均を等重みで最小化。テスト生成前に固定し、テスト最良への差し替えはしない。尾部を制約した選択ではないため平均改善が大外れ改善を意味するとは限らない。',
    '## 大外れ・成功率の対応差',table(pd.DataFrame(tails).round(5)),
    '## 4msから8msへの追加効果',
    '相対SVR1の対応付き平均差。負なら8msが改善。追加の記述的95%区間で、4主比較の補正対象には含めていない。',
    table(pd.DataFrame(incremental).round(5)),
    '## 時間のコスト',
    '最低でも観測待ちに2.048/4.096/8.192msを要し、この後に特徴抽出とモデル推論が必要。4msは約2ms、8msは約6ms余分に待つ。長窓の良い座標を2ms時点で使えるわけではない。',
    'audit.jsonに保存した推論時間は一括predictの1ショット当たり平均で、特徴抽出・読込を除くバッチ処理量の指標。単一ショットの遅延や、実時間で復号に間に合うことは検証していない。',
    '## 適用版REI',table(nice(m[m.model.str.endswith('_REI')&m.group.isin(['weak','medium','strong'])])),
    'REIは各観測終了時点の末尾1,024roundという同じ保持長で評価する。長窓REIが全履歴を累積するわけではなく、長く待つと初期の位置情報を捨てる可能性がある。REIの原著条件やオンライン検出の優劣とは扱わない。回答率・固定中心fallbackはaudit.jsonに保存。',
    '## 全モデル・形状別',table(nice(m)),
    '## 再現性と限界',
    'block_metrics.csvに、独立イベントの3テストブロック（各180イベント、各強度60）の結果を保存。生成rootは1つで、3独立rootではない。学習集合の再抽出や複数学習seedは今回未実施。全強度が存在すると既知の観測窓から位置を推定する課題であり、検出・復号効果は別課題。',
    '長時間での物理モデルの妥当性・別デバイスへの汎化は未確認。8msという長さは設定したQP/伝播モデル内での比較。真の座標・時刻・強度や未来のシンドロームを推論には使わない。',
    '## 監査',
    f'実装テスト{tests["tests"]}件通過。学習/検証/テストのイベント非重複、既存生成seedとのテスト非重複、同一ハードウェア、モデル・データhash、短窓の未来非参照を確認。',
    'protocol.json / selection.json / validation_metrics.csv / metrics.csv / paired.json / tail_intervals.json / block_metrics.csv / audit.json / tests.xml。',
    '実行: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.long_observation <prepare|generate_fit|train|generate_test|evaluate> <出力先>。レポート: examples/report_long_observation.py <出力先>。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
audit.update(tests=tests,hardware_equal=True,split_audit='passed',report_source_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit);print(root/'RESULT_JA.md')
