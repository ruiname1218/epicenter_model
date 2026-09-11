"""Audit specialist train/test separation and report practical vs oracle routing."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file
from qp_ode_simulator.specialist_study import POOLS, PROFILES

root=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text())
p=read(root/'protocol.json'); s=read(root/'selection.json')
source=Path(p['source']); parent=Path(p['parent']); oldest=Path(read(parent/'protocol.json')['parent'])
old=pd.DataFrame([read(path) for folder in [source,parent,oldest] for path in (folder/'events').glob('*.json')])
fresh=pd.DataFrame([read(path) for path in (root/'events').glob('*.json')])
assert len(fresh)==720 and fresh.event_uid.nunique()==720 and fresh.generation_seed.nunique()==720
assert not set(fresh.generation_seed)&set(old.generation_seed)
assert not fresh.is_control.any()
cells=fresh.groupby(['geometry','propagation_law','epicenter_region','strength_band']).size()
assert len(cells)==36 and (cells==20).all()
for field in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert fresh[field].nunique()==1 and fresh[field].iloc[0]==old[field].iloc[0]
rows=pd.read_csv(source/'rows.csv',float_precision='round_trip'); train=rows[rows.role=='train']
assert not set(train.event_uid)&set(old[old.role!='train'].event_uid)
assert not ((rows.geometry=='elliptical')&(rows.strength_band==2)).any()
for pool,info in s['experts'].items():
    expected=train[train.strength_band.isin(POOLS[pool])]
    assert set(info['training_uids'])==set(expected.event_uid)
    assert info['train_events']==expected.event_uid.nunique()
totals={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in totals:totals[key]+=int(suite.attrib.get(key,0))
assert totals['tests']>0 and totals['failures']==totals['errors']==0
audit=read(root/'audit.json'); audit.update(all_historical_seed_overlap=0,hardware_equal=True,balanced36=True,
    expert_training_pools='passed',old_holdouts_not_in_training=True,tests=totals,report_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit)

m=pd.read_csv(root/'metrics.csv'); paired=read(root/'paired.json')
groups=['ordinary','weak','strong_circle','strong_ellipse']; titles=['通常全体','弱イベント','強い円形','強い楕円形']


def table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def means(names):
    records=[]
    for group,title in zip(groups,titles):
        r={'条件':title}
        for name,label in names:
            r[label]=f'{m[(m.group==group)&(m.model==name)].iloc[0].mean_mm:.4f}'
        records.append(r)
    return table(pd.DataFrame(records))


names=[('parent','旧SVR+ET'),(s['primary'],'検証全体選択'),(s['weak'],'弱重視選択'),(s['strong'],'強重視選択'),('prior','固定位置'),('REI','適用REI')]
lines=['# d5: 弱・中・強イベント専門家と切り替え',
    '## 結論（新規テスト）',
    '以下の選択は全て新規720イベントの生成前に固定。通常600、弱240、強い円120、強い楕円120イベント。各2ショットを別々に推定し、統計区間は物理イベント単位でまとめる。単位mm、小さいほど良い。',
    means(names)]
for key,title in [('primary','主評価'),('weak','弱重視・副評価'),('strong','強重視・副評価')]:
    group='ordinary' if key=='primary' else 'weak' if key=='weak' else 'strong_circle'
    r=next(a for a in paired if a['model']==s[key] and a['reference']=='parent' and a['group']==group)
    lo,hi=r['ci95']; judgment='改善側の区間' if hi<0 else '悪化側の区間' if lo>0 else '改善は明確でない'
    lines.append(f'- {title}: `{s[key]}`。旧モデルとの差 {r["difference_mm"]:+.4f} mm、記述的95%区間 [{lo:+.4f}, {hi:+.4f}]（{judgment}）。')
lines+=['## 専門家の学習',
    '旧モデルと同じ2,880学習イベント・169特徴量。専門家はそれぞれの強度範囲だけで学習し、その範囲の検証で選択。強い楕円は全学習・検証選択から除外。',
    table(pd.DataFrame([{'専門家':pool,'学習イベント':info['train_events'],'検証イベント':info['validation_events'],'選択モデル':info['selected'],
                         '対象検証誤差mm':round(info['scores'][info['selected']],5)} for pool,info in s['experts'].items()])),
    '- 弱/中/強の真ラベルは生成強度帯。弱[1e-10,1e-9]、中[1e-9,1e-8]、強[1e-8,3e-7] /us（生成率proxy）。学習・採点には使うが実用版の推定入力には使わない。',
    '- 弱のみ、中のみ、強のみ、弱＋中、中＋強の5学習プール。各プールでRidge 2設定、SVR 2設定、ExtraTrees 2設定（30学習）とSVR/ETの座標平均を比較。',
    '- シンドローム169特徴量から弱/中/強を推定する分類器4学習（LogisticRegression C0.1/1、ExtraTrees leaf16/64）。合計34学習。',
    '## 切り替え方式',
    '- strict_two: 弱だけ/強だけで学習した2専門家。弱側重みはP(弱)+0.5P(中)。',
    '- overlap_two: 弱＋中/中＋強で学習した2専門家。同じ重み。中程度を学習範囲で重ねる。',
    '- three: 弱/中/強の3専門家。3クラス確率で座標を平均。',
    '- softは推定確率で混ぜる。hardは最尤クラスに変換（2専門家で中と判定した場合は50:50）。',
    '- さらに旧モデルと0.5:0.5で混ぜる場合／専門家のみの場合を比較。3構成×4分類器×hard/soft×2混合率＝48候補＋旧モデル。',
    '- 主選択は通常検証平均最小。弱・強向けは通常検証が旧モデル+0.01 mm以内という制約で各対象の平均を最小化。',
    f'- 実用選択: 全体 `{s["primary"]}`、弱 `{s["weak"]}`、強 `{s["strong"]}`。',
    '## 真の強弱で切り替える診断（実用性能ではない）',
    '同じ専門家の予測を真の強度クラスで切り替える。2専門家では中を50:50とする。これは発生中心そのものを教える診断ではないが、真の強度を使うため実用版と混同しない。また最適推定器の厳密な上限ではない。',
    means([('parent','旧モデル'),('routed_only/three','3専門家・推定切替のみ'),('oracle/three','3専門家・真の強弱で切替'),('expert/weak','弱専門家単体'),('prior','固定位置')]),
    '弱専門家単体の強イベント誤差は対象外への適用診断であり、通常運用の評価ではない。',
    '## 切り替えだけの寄与（同じ専門家・同じ混合率の対応付き比較）']
diagnostics=[]
for r in paired:
    if r['group'] not in groups or not (r['model'].startswith('oracle_matched/') or (r['model'].startswith('oracle/') and r['reference'].startswith('routed_only/'))):continue
    diagnostics.append({'条件':r['group'],'真ラベル版':r['model'],'観測版':r['reference'],'差mm':round(r['difference_mm'],5),'95%区間':str(np.round(r['ci95'],5).tolist())})
lines += [table(pd.DataFrame(diagnostics)), '## 全テスト結果（事前に固定した副評価を含む）',table(m[m.group.isin(groups)].round(5)),
    '## 検証スコア',table(pd.read_csv(root/'validation_scores.csv').rename(columns={'Unnamed: 0':'model'}).round(5)),
    '## 注意点・監査',
    '- 弱・強の切り替えに真の強度を使う結果はoracleの表だけ。実用版は1ショットの二値detector events [24,2047]からのみ推定。',
    '- d5、49量子ビット、24チェック、2.048 ms、背景ノイズ・固定ハードウェア・座標領域は前回と同じ。独立の旧平常較正128ショットも固定。',
    '- 中イベントは単純な二択ではない。境界を重ねる2専門家と3専門家を用意した。',
    '- 専門家の学習件数は分割により少なくなる。単なる同サイズモデル比較ではなく、同じ総学習イベント予算での構成比較。',
    '- 旧検証は再利用するが旧テストは選択に使わない。今回新規テストの結果で選び直さない。',
    '- 95%区間は10,000回のイベント単位paired bootstrap。主評価以外は記述的副評価、多重比較・学習集合/seed不確実性は未補正。',
    '- 主選択の通常p90は旧4.2265→4.2702 mm。今回の平均改善は大外れの改善を意味しない。単純な弱のみ/強のみの2専門家は通常平均2.6221 mmで旧2.5798 mmより悪かった。',
    '- REIは旧設定の履歴長1024・適用版。原著条件への優越性の主張ではない。',
    f'- 全歴史データとの生成seed非重複、36層均等、専門家の学習UID、同一ハードウェア、モデル/テストhashを監査。実装テスト{totals["tests"]}件通過。',
    '## 再実行',
    'リポジトリ直下で `.venv/bin/python -m qp_ode_simulator.specialist_study {prepare,train,freeze,generate,evaluate} <新しい出力先>` を順に実行。trainは`--workers 3`、generateは`--workers 8`。BLAS/OMPスレッドは1推奨。既存出力を上書きしない。',
    '- `protocol.json` / `selection.json`: 探索計画、テスト前固定。',
    '- `metrics.csv` / `paired.json` / `predictions.csv`: 全結果、差の区間、各予測。',
    '- `gate_test.json`: 4分類器の混同行列（行:真、列:予測、弱/中/強の順）とlog loss。',
    '- 各専門家ディレクトリの`.joblib`と`frozen.json`: 学習済み重み、選択モデル・学習イベントID。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
print(root/'RESULT_JA.md'); print('AUDIT',audit)
