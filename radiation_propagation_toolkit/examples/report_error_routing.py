"""Report all prespecified candidates; never replace selection with test winner."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.error_routing_study import verify
from qp_ode_simulator.localization import _hash_file

root=Path(sys.argv[1]).resolve();read=lambda p:json.loads(p.read_text())
p,s=verify(root);m=pd.read_csv(root/'metrics.csv');pred=pd.read_csv(root/'predictions.csv');pairs=read(root/'paired.json')
folds=pd.read_csv(root/'oof_folds.csv');assert folds.groupby('event_uid').fold.nunique().eq(1).all()
assert folds.event_uid.nunique()==2880 and set(folds.fold)=={0,1,2}
source=pd.read_csv(Path(p['source'])/'rows.csv')
assert set(folds.event_uid)==set(source[source.role=='train'].event_uid)
assert not set(pred.event_uid)&set(source.event_uid)
tests={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in tests:tests[key]+=int(suite.attrib.get(key,0))
assert tests['tests']>0 and tests['failures']==tests['errors']==0
audit=read(root/'audit.json');assert audit['status']=='passed'
fresh=pd.DataFrame([read(path) for path in (root/'events').glob('*.json')])
for key in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert fresh[key].nunique()==1 and fresh[key].iloc[0]==source[key].iloc[0]
timing=read(root/'inference_timing.json')
chosen=s['selected'];alternate=s['unconstrained'];names=list(dict.fromkeys(['REI','ThreeHard','Overlap','SVR',chosen,alternate,'MiddleGlobalSVR','MiddleOnlySVR']))

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def nice(df):
    a=df[['group','model','events','mean_mm','within_1mm','p90_mm','p95_mm','over_3mm']].copy()
    for c in ['within_1mm','over_3mm']:a[c]*=100
    return a.round(4).rename(columns={'group':'強度・形状','model':'モデル','events':'イベント数','mean_mm':'平均mm','within_1mm':'1mm以内%','p90_mm':'p90 mm','p95_mm':'p95 mm','over_3mm':'3mm超%'})

# Descriptive paired whole-event bootstrap for rates and tails of primary.
tails=[]
for group,band in [('weak',0),('medium',1),('strong',2)]:
    a=pred[(pred.model==chosen)&(pred.strength_band==band)].sort_values(['event_uid','shot'])
    for ref in ['ThreeHard','REI']:
        b=pred[(pred.model==ref)&(pred.strength_band==band)].sort_values(['event_uid','shot'])
        assert a[['event_uid','shot']].reset_index(drop=True).equals(b[['event_uid','shot']].reset_index(drop=True))
        assert a.groupby('event_uid').size().eq(2).all()
        ae=a.error_mm.to_numpy().reshape(-1,2);be=b.error_mm.to_numpy().reshape(-1,2)
        rng=np.random.default_rng(2026091071)
        indices=rng.integers(0,len(ae),size=(10000,len(ae)))
        for name,fn,factor in [('within1',lambda z,axis:np.mean(z<=1,axis=axis),100),
                               ('over3',lambda z,axis:np.mean(z>3,axis=axis),100),
                               ('p90',lambda z,axis:np.quantile(z,.9,axis=axis),1),
                               ('p95',lambda z,axis:np.quantile(z,.95,axis=axis),1)]:
            # Chunks bound memory for quantiles of clustered two-shot resamples.
            bootstrap=[]
            for start in range(0,len(indices),250):
                ids=indices[start:start+250]
                bootstrap.extend(factor*(fn(ae[ids].reshape(len(ids),-1),1)-fn(be[ids].reshape(len(ids),-1),1)))
            tails.append(dict(group=group,model=chosen,reference=ref,metric=name,
                difference=factor*float(fn(ae,None)-fn(be,None)),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                unit='percentage points' if factor==100 else 'mm'))
_write_json(root/'tail_intervals.json',tails)

# Actual qubit convex-hull membership, not the simulator's region label.
hull=ConvexHull(np.asarray(p['geometry']['distances']['5']['physical_mm']))
coords=pred[['epicenter_row','epicenter_col']].to_numpy()
pred['hull_region']=np.where((coords@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9).all(1),'inside_hull','outside_hull')
from qp_ode_simulator.strength_benchmark import summary
regions=[]
for (model,band,region),a in pred[pred.model.isin(names)].groupby(['model','strength_band','hull_region']):
    regions.append(dict(model=model,strength_band=int(band),region=region,events=a.event_uid.nunique(),**summary(a.error_mm)))
pd.DataFrame(regions).to_csv(root/'convex_hull_metrics.csv',index=False)

lines=['# 誤差を直接減らす混合と中担当SVRの改善実験（2026-09-10）',
    '## 結論',
    '中担当を中専用SVRへ変更すると、新規テストの中平均が2.7857→2.7674 mm（約0.66%減）になり、主比較97.5%区間は改善側。3つのテスト生成rootすべてで平均が改善した。強は0.8434→0.8365 mmだが、主比較区間が0を含み明確な改善とは言えない。中p90は4.1448→4.1932 mm、強p90は1.6357→1.6730 mmで、尾部改善は確認できなかった。',
    '副候補の誤差直接学習は中2.7495、強0.8285 mm。平均では有望だが、中p90と弱平均の悪化があり、検証で主候補にしなかった。全強度SVRを中担当に入れる変更は中2.8117 mmで悪化。難しい層の重み付けSVRも中の数値は改善するが、強・弱や尾部とのトレードオフが残る。大幅な精度向上ではなく、限定的な改善と失敗条件の整理という結果。',
    'モデルと予測は新規ディレクトリに保存した。既存の既定モデルは自動置換していない。',
    '## 事前選択',
    f'主候補: `{chosen}`。大外れ制約を外した副候補: `{alternate}`。選択後に新規1,080イベントを生成。テスト最良への差し替えはしていない。',
    '同じ学習2,880イベント・検証360イベント、d5、24チェック×2,047内部round、約2.048ms、通常ノイズ。テストは弱/中/強各360物理イベント、各2ショット。テスト生成rootは3個。',
    '## 共通未使用テスト結果',table(nice(m[m.model.isin(names)&m.group.isin(['weak','medium','strong'])])),
    '## 主比較と解釈']
for group in ['medium','strong']:
    r=next(r for r in pairs if r['group']==group and r['model']==chosen and r['reference']=='ThreeHard')
    lines.append(f'{group}: 主候補−従来ThreeHardの平均差 {r["difference_mm"]:+.5f} mm、97.5%対応イベントbootstrap区間 {np.round(r["ci975"],5).tolist()}。負が改善。中・強の2主比較にBonferroni対応した近似区間。')
lines+=['検証でp90悪化を制限しても、新規テストでのp90改善を保証しない。平均・1mm以内率・p90/p95・3mm超率を別々に判断する。',
    '## 何を実装したか',
    '- ThreeHardの中担当を、従来の中のみRidge／全強度SVR／中のみSVRの3通りで比較。強度は観測特徴から分類し、真の強度は渡さない。',
    '- ErrorGate: Ridge・SVR・ExtraTrees・ThreeHardの4座標を、シンドローム169特徴＋その予測座標だけからsoftmaxで凸混合。分類精度ではなく位置のEuclidean誤差を直接学習。L2正則化3通り×上位10%平均誤差への重み2通り、各3初期値。学習目的の強度重みは弱.2/中.4/強.4。',
    '- 3-foldの物理イベント単位out-of-fold予測で混合器を学習。同イベントの2ショットは必ず同fold。各fold内でスケーラーと回帰器・分類器を再学習し、予測対象イベントは基礎モデルの学習から除外。基礎ET系は3seedの座標平均。',
    '- OOFの強度×領域×形状別誤差で難しい学習層に重みを付けたSVRを2種類比較。弱は重み1。追加データ生成ではなく、既存学習データの重み付けという低コストの予備実験。',
    '- 新規学習はOOF基礎モデル48学習、中専用/重み付きSVR3学習、混合器18学習の計69学習。検証選択の候補は既存方式込み15構成。最終基礎モデルは前回の凍結済みモデルを再利用。',
    '- 検証選択は中・強の平均を等重みで最小化。ただし各帯p90がThreeHard+.02mm以内、弱平均が+.03mm以内。従来モデルも候補に残し、改善不成立なら維持する。',
    '## 検証での失敗条件',
    'validation_failure_analysis.csv: 中央/端/外側、円/楕円、伝播則、発生時刻・到達範囲・軸比の3分位別。各分位境界は分析対象集合から作る記述的分析で、モデル入力やテスト再選択には使わない。training_difficult_strata.csvはOOF学習誤差。',
    table(pd.read_csv(root/'validation_failure_analysis.csv').query("model == 'ThreeHard' and strength_band == 1 and condition == 'epicenter_region'").round(4)),
    '## 全候補・形状別結果',table(nice(m[m.group.isin(['weak','medium','strong','strong_circle','strong_ellipse'])])),
    '## 尾部・成功率の差（記述的95%区間）',table(pd.DataFrame(tails).round(5)),
    '## 凸包内外の比較',
    'REIの重心推定は出力可能領域に制限があるため、実際の物理qubitの凸包で内外を分けた。シミュレータのinside/edge/outside区分とは別。外側だけの利益を普遍的優位性としない。',
    '重要: 今回の1,080イベントは全てd5の物理qubitの凸包内にある。生成ラベルのoutsideは基準領域[-3,3]mmの外側を意味し、量子ビット配置の外側ではない。物理qubitの座標範囲は各軸[-5,5]mm。今回、チップ外への外挿性能は検証できていない。',
    table(pd.DataFrame(regions).round(4)),
    '## 再現性',
    'test_seed_metrics.csv: 独立生成root3個の結果。seed_variation.csv: 従来系は木/分類器seed、ErrorGateは基礎アンサンブルを固定した混合器初期値の変動。両者を同じ種類の学習変動と扱わない。学習集合の再抽出や異なるデバイスへの再現性は未検証。',
    table(pd.read_csv(root/'seed_variation.csv').query("group in ['medium','strong']").round(5)),
    '## 1ショット推論時間',
    'CPU単一スレッド、モデル読み込み後、特徴抽出込み、16物理イベント×3反復の48回/モデル。中央値とp90を表示。観測データ取得と学習時間は含めない。現在のPython実装の診断であり、最適化後の限界やリアルタイム保証ではない。予測が保存済み結果と一致することも確認した。',
    table(pd.DataFrame([dict(model=k,**v) for k,v in timing['models'].items()]).round(4)),
    '精度改善には計算コストがある。約2.048msの観測窓に対して現行3seed専門家アンサンブルは約90ms/shotを要し、この実装のまま各窓を単一ワーカーで連続処理できるとは言えない。低遅延運用にはモデル圧縮・蒸留・推論の最適化などを別途検証する必要がある。',
    '## 研究上の位置付けと未実施項目',
    'REI原著は検出・中心・影響範囲・実行時間・復号への応用を扱う。今回はイベントがある観測窓からの位置推定だけを比較しており、検出性能・論理誤り率の改善を実証していない。[REI原著](https://arxiv.org/html/2506.16834v1)。',
    '2026-09-10に過去の監査で記録した実装候補 https://github.com/HicrestLaboratory/REI は404で取得できなかった。arXiv v1のリポジトリ引用は公開予定のプレースホルダーだった。原著実装との完全同一性は未確認。現行適用版REIを比較対象と明記する。',
    '両者の観測元は同じ内部detector bitsだが、学習方式は全2,047roundの特徴と独立平常較正・2,880正解イベントを利用し、REIは末尾1,024roundの固定履歴を使用。利用する履歴処理、較正情報、学習コストは等しくない。',
    '強い楕円は学習・検証に含まない。円から楕円への形状×強度の未学習組合せを評価するが、完全未知の伝播則や別ハードウェア・別シミュレータへの汎化ではない。強度区分はシミュレータの生成率proxyであり実機での普遍的分類ではない。',
    '大規模な失敗層追加生成、信頼区間付き位置出力/回答棄却、異なる物理モデルでの評価、影響領域推定、復号器への接続は今回は未実施。モデルの小幅な改善だけで新規性が十分とは断定しない。',
    '## 監査と成果物',
    f'実装テスト{tests["tests"]}件通過。OOFイベント分離、学習/検証/新規テスト非混入、旧生成seed非重複、3root×36層均等、重み・コード・データhashを確認。REI回答率 {audit["rei_answer_rate"]:.4f}。',
    'timingはaudit.jsonに保存したバッチ壁時計の診断値。基礎モデル一括計測には比較用の余分なモデルと読み込み時間を含むため、REIとの速度倍率やオンライン遅延の主張には使わない。',
    'protocol.json / selection.json / validation_metrics.csv / validation_failure_analysis.csv / oof_folds.csv / metrics.csv / paired.json / tail_intervals.json / test_failure_analysis.csv / convex_hull_metrics.csv / test_seed_metrics.csv / seed_variation.csv / audit.json。',
    '実行: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.error_routing_study <prepare|train|generate|evaluate> <出力先>。レポート: examples/report_error_routing.py <出力先>。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
audit.update(tests=tests,oof_event_separation='passed',hardware_equal=True,all_events_inside_physical_hull=bool((pred.hull_region=='inside_hull').all()),report_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit)
print(root/'RESULT_JA.md')
