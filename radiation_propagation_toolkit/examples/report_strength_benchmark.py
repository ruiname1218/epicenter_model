"""Audit fresh data, summarize severity, tail errors and both kinds of seed variation."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file
from qp_ode_simulator.strength_benchmark import SEEDS,TEST_SEEDS

root=Path(sys.argv[1]).resolve();read=lambda p:json.loads(p.read_text())
p=read(root/'protocol.json');s=read(root/'selection.json')
fresh=pd.DataFrame([read(path) for path in sorted((root/'events').glob('*.json'))])
assert len(fresh)==1080 and fresh.event_uid.nunique()==fresh.generation_seed.nunique()==1080
old=[]
for folder in ['distance_study_20260909','distance_growth_20260909','syndrome_features_20260909','specialists_20260909','weak_information_20260909']:
    old.extend(read(path) for path in (root.parent/folder/'events').glob('*.json'))
old=pd.DataFrame(old);assert not set(fresh.generation_seed)&set(old.generation_seed)
for seed in TEST_SEEDS:
    a=fresh[fresh.test_seed==seed];assert len(a)==360
    counts=a.groupby(['geometry','propagation_law','epicenter_region','strength_band']).size();assert len(counts)==36 and (counts==10).all()
for key in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert fresh[key].nunique()==1 and fresh[key].iloc[0]==old[key].iloc[0]
rows=pd.read_csv(Path(p['source'])/'rows.csv');train=rows[rows.role=='train']
assert train.event_uid.nunique()==2880 and not set(train.event_uid)&set(old[old.role!='train'].event_uid)
assert not ((train.geometry=='elliptical')&(train.strength_band==2)).any()
for name,digest in p['cache_hashes'].items():assert _hash_file(Path(p['source'])/name)==digest
for name,digest in s['hashes'].items():assert _hash_file(root/name)==digest
manifest=read(root/'manifest.json');assert manifest['selection_hash']==_hash_file(root/'selection.json')
for name,digest in manifest['hashes'].items():assert _hash_file(root/'events'/name)==digest
tests={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in tests:tests[key]+=int(suite.attrib.get(key,0))
assert tests['tests']>0 and tests['failures']==tests['errors']==0
m=pd.read_csv(root/'metrics.csv');pairs=read(root/'paired.json');pred=pd.read_csv(root/'predictions.csv')
assert pred.groupby(['model','training_seed']).event_uid.nunique().eq(1080).all()
assert pred.groupby(['model','training_seed']).size().eq(2160).all()
agg=m[(m.test_seed==0)&(m.training_seed<=0)].copy()
labels={'weak':'弱','medium':'中','strong':'強（円＋楕円）','strong_circle':'強い円','strong_ellipse':'強い楕円'}

def table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def nice(frame):
    cols=['group','model','events','mean_mm','within_1mm','p90_mm','p95_mm','over_3mm']
    a=frame[cols].copy();a['group']=a.group.map(labels).fillna(a.group)
    a['within_1mm']=(100*a.within_1mm).round(1);a['over_3mm']=(100*a.over_3mm).round(1)
    return a.round(4).rename(columns={'group':'強度','model':'モデル','events':'イベント','mean_mm':'平均mm','within_1mm':'1mm以内%','p90_mm':'p90 mm','p95_mm':'p95 mm','over_3mm':'3mm超%'} )

variation=[]
for (name,group),a in m[(m.test_seed==0)&(m.training_seed.isin(SEEDS))].groupby(['model','group']):
    variation.append(dict(model=name,group=group,seeds=len(a),mean_mm=a.mean_mm.mean(),seed_sd_mm=a.mean_mm.std(ddof=1),min_mm=a.mean_mm.min(),max_mm=a.mean_mm.max(),
        within1_min_percent=100*a.within_1mm.min(),within1_max_percent=100*a.within_1mm.max()))
v=pd.DataFrame(variation);v.to_csv(root/'training_seed_variation.csv',index=False)
block=m[(m.test_seed!=0)&(m.training_seed<=0)]
block.to_csv(root/'test_seed_metrics.csv',index=False)
# Show signs of effect in every training-seed x fresh-data-seed cell; never select a winning cell.
grid=[]
for group in ['weak','medium','strong','strong_circle','strong_ellipse']:
    for seed in SEEDS:
        for test in TEST_SEEDS:
            for name in ['Blend','Overlap','ThreeSoft','ThreeHard','CNN']:
                a=m[(m.group==group)&(m.model==name)&(m.training_seed==seed)&(m.test_seed==test)].iloc[0]
                b=m[(m.group==group)&(m.model=='REI')&(m.test_seed==test)].iloc[0]
                grid.append(dict(group=group,model=name,training_seed=seed,test_seed=test,mean_mm=a.mean_mm,rei_mm=b.mean_mm,difference_mm=a.mean_mm-b.mean_mm))
pd.DataFrame(grid).to_csv(root/'seed_grid.csv',index=False)
hist=pd.read_csv(root/'historical_strength_metrics.csv');catalog=pd.read_csv(root/'historical_catalog.csv')
tail=[]
for r in read(root/'tail_intervals.json')['records']:
    factor=100 if r['metric'] in ['within_1mm','over_3mm'] else 1
    tail.append({'強度':labels[r['group']],'指標':r['metric'],
        '単位':'percentage points' if factor==100 else 'mm',
        'Overlap−REI':round(factor*r['difference'],5),
        '記述的95%区間':str([round(factor*x,5) for x in r['ci95']])})
lines=['# 強・中・弱の再集計と新規複数seed比較',
    '## 結論',
    '事前固定したOverlap方式は、適用版REIに対して中の平均誤差を約7.6%、強を約46.4%低減した。中・強とも学習seed3×テスト生成root3の9セル全てで平均が改善した。ただしセルは学習集合やテストを共有し、9回の独立実験ではない。',
    '強は1mm以内率・p90・p95・3mm超率も改善した。中は1mm以内率が5.0→7.8%、3mm超率が51.4→41.9%に改善したが、p90/p95の改善は確認できず、まだ高精度とは言いにくい。弱の平均・1mm以内率に優位性はなく、p90/p95は悪化した。強・中の位置推定を主成果とし、弱と尾部誤差を限界として報告するのが妥当。',
    '## 新規テストの主結果',
    '学習2,880イベント、d5、24チェック×2,047内部round、通常ノイズ、約2.048ms。弱/中/強各360の新規1,080物理イベント（各2ショット）。学習seed41/42/43、独立テスト生成root3個。全モデルに同じテストを入力した。',
    '確率的モデルの主表は3seedの座標平均アンサンブル。Ridge/SVR/適用REI/固定位置は決定的な1モデルであり、同じものを3seed扱いにはしていない。',
    table(nice(agg[agg.group.isin(['weak','medium','strong'])&agg.model.isin(['REI','Blend','Overlap','ThreeSoft','CNN','Prior'])]))]
for group in ['medium','strong']:
    r=next(a for a in pairs if a['group']==group and a['model']=='Overlap' and a['reference']=='REI')
    lo,hi=r['ci975'];lines.append(f'- 事前固定の主比較・{labels[group]}: Overlap−適用REI = {r["difference_mm"]:+.4f} mm、97.5%イベントbootstrap区間 [{lo:+.4f},{hi:+.4f}]。中・強の2比較にBonferroni対応した区間。')
lines+=['## 全モデル・形状別',table(nice(agg[agg.group.isin(labels)])),
    '## 学習seedでの再現性',
    '以下は各seed単体の平均誤差のばらつき。上の座標平均アンサンブルと区別する。学習データ集合は同じで、木・分類器の乱数またはCNN初期値が異なる。決定的モデルにはseed分散を付けない。',
    table(v[v.group.isin(['weak','medium','strong'])].round(5)),
    '## 独立した3テスト生成rootでの再現性',
    table(block[block.group.isin(['weak','medium','strong'])&block.model.isin(['REI','Blend','Overlap','ThreeSoft','CNN'])][['group','model','test_seed','events','mean_mm','p90_mm','within_1mm']].round(5)),
    '学習seed3×テスト生成root3の全セルをseed_grid.csvに保存。良かったseedだけを選ばない。学習集合そのものの再抽出やハードウェア変動への再現性は未検証。',
    '## 対応付き差（副評価は記述的95%区間）',
    table(pd.DataFrame([{**{k:v for k,v in r.items() if not k.startswith('ci')},'95%区間':str(np.round(r['ci95'],5).tolist())} for r in pairs if r['group'] in ['weak','medium','strong']]).round(5)),
    '## 1mm以内率・大外れの差の不確実性',
    '同一物理イベントを対応させた10,000回bootstrap。率はパーセントポイント差。1mm以内率は正、3mm超率・p90・p95は負が改善。副評価の記述的95%区間であり、多重比較補正はしていない。',
    table(pd.DataFrame(tail)),
    '## 既存結果の再集計',
    f'座標誤差のある既存predictions.csv {int((catalog.status.str.startswith("Aggregated")).sum())}ファイルを、強度・形状・モデル・学習数・seed・元のsplitを保持して再集計した。{len(hist)}行をhistorical_strength_metrics.csvに保存。位置の左右分類は座標誤差ではないため除外。',
    '異なるファイルで同じイベントを再採点している場合もある。実験間の行を合算して独立サンプル数を増やさない。異なるテスト・観測条件の数値を単純に順位付けしない。弱専用実験は弱のみで、中・強の欠落を0件の成功などとは扱わない。oracle/特権情報を含む候補も元の名称を残し、実用方式とは分けて読む。',
    '参考: 直前の専門家実験を強度で分けた結果（今回とは別テスト）',
    table(hist[(hist.source=='specialists_20260909/predictions.csv')&(hist['shape']=='both')&hist.model.isin(['REI','parent','overlap_two/ET16/hard/0.5','three/logistic1/soft/1.0','prior'])][['model','strength','events','mean_mm','within_1mm','p90_mm']].round(5)),
    '## 比較条件・解釈',
    '- 弱/中/強は生成率proxyの[1e-10,1e-9] / [1e-9,1e-8] / [1e-8,3e-7] /us。位置誤差で強度を定義し直していない。',
    '- 強は円・楕円50:50の人工的な均等テスト。強い楕円は学習に不使用なので、形状×強度の未学習組合せに対する結果として別表も提示。実際の発生頻度での平均とは異なる。',
    '- Blend: 全強度SVR+ExtraTrees。Overlap: 弱＋中Ridge／中＋強SVRをET推定クラスで切り替え、Blendと50:50。ThreeSoft: 弱ET／中Ridge／強SVR+ETをロジスティック分類確率で混合。ThreeHard: 推定クラスで3専門家を切り替え、Blendと50:50。真の強度は推定に使わない。',
    '- 6個の決定的回帰器、1分類器、3seed×4個の確率的学習＝19学習。CNNは同じ学習データで以前学習済みの3seedチェックポイントを凍結したまま再評価。',
    '- 全モデルの設定は過去の検証で固定。今回のテストで選び直さない。強・中に研究の焦点を移したこと自体は過去結果を見た後の判断であり、今回の新規テストで確認する。',
    '- REIは履歴長1024、r=1、既存の空間閾値を固定した適用版。全モデルともイベントがある観測窓から1ショットで位置のみを推定。オンライン検出・誤検出率・原著REI実装/条件の優越性は検証していない。',
    '- REIが回答しない場合の固定中心fallbackと回答率を保存。主表の誤差はfallbackを含む。',
    '- p90は誤差の90パーセンタイル、p95は95パーセンタイル。3mm超率は大外れの補助指標であり、平均誤差だけで全ての指標が改善したとはしない。',
    '- 10,000回の物理イベント単位paired bootstrap。2ショットを独立イベントと数えない。主比較2個以外は記述的95%区間。学習seedの不確実性は区間に混ぜず別表で報告。',
    f'- {tests["tests"]}実装テスト通過。旧テスト非混入、全旧生成seed非重複、3root×36層均等、同一ハードウェア、モデル/データhashを監査。',
    '## 成果物',
    'historical_strength_metrics.csv / historical_catalog.csv: 既存集計と対象一覧。metrics.csv / paired.json / predictions.csv: 新規比較。training_seed_variation.csv / test_seed_metrics.csv / seed_grid.csv: 再現性。protocol.json / selection.json: 事前固定。audit.json / tests.xml: 監査。',
    '再実行: `.venv/bin/python -m qp_ode_simulator.strength_benchmark <stage> <新規出力先>`、stageはprepare→train→freeze→generate→evaluate。OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
audit=dict(status='passed',test_events=1080,train_events=2880,training_seeds=list(SEEDS),test_roots=list(TEST_SEEDS),historical_seed_overlap=0,
    hardware_equal=True,old_holdouts_not_trained=True,all_hashes='passed',tests=tests,rei_answer_rate=read(root/'evaluation.json')['rei_answer_rate'],report_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit)
print(root/'RESULT_JA.md');print(audit)
