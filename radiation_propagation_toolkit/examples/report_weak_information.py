"""Audit and distinguish operational weak-event gains from privileged diagnostics."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file
from qp_ode_simulator.weak_information_study import NTRAIN,NVAL,NTEST,role

root=Path(sys.argv[1]).resolve(); read=lambda p:json.loads(p.read_text())
p=read(root/'protocol.json');s=read(root/'selection.json')
records=[]
for i in range(NTRAIN+NVAL+NTEST):
    path=root/'events'/f'{i:04d}.json';row=read(path);assert row['event']==i and row['role']==role(i);records.append(row)
labels=pd.DataFrame(records)
assert labels.event_uid.nunique()==len(labels) and labels.generation_seed.nunique()==len(labels)
assert not labels.is_control.any() and (labels.strength_band==0).all()
assert (labels.qp_generation_scale_per_us.between(1e-10,1e-9)).all()
for split,count in [('train',144),('validation',24),('test',48)]:
    cells=labels[labels.role==split].groupby(['geometry','propagation_law','epicenter_region']).size()
    assert len(cells)==12 and (cells==count).all()
older=[]
for folder in ['distance_study_20260909','distance_growth_20260909','syndrome_features_20260909','specialists_20260909']:
    older.extend(read(path) for path in (root.parent/folder/'events').glob('*.json'))
old=pd.DataFrame(older);assert not set(labels.generation_seed)&set(old.generation_seed)
for field in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']:
    assert labels[field].nunique()==1 and labels[field].iloc[0]==old[field].iloc[0]
for split in ['fit','test']:
    manifest=read(root/f'{split}_manifest.json')
    for name,digest in manifest['hashes'].items():assert _hash_file(root/'events'/name)==digest
for name,digest in s['cache_hashes'].items():assert _hash_file(root/name)==digest
for name,digest in s['hashes'].items():assert _hash_file(root/name)==digest
assert read(root/'test_manifest.json')['selection_hash']==_hash_file(root/'selection.json')
for i in range(NTRAIN+NVAL+NTEST):
    with np.load(root/'events'/f'{i:04d}.npz') as z:
        for key in ['nominal','quiet10']:
            assert z[key].shape==(2,24,4095) and np.isin(z[key],[0,1]).all()
        assert z['latent_t1'].shape==(196,)
        for prefix in ['nom2','nom4','quiet2','quiet4']:
            assert z['latent_'+prefix].shape==(24,16) and np.isfinite(z['latent_'+prefix]).all()
            if i<NTRAIN:np.testing.assert_array_equal(np.load(root/f'bank_{prefix}.npy',mmap_mode='r')[i],z['latent_'+prefix])
totals={'tests':0,'failures':0,'errors':0}
for suite in ET.parse(root/'tests.xml').getroot().iter('testsuite'):
    for key in totals:totals[key]+=int(suite.attrib.get(key,0))
assert totals['tests']>0 and totals['failures']==totals['errors']==0
audit=dict(status='passed',train=NTRAIN,validation=NVAL,test=NTEST,old_generation_seed_overlap=0,baseline_hardware_equal=True,
    paired_conditions=True,binary_shapes=[2,24,4095],template_bank_train_only=True,balanced_12_strata=True,
    hashes='passed',tests=totals,selection_hash=_hash_file(root/'selection.json'),report_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json',audit)
m=pd.read_csv(root/'metrics.csv');paired=read(root/'paired.json');pred=pd.read_csv(root/'predictions.csv')


def table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


names={'parent':'旧全強度SVR+ET','previous_weak_routed':'前回の弱重視切り替え',s['fixed']:'固定座標基準'}
conditions=[('通常ノイズ2ms',s['chosen']['nom2']),('通常ノイズ4ms',s['chosen']['nom4']),
            ('背景1/10・2ms',s['chosen']['quiet2']),('背景1/10・4ms',s['chosen']['quiet4'])]
summary=[]
for label,name in list((v,k) for k,v in names.items())+conditions:
    r=m[m.model==name].iloc[0]
    summary.append({'条件':label,'選択モデル':name,'平均mm':round(r.mean_mm,5),'p90 mm':round(r.p90_mm,5),'1mm以内率':round(r.within_1mm,5)})
lines=['# d5弱イベント: 観測・時刻・物理情報の切り分け',
    '## 新規テスト576イベントの結果',
    '全て弱イベント。イベント存在・弱強度帯が既知の評価であり、全強度で運用した性能ではない。学習1,728・検証288・新規テスト576物理イベント、各2ショット。',table(pd.DataFrame(summary))]
for label,name in conditions:
    r=next(v for v in paired if v['model']==name and v['reference']=='parent'); lo,hi=r['ci95']
    judgment='改善側' if hi<0 else '悪化側' if lo>0 else '明確な差なし'
    fixed=next(v for v in paired if v['model']==name and v['reference']==s['fixed']); flo,fhi=fixed['ci95']
    lines.append(f'- {label}: 旧モデルとの差 {r["difference_mm"]:+.5f} mm、95%区間 [{lo:+.5f},{hi:+.5f}]（{judgment}）。固定座標との差 {fixed["difference_mm"]:+.5f} mm、区間 [{flo:+.5f},{fhi:+.5f}]。')
lines+=['## 特権情報による診断（シンドロームだけの実用性能ではない）']
diag=[]
for case,name in s['diagnostics'].items():
    r=m[m.model==name].iloc[0];diag.append({'診断':case,'モデル':name,'平均mm':round(r.mean_mm,5),'p90 mm':round(r.p90_mm,5)})
lines+=[table(pd.DataFrame(diag)),
    '- nom2_oracle_time: 真の発生時刻を与えて4区間に分割。学習・推定時とも真の時刻を使う診断。',
    '- oracle_t1: ノイズのないd5内49量子ビットのT1過剰緩和率を2.048ms内で4分割平均（196特徴量）。座標・真の時刻を入力に足さない。',
    '- oracle_marginals: 2.048msの24チェックの正確な条件付きdetector発生確率を16分割平均（384特徴量）。有限ショットのゆらぎがないため、実観測では得られない。',
    '- 物理情報で当たってもシンドロームから当たる証拠ではない。逆にモデルが外れても数学的な推定不可能性の証明ではない。',
    '## 試した改善方法',
    '- 通常2ms: 固定切り出し、観測シンドロームの集計から推定した変化点、5つの時間分割候補を結合した特徴量。',
    '- 同じ4.096msシンドロームから2.048msのprefixを取り出して比較。時間窓の比較で別イベントにしない。',
    '- 背景1/10: T1/T2の平常緩和率と回路背景ノイズを1/10にする仮想ハードウェア介入。放射線による過剰緩和率は保存。ソフトウェアのみの改善ではない。',
    '- 平常較正は条件ごとに256独立ショット。旧モデルの評価には旧較正をそのまま使う。',
    '- 9特徴量ケースそれぞれRidge2、SVR2、ExtraTrees2設定（54回の回帰学習）。各ケースの設定は検証で選択。',
    '- 学習イベントの物理的なdetector発生確率をテンプレートにし、観測ビットの時間窓発生率と照合。Bernoulli複合スコア／平常空間共分散で補正するGLSスコア、温度0.25/1/4（4観測条件×6設定）。座標はテンプレート重み付き平均。',
    '- テンプレートに真の中心座標を付けるのは学習イベントだけ。問い合わせの真の確率・中心・時刻はテンプレート推定には使わない。チェックの相関を完全には扱わず、スコアを正確な同時尤度とは呼ばない。',
    '- 固定座標は学習座標平均／既知の幾何中心(0,0)を検証で選択。これを超えるか確認する。',
    '## 全候補のテスト値（テスト順位で選び直さない）',table(m.round(5)),
    '## 対応付き比較',table(pd.DataFrame([{**{k:v for k,v in r.items() if k!= 'ci95'},'ci95':str(np.round(r['ci95'],6).tolist())} for r in paired]).round(6)),
    '## 検証時の選択',table(pd.read_csv(root/'validation_scores.csv').round(5))]
if (root/'signal_diagnostic.json').exists():
    info=read(root/'signal_diagnostic.json')['results']
    count_rows=[{'条件':key,'平常総数期待値':round(value['baseline_expected_total'],2),'平常総数の標準偏差':round(value['calibration_total_sd'],2),
                 '放射線による増加期待値の中央値':round(value['extra_expected_count_quantiles'][1],2),
                 '増加/平常標準偏差の中央値':round(value['total_count_snr_quantiles'][1],3)} for key,value in info.items()]
    lines+=['## なぜ弱信号が難しいか（総検出数の診断）',table(pd.DataFrame(count_rows)),
        '通常2msでは放射線による検出数増加の期待値は中央値約5件なのに対し、平常総数のショット間標準偏差は約58件。背景低減・観測延長でこの比は改善する。ただし全チェック合計の尺度であり、空間・時間パターンを全て評価した情報量上限ではない。診断用の潜在確率を使用し、モデル入力には渡さない。']
# Predefined lower/upper half of the log-uniform weak generation band, not test-optimized boundaries.
subgroups=[]
for name in set(['parent',s['fixed']]+list(s['chosen'].values())):
    f=pred[pred.model==name]
    for label,mask in [('lower_weak',f.qp_generation_scale_per_us<np.sqrt(1e-19)),('upper_weak',f.qp_generation_scale_per_us>=np.sqrt(1e-19))]:
        subgroups.append({'model':name,'group':label,'events':f[mask].event_uid.nunique(),'mean_mm':float(f[mask].error_mm.mean())})
pd.DataFrame(subgroups).to_csv(root/'strength_subgroups.csv',index=False)
lines+=['## 弱強度帯の前半・後半（副評価）',
    '境界は対数強度帯の中央sqrt(1e-10×1e-9) /us。テストに合わせて境界を探索しない。',table(pd.DataFrame(subgroups).round(5)),
    '## 注意事項',
    '- 2ms通常条件の検証選択を主比較とし、他は副評価。今回のテストを使った再調整はしない。',
    '- 弱専用学習は1,728イベント。旧全強度モデルは2,880（うち弱1,152）なので、旧モデルとの差は純粋な特徴量だけの効果ではない。同じ新学習のnom2_fixedへの対応付き差も保存。',
    '- 真の時刻と潜在物理特徴は診断専用。潜在値はlatent_という別名で保存し、観測推定関数はビットからの特徴量のみを受け取る。',
    '- 95%区間は物理イベント単位の10,000回paired bootstrap。2ショットを独立イベントとは数えない。多重比較、学習データ集合・学習seedの不確実性は未補正。',
    '- d5固定、量子ビット間隔・source領域・公称ハードウェアは前回と同一。BB code・複数イベント・連続検出・未知強度の統合運用には未検証。',
    '- 背景1/10・4msの平均改善は弱帯の上半分に偏る。下半分の平均3.296 mmは固定位置3.274 mmを超えていない。全体p90も前回弱モデル4.455→4.870 mmで悪化方向。平均だけで高精度化・大外れ解消とは主張しない。',
    f'- 実装テスト{totals["tests"]}件通過。全2,592イベントのhash・ビット形状・12層均等・過去seed非重複・物理ハードウェア同一・学習限定テンプレートを監査。',
    '## 成果物',
    '`protocol.json`, `selection.json`, `validation_scores.csv`, `metrics.csv`, `paired.json`, `predictions.csv`, `strength_subgroups.csv`, `audit.json`, `tests.xml`。学習済み回帰器は各ケースのディレクトリ、テンプレートはbank_*.npy。',
    '再実行: リポジトリで `.venv/bin/python -m qp_ode_simulator.weak_information_study <stage> <新出力先>`。stageはprepare→generate_fit→train→freeze→generate_test→evaluate。生成は--workers 8、学習は--workers 3を推奨。OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')
print(root/'RESULT_JA.md');print('AUDIT',audit)
