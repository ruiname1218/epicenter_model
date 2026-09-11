"""Independent data audit and report for the preselected feature ablations."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file


root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text())
p = read(root/'protocol.json'); s = read(root/'selection.json')
parent = Path(p['parent']); older = Path(read(parent/'protocol.json')['parent'])
cache = read(root/'cache.json')
for name, digest in cache['hashes'].items():
    assert _hash_file(root/name) == digest, name
assert _hash_file(parent/'protocol.json') == p['parent_protocol_hash']
assert _hash_file(parent/'progress.json') == p['parent_data_hash']
history = pd.DataFrame([read(path) for folder in (parent, older) for path in sorted((folder/'events').glob('*.json'))])
fresh = pd.DataFrame([read(path) for path in sorted((root/'events').glob('*.json'))])
assert len(fresh) == 720 and fresh.event_uid.nunique() == 720 and fresh.generation_seed.nunique() == 720
assert not set(fresh.generation_seed) & set(history.generation_seed)
assert not fresh.is_control.any()
strata = fresh.groupby(['geometry', 'propagation_law', 'epicenter_region', 'strength_band']).size()
assert len(strata) == 36 and (strata == 20).all()
for field in ('qp_baseline_t1_by_qubit_us', 'qp_baseline_t2_by_qubit_us', 'qp_qubit_frequency_by_qubit_ghz'):
    assert fresh[field].nunique() == 1 and fresh[field].iloc[0] == history[field].iloc[0]
used = pd.read_csv(root/'rows.csv')
train = used[used.role == 'train']; val = used[used.role == 'validation']
assert train.event_uid.nunique() == 2880 and val.event_uid.nunique() == 360
assert not set(train.event_uid) & set(history[history.role != 'train'].event_uid)
assert not set(val.event_uid) & set(history[history.role == 'test'].event_uid)
assert not ((used.geometry == 'elliptical') & (used.strength_band == 2)).any()
audit = read(root/'audit.json')
test_suites = ET.parse(root/'tests.xml').getroot().iter('testsuite')
test_summary = {'tests':0, 'failures':0, 'errors':0}
for suite in test_suites:
    for key in test_summary: test_summary[key] += int(suite.attrib.get(key, 0))
assert test_summary['tests'] > 0 and test_summary['failures'] == test_summary['errors'] == 0
audit.update(hardware_equal_to_parent=True, all_old_seed_overlap=0, balanced_36_strata=True,
             no_old_holdouts_in_training=True, all_feature_cache_hashes='passed', tests=test_summary, report_source_hash=_hash_file(Path(__file__)))
_write_json(root/'audit.json', audit)

m = pd.read_csv(root/'metrics.csv'); pairs = read(root/'paired.json')
weak_delta = next(r for r in pairs if r['group'] == 'weak' and r['model'] == s['weak_specialist'] and r['reference'] == 'parent')
ordinary_delta = next(r for r in pairs if r['group'] == 'ordinary' and r['model'] == s['weak_specialist'] and r['reference'] == 'parent')
groups = ['ordinary', 'weak', 'strong_circle', 'strong_ellipse']
titles = ['通常テスト', '弱いイベント', '強い円形', '強い楕円形']
names = list(dict.fromkeys(['parent', s['primary'], s['weak_specialist'], 'prior', 'REI']))


def table(frame):
    columns = list(frame.columns)
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |'] +
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


lines = ['# d5: 時間窓・相関・背景補正の比較実験',
         '## 結論の読み方',
         '通常検証で最良だったのは既存モデルparentであり、新方式を全体用モデルとして採用する根拠は得られなかった。弱信号向け候補は別の副評価として扱う。平均誤差が減っても、固定位置基準を超えなければ位置情報の抽出に成功したとは主張しない。' if s['primary'] == 'parent' else '主モデルと弱信号向け候補は検証で選択し、新規テスト前に固定した。',
         f'弱信号候補−旧モデルの新規テスト差: 弱 {weak_delta["difference_mm"]:+.4f} mm、95%区間 [{weak_delta["ci95"][0]:+.4f}, {weak_delta["ci95"][1]:+.4f}]。通常 {ordinary_delta["difference_mm"]:+.4f} mm、95%区間 [{ordinary_delta["ci95"][0]:+.4f}, {ordinary_delta["ci95"][1]:+.4f}]。',
         '## 検証で固定したモデル',
         f'- 主評価モデル: `{s["primary"]}`。通常検証の平均位置誤差で選択。',
         f'- 弱イベント重視モデル: `{s["weak_specialist"]}`。通常検証が旧モデル+0.01 mm以内という制約で、弱検証の平均誤差を最小化。',
         '- `parent` は前回の2,880イベント学習SVR＋ExtraTreesを再学習せず使用。`prior` は学習座標の平均を常に答える基準。',
         '## 新規テスト結果',
         '全数値は今回の同じ未使用720イベント（各2ショット）で比較。小さいほど良い。通常600、弱240、強い円120、強い楕円120イベント。通常集合に弱・円を含む。']
summary = []
for group, title in zip(groups, titles):
    row = {'条件': title}
    for name in names:
        row[name+' (mm)'] = f'{m[(m.group == group) & (m.model == name)].iloc[0].mean_mm:.4f}'
    summary.append(row)
lines += [table(pd.DataFrame(summary)), '## 対応付き差と95%区間',
          '負の差は改善。物理イベント単位で2ショットをまとめて10,000回bootstrap。記述的区間であり、多重比較補正・学習集合再抽出は行っていない。主検定対象は主評価モデル−parentの通常集合。']
pair_rows = []
for r in pairs:
    if r['group'] not in groups: continue
    lo, hi = r['ci95']
    pair_rows.append({'条件': r['group'], 'モデル': r['model'], '基準': r['reference'], '差 mm': f'{r["difference_mm"]:+.4f}',
                      '95%区間': f'[{lo:+.4f}, {hi:+.4f}]', '判断': '改善側' if hi < 0 else '悪化側' if lo > 0 else '差は明確でない'})
lines += [table(pd.DataFrame(pair_rows)), '## 実験した変更',
          '- baseline: 従来の169特徴量。新しい学習手法との対照。',
          '- multiscale: 4/16等分時間窓のチェック別発生率を追加（649特徴量）。16分割は約128 round。',
          '- whitened: multiscaleの空間24チェックの共分散を独立平常データから推定し、縮小共分散で補正（649特徴量）。',
          '- correlation: 近隣チェックの同時発火、±1 roundの遅延ペア、自己の1 round遅延を4窓で集計（1,009特徴量）。平常時のペア発生率を差し引く。',
          '- combined: 背景補正付き時間窓＋相関（1,489特徴量）。',
          '- 各入力でRidge 2設定、SVR 3設定、ExtraTrees 3設定、弱イベント4倍重みのExtraTreesを学習。計45学習。SVR/ETの座標平均も比較。',
          '- 初期35学習の探索に旧選択ハイパーパラメータが含まれていなかったため、検証段階で全5入力に等しく追加した（validation_amendment.json、追加10学習）。探索的な検証変更であり、未使用テスト生成前に全設定を固定した。',
          '- 検証で改善しなかったため、multiscale/combinedに学習データのみでfitするPCA 32/128次元＋SVRを8学習追加。さらにbaseline特徴から弱信号を推定する分類器（ロジスティック回帰/ExtraTrees 2設定）を3学習し、旧予測と固定座標/弱専門家を混ぜる24設定を比較した。合計56学習。追加はrefinement_amendment.jsonに記録、最終テスト生成前に固定。',
          '- 観測信号が小さいときに学習平均座標へ寄せる3閾値のゲートを検証。閾値は学習信号の分位点から固定。真の強度は推定時に使わない。',
          '## 検証スコア（テストでモデルを選び直さない）',
          table(pd.read_csv(root/'validation_scores.csv').rename(columns={'Unnamed: 0': 'model'}).round(5)),
          '## 各入力の検証最良設定をテストで評価（副評価）',
          table(m[m.group.isin(groups)].round(5)),
          '## 再現性・制約',
          '- d5、49量子ビット、24チェック、2.048 ms、通常背景ノイズ、同じ固定ハードウェア・発生領域。入力は1ショットの[24,2047] detector events。出力は2次元座標。',
          '- 独立学習2,880イベント・検証360イベントは前回と同じ。過去テストは学習・選択に不使用。検証集合は再利用しているため、検証改善のみを成功とはしない。',
          '- 今回の平常較正は独立512ショット。旧モデルの128ショット較正は変更せず維持。追加較正資源を伴うので、全てを純粋なモデル構造改善とは呼ばない。',
          '- 強い楕円は学習・検証選択から除外した形状×強度の組合せ外挿。弱/中強度の楕円は学習に含む。',
          '- 1つの放射線イベントがあると既知の観測窓。発生有無・オンライン時刻検出・複数イベント・別ハードウェア・BB codeには未検証。',
          '- REIは前回選択済みの履歴長1024を固定した適用版。原著実装・原著条件を上回るとの主張ではない。',
          '- モデル/特徴量/データハッシュ、36層均等、全旧データとのseed非重複、ハードウェア同一、旧holdout非混入を監査。',
          f'- 実装テスト{test_summary["tests"]}件通過。tests.xmlに保存。',
          '## 成果物',
          '- `protocol.json`, `selection.json`: 探索計画と新規テスト生成前の設定固定。',
          '- `validation_scores.csv`, `metrics.csv`, `paired.json`, `predictions.csv`: 検証・テスト・対応付き差・予測。',
          '- `calibration_bits.npz`, `calibration.joblib`: 独立平常較正。',
          '- 各入力名ディレクトリの `.joblib`: 学習済みモデル。`audit.json`: 監査。']
(root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
plot_names = list(dict.fromkeys(['parent', s['primary'], s['weak_specialist']]))
labels = {name: f'Candidate {i}' for i, name in enumerate(plot_names)}
labels['parent'] = 'Previous SVR + ET'
if s['primary'] != 'parent': labels[s['primary']] = 'Validation-selected'
if s['weak_specialist'] != s['primary']: labels[s['weak_specialist']] = 'Weak-focused'
for ax, group, title in zip(axes.flat, groups, ['Standard test', 'Weak events', 'Strong circular', 'Strong elliptical']):
    values = [m[(m.group == group) & (m.model == name)].iloc[0].mean_mm for name in plot_names]
    bars = ax.bar([labels[n] for n in plot_names], values, color=['#527a9f', '#32957c', '#db923c'][:len(values)])
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set_ylim(0, max(values)*1.18); ax.set_ylabel('Mean location error (mm)'); ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Syndrome feature experiments — fresh held-out events')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(root/f'comparison.{ext}', dpi=160)
print(root/'RESULT_JA.md')
print('AUDIT', audit)
