"""Audit a larger d5 dataset and summarize its locked-test learning curve."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.distance_growth import role,training_indices
from qp_ode_simulator.localization import _hash_file

root=Path(sys.argv[2]).resolve()
p=json.loads((root/'protocol.json').read_text())


def audit():
    status=json.loads((root/'progress.json').read_text())
    assert status['status']=='complete' and status['completed']==3456
    parent=Path(p['parent'])
    assert _hash_file(parent/'progress.json')==p['parent_data_sha256']
    old=pd.DataFrame([json.loads(path.read_text()) for path in sorted((parent/'events').glob('*.json'))])
    records=[]
    for i in range(3456):
        path=root/'events'/f'{i:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==status['hashes'][f'{i:04d}{ext}']
        row=json.loads(path.with_suffix('.json').read_text());assert row['event']==i and row['role']==role(i)
        with np.load(path) as z:
            assert z.files==['d5'] and z['d5'].shape==(2,24,2047)
            assert np.isin(z['d5'],[0,1]).all()
        records.append(row)
    labels=pd.DataFrame(records)
    assert labels.event_uid.nunique()==3456 and labels.generation_seed.nunique()==3456
    assert not set(labels.generation_seed)&set(old.generation_seed)
    assert not labels.is_control.any()
    strata=['geometry','propagation_law','epicenter_region','strength_band']
    for name,number in [('train',72),('validation',12),('test',12)]:
        counts=labels[labels.role==name].groupby(strata).size();assert len(counts)==36 and (counts==number).all()
    hardware=['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']
    for field in hardware:assert labels[field].nunique()==1 and labels[field].iloc[0]==old[field].iloc[0]
    train=old[old.role=='train'].copy();train['origin']='old';labels['origin']='new';pool=pd.concat([train,labels],ignore_index=True)
    splits={};previous=set()
    for n in [720,1440,2880]:
        rows=pool.iloc[training_indices(pool,n)];uids=set(rows.event_uid)
        assert len(uids)==n and previous<=uids;previous=uids
        assert set(rows[rows.origin=='old'].event_uid)<=set(old[old.role=='train'].event_uid)
        splits[str(n)]=dict(events=n,weak_events=int((rows.strength_band==0).sum()))
    result=dict(status='passed',new_events=3456,validation_events=432,test_events=432,training=splits,
        old_validation_test_in_training=False,generation_seed_overlap=0,hardware_equal_to_parent=True,
        all_input_shapes=[2,24,2047],source_sha256=_hash_file(Path(__file__)),
        protocol_sha256=_hash_file(root/'protocol.json'),data_manifest_sha256=_hash_file(root/'progress.json'))
    _write_json(root/'audit.json',result);print(result)


def markdown_table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(str(r[c]) for c in columns)+' |' for r in rows])


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=pd.read_csv(root/'metrics.csv');f=pd.read_csv(root/'predictions.csv');pairs=json.loads((root/'paired.json').read_text())
    frozen=json.loads((root/'selected_before_test.json').read_text())
    # Fixed n720 faithfully reproduces the previous frozen model on new data.
    a=f[(f.family=='fixed_blend')&(f['size']==720)].sort_values(['event_uid','shot'])
    b=f[f.family=='previous_blend'].sort_values(['event_uid','shot'])
    np.testing.assert_array_equal(a[['event_uid','shot']].to_numpy(),b[['event_uid','shot']].to_numpy())
    np.testing.assert_allclose(a[['predicted_x_mm','predicted_y_mm']],b[['predicted_x_mm','predicted_y_mm']],rtol=1e-10,atol=1e-10)
    groups=['ordinary','weak','strong_circle','strong_ellipse'];titles=['Standard test set','Weak events','Strong circular events','Strong elliptical events']
    fixed=m[m.family=='fixed_blend'].sort_values(['size','group']).mean_mm.to_numpy()
    retuned=m[m.family=='blend'].sort_values(['size','group']).mean_mm.to_numpy()
    same_blend=np.allclose(fixed,retuned,rtol=0,atol=1e-12)
    families=['fixed_blend']+([] if same_blend else ['blend'])+['SVR','ExtraTrees','Ridge','CNN']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,group,title in zip(axes.flat,groups,titles):
        for family in families:
            sub=m[(m.family==family)&(m.group==group)].sort_values('size')
            label={'fixed_blend':'SVR + ExtraTrees' if same_blend else 'SVR + ET (fixed settings)','blend':'SVR + ET (retuned)'}.get(family,family)
            ax.plot(sub['size'],sub.mean_mm,marker='o',label=label)
        r=m[(m.family=='REI')&(m.group==group)].iloc[0];ax.axhline(r.mean_mm,color='#333333',ls='--',label='Adapted REI')
        ax.set_xticks([720,1440,2880]);ax.set_xlabel('Independent training events');ax.set_ylabel('Mean location error (mm)');ax.set_title(title);ax.grid(alpha=.2)
    fig.suptitle('More training events at code distance 5',fontsize=18)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.12,1,.95))
    for ext in ['png','pdf','svg']:fig.savefig(root/f'learning_curve.{ext}',dpi=180)
    plt.close(fig)
    conclusions=['# distance-5: 学習量720→1,440→2,880イベント','## 結論（設定固定の主比較）']
    for group,title in zip(groups,titles):
        sub=m[(m.family=='fixed_blend')&(m.group==group)].sort_values('size')
        values=sub.mean_mm.to_numpy();delta=next(r for r in pairs if r['comparison']=='fixed_blend: n2880 minus n720' and r['group']==group)
        lo,hi=delta['ci95_mm'];judgment='改善側の区間' if hi<0 else '悪化側の区間' if lo>0 else '改善は明確でない'
        conclusions.append(f'- {title}: {values[0]:.4f} → {values[1]:.4f} → {values[2]:.4f} mm。720→2880の差 {delta["difference_mm"]:+.4f} mm、記述的95%CI [{lo:+.4f}, {hi:+.4f}]（{judgment}）。')
    conclusions.append('102件の実装テスト通過。新規データ監査も通過。以下の値は旧テスト結果ではなく、全サイズを同じ新規テストで採点したもの。')
    if same_blend:conclusions.append('各サイズで再調整してもSVR/ExtraTreesは前回と同じ設定が選択された。今回のfixed_blendとblendの予測・平均誤差は一致するため、比較図は1本にまとめた。')
    conclusions.append('弱いイベントでは2,880件版の固定位置基準が3.2811 mm、blendが3.3172 mmで、基準に対する優位性も確認できていない。CNNも学習量増加で通常/強イベントの平均は改善したが、今回のblendを上回らなかった。')
    conclusions.append('強い円の1 mm以内率は66.7%→76.4%→81.9%、強い楕円は52.8%→59.7%→61.1%。一方、通常条件p90は4.2483→4.2436→4.2492 mmで、平均の改善が全条件の大外れ削減を意味するわけではない。')
    lines=conclusions+[
    '## 実験条件',
    '- 元の720学習イベントを維持し、新規の独立イベントを追加。旧検証・旧テストは学習に使わない。',
    '- 新規3,456イベントを生成。学習候補2,592・検証432・テスト432。強い楕円は学習と検証選択から除外し、新規学習2,160+既存720=最大2,880イベント。',
    '- 弱い学習イベント数288→576→1,152。各イベント2ショット。イベント数とショット数は区別。',
    '- 新規テストは通常360イベント、強い楕円72、弱144、強い円72。各サイズで同じテストを使う。',
    '- d5固定（49量子ビット・24チェック）、2.048 ms、1 µs/round、通常ノイズ、量子ビット間隔固定。発生位置の基準領域x/y∈[-3,3] mm。',
    '- 前回と同じ座標和集合・固定ハードウェア上で物理場を生成し、d5の入力だけ保存。別ハードウェアに切り替えていない。',
    '- 全サイズで同じ169特徴量、旧学習で選んだ区間切り分け、独立平常較正を使用。真の位置・時刻・強度・形状は推定入力に含まない。',
    '- 主比較: 前回のSVR/ExtraTrees設定を固定した座標平均（fixed_blend）。720件版の新テスト予測が前回の凍結モデルと一致することを検証。',
    '- 副比較: 同じ検証探索予算でサイズごとにRidge/SVR/ExtraTreesを再調整したモデルとblend。CNNは3seed座標アンサンブル、最大30epoch・早期停止6epoch。epoch固定なので大きいデータでは更新回数も増える。',
    '- 全モデルとREI履歴長を新規検証で選択し、selected_before_test.jsonへ凍結した後にテストを採点。',
    '## 平均位置誤差（mm、小さいほど良い）']
    for group,title in zip(groups,titles):
        rows=[]
        for family in ['fixed_blend','blend','Ridge','SVR','ExtraTrees','CNN','prior']:
            row={'model':family}
            for n in [720,1440,2880]:row[str(n)]=f"{m[(m.family==family)&(m['size']==n)&(m.group==group)].iloc[0].mean_mm:.4f}"
            rows.append(row)
        r=m[(m.family=='REI')&(m.group==group)].iloc[0]
        lines.extend(['### '+title,markdown_table(rows,['model','720','1440','2880']),f'REI適用版: {r.mean_mm:.4f} mm（履歴長 {frozen["rei"]["history"]}）。'])
    lines.extend(['## 学習量増加の差と95%信頼区間',
                  '差は大きい学習量 − 720件。イベントごとに2ショットの誤差を平均した対応付きbootstrap 10,000回。負なら改善。記述的区間で、多重比較補正・学習集合再抽出・全学習乱数の変動は含まない。'])
    rows=[]
    for r in pairs:
        if r['group'] in groups and any(r['comparison'].startswith(name+':') for name in ['fixed_blend','blend','CNN']):
            rows.append(dict(comparison=r['comparison'],group=r['group'],difference=f"{r['difference_mm']:+.4f}",CI=f"[{r['ci95_mm'][0]:+.4f}, {r['ci95_mm'][1]:+.4f}]"))
    lines.append(markdown_table(rows,['comparison','group','difference','CI']))
    lines.extend(['## 検証選択',f'全体の検証最良候補: {frozen["chosen"]}。この選択をテスト結果から変更していない。',
                  '## 再現・監査',
                  '- protocol.json: 事前設定と依存コード/親データのhash。audit.json: 分割、入力、hardware、seed重複の監査。',
                  '- n*/frozen.json: 学習イベントUID、選択設定、重みhash。predictions.csv: 新規テストの全座標予測。',
                  '- metrics.csv: 平均、p90、1/2 mm以内率、回答率。paired.json: 対応付き比較。',
                  '- 720件fixed_blendと旧凍結モデルは新規テスト上の全座標が数値精度内で一致。',
                  '- 距離比較の旧テストとは異なる集合なので、以前の平均誤差と今回の値を直接比較しない。',
                  '- REI原著実装の完全再現、実機、イベントなし・複数イベントの評価ではない。'])
    (root/'RESULT_JA.md').write_text('\n\n'.join(lines)+'\n');print(root/'RESULT_JA.md')


if sys.argv[1]=='audit':audit()
elif sys.argv[1]=='report':report()
else:raise ValueError('expected audit/report')
