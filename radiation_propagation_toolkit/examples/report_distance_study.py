"""Summarize frozen distance study predictions without model reselection."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root=Path(sys.argv[1]).resolve()
m=pd.read_csv(root/'metrics.csv')
f=pd.read_csv(root/'predictions.csv')
selection=json.loads((root/'selected_before_test.json').read_text())
paired=json.loads((root/'paired.json').read_text())
families=['prior','REI','Ridge','SVR','ExtraTrees','SVR_ET_blend','CNN']
groups=['ordinary','weak','strong_circle','strong_ellipse']
labels=['Standard test set','Weak events','Strong circular events','Strong elliptical events']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(11,8))
colors=['#888888','#333333','#7e57a4','#3276ac','#349c79','#cf8c24','#cf5555']
for ax,group,title in zip(axes.flat,groups,labels):
    for family,color in zip(families,colors):
        sub=m[(m.family==family)&(m.group==group)].sort_values('distance')
        name={'REI':'Adapted REI','prior':'Fixed position','SVR_ET_blend':'SVR + ExtraTrees','CNN':'CNN (3-seed ensemble)'}.get(family,family)
        ax.plot(sub.distance,sub.mean_mm,marker='o',label=name,color=color,linestyle='--' if family=='prior' else '-')
    ax.set_xticks([3,5,7]);ax.set_xlabel('Code distance');ax.set_ylabel('Mean location error (mm)');ax.set_title(title);ax.grid(alpha=.2)
fig.suptitle('Localization versus surface-code distance',fontsize=19)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,.1,1,.95))
for ext in ['png','pdf','svg']:fig.savefig(root/f'distance_comparison.{ext}',dpi=180)
plt.close(fig)

def table(rows,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:lines.append('| '+' | '.join(str(row[c]) for c in columns)+' |')
    return '\n'.join(lines)

report=['# 符号距離 d=3・5・7 の対応付き比較',
'## 結論',
'距離を増やす効果は強いイベントで確認できた。一方、弱いイベントの精密推定は未解決。全距離で検証選択されたSVR+ExtraTrees座標平均を主要な比較対象とする。',
'- 通常平均: d3 2.8575 → d5 2.6720 → d7 2.6811 mm。d5/d7とd3の対応付き差の95%区間は改善側。',
'- 強い円: 1.4164 → 0.9959 → 0.9151 mm。d3→d7で35.4%改善。1 mm以内率44.8%→63.5%→70.8%。',
'- 強い楕円: 1.5721 → 1.1412 → 1.1622 mm。d3→d7で26.1%改善。',
'- 弱いイベント: 3.3941 → 3.3777 → 3.3490 mm。d7−d3の差 −0.0452 mm、95%CI [−0.1710,+0.0823]。固定位置3.3534 mmに対する優位性も未確認。',
'- REIも距離増加で改善。d7の通常/強い円/強い楕円は2.7633/1.2083/1.3403 mm。',
'- d7のSVR+ExtraTreesはREIより通常/強い円で平均が小さく、差の95%CIも改善側。ただし強い楕円の差は −0.1781 mm [−0.3522,+0.0037]で明確ではない。',
'- d5→d7の追加効果は補助集計で明確ではない。平均とp90で順位も異なる。d7が必ず最良とは言えない。',
'- 強いイベントのinside区分（d3基準）も平均1.1970→0.8493→0.9040 mmと改善傾向。境界外イベントの包含だけで全改善を説明するものではないが、領域別の小標本・事後集計である。',
'- 実装・データ監査: 100 tests passed。固定ハードウェア、同一物理イベント、共通位置の役割一致、過去生成seedとの非重複を確認。',
'## 実験条件',
'- 新規物理イベント1,440件 × 3距離 × 2ショット。独立イベント数は1,440件であり、8,640ショットではない。',
'- 36条件（円/楕円 × ballistic/diffusive × d3基準の内/端/外 × 弱/中/強）を均等生成。',
'- 864学習候補・288検証・288テスト。強い楕円を学習/検証選択から除き、実際の学習720イベント、検証240イベント。テスト通常240、強い楕円48、弱96、強い円48。',
'- 量子ビット座標の刻み1 mm、データ/測定間の最短距離√2 mm、同種間最短距離2 mm。d=3/5/7で17/49/97量子ビット、8/24/48チェック。',
'- 同じ物理座標の和集合上でQP/T1/T2を一度だけ計算し、各回路に切り出す。共通座標の物理場・ハードウェア特性は同じ。回路測定の乱数は距離ごとに独立。',
'- 発生位置の基準領域は全距離でx,y∈[-3,3] mmに固定。outsideはここから0.25–1 mm外側。大きい符号では旧outsideも内部になり得る。',
'- 同一面積に量子ビットを詰める実験ではなく、間隔を固定して観測範囲を拡大した実験。',
'- surface_code:rotated_memory_z、2.048 ms、1 µs/round、1イベント既知の固定窓。通常背景ノイズ。',
'- 独立平常128ショットで較正。特徴量は7×チェック数+1（57/169/337）。切り分け時刻は学習イベントの開始時刻中央値で固定。推定時には真の位置・強度・形・時刻を渡さない。',
'- Ridge α3候補、SVR C3×γ3候補、ExtraTrees葉サイズ3候補を各距離の検証集合で選択。γは特徴次元数で補正。SVRとExtraTreesの座標平均も評価。',
'- CNNは各距離で3seed、最大30epoch・早期停止6epoch、座標アンサンブル。REIは履歴長5候補、r=1・空間倍率2。',
'- 全距離のモデル/設定を selected_before_test.json に凍結してからテスト採点。推定不能REIは学習位置平均にfallbackし、回答率も報告。',
'- 以前のd3単独実験とは新規データ・ハードウェア集合・学習規模が異なる。旧数値との直接ランキングはしない。',
'## 同一テストにおける平均位置誤差（mm）']
for group,title in zip(groups,labels):
    rows=[]
    for family in families:
        row={'model':family}
        for d in [3,5,7]:row[f'd{d}']=f"{m[(m.family==family)&(m.group==group)&(m.distance==d)].iloc[0].mean_mm:.4f}"
        rows.append(row)
    report.extend(['### '+title,table(rows,['model','d3','d5','d7'])])
report+=['## d3からの差と95%信頼区間',
'差は大きい距離 − d3。負なら改善。イベントごとに2ショットを平均して対応付きbootstrap 10,000回。多重比較補正・学習データ再抽出による不確かさは含まない。']
rows=[]
for item in paired:
    if 'minus d3' not in item['comparison'] or item['group'] not in groups:continue
    rows.append(dict(comparison=item['comparison'],group=item['group'],difference=f"{item['difference_mm']:+.4f}",CI=f"[{item['ci95_mm'][0]:+.4f}, {item['ci95_mm'][1]:+.4f}]"))
report.append(table(rows,['comparison','group','difference','CI']))
report.append('## 補助集計: d7からd5を引いた差')
extra=[]
for group in groups:
    sub=f[f.family=='SVR_ET_blend']
    if group=='ordinary':sub=sub[~((sub.geometry=='elliptical')&(sub.strength_band==2))]
    elif group=='weak':sub=sub[sub.strength_band==0]
    elif group=='strong_circle':sub=sub[(sub.strength_band==2)&(sub.geometry=='circular')]
    else:sub=sub[(sub.strength_band==2)&(sub.geometry=='elliptical')]
    a=sub[sub.distance==7].groupby('event').error_mm.mean();b=sub[sub.distance==5].groupby('event').error_mm.mean()
    delta=(a-b).to_numpy();rng=np.random.default_rng(2026090905)
    ci=np.quantile(rng.choice(delta,(10000,len(delta))).mean(1),[.025,.975])
    extra.append(dict(group=group,difference_mm=float(delta.mean()),ci95_mm=ci.tolist()))
    report.append(f"- {group}: {delta.mean():+.4f} mm [{ci[0]:+.4f}, {ci[1]:+.4f}]")
report.append('上のd7対d5は追加の記述的比較。これを使ったモデル再選択・再学習はしていない。')
(root/'supplementary_d7_vs_d5.json').write_text(json.dumps(extra,indent=2)+'\n')
report.append('## 検証集合で選ばれた設定')
for d,info in selection.items():
    report.append(f"- d{d}: 最良family（検証選択）={info['best_family']}, REI履歴={info['selected']['REI']['history']}, Ridge={info['selected']['Ridge']['config']}, SVR={info['selected']['SVR']['config']}, ExtraTrees={info['selected']['ExtraTrees']['config']}")
# Report region-dependent effects, without selecting models from those results.
region=f.groupby(['distance','family','epicenter_region','strength_band']).error_mm.agg(['mean',lambda a:a.quantile(.9)])
region.columns=['mean_mm','p90_mm'];region.to_csv(root/'region_strength_metrics.csv')
report+=['## 注意点と成果物',
'- これはREI適用版との比較であり、原著の入力対応・オンライン検出条件の完全再現ではない。',
'- 形状/強度等のラベルは学習集合の層化と事後評価に使う。モデル選択には通常条件の検証平均のみを使う。',
'- 広い面積の観測と、境界の外側を囲む効果が含まれる。dを増やすだけで局所空間分解能が上がったとは言えない。',
'- 円/楕円ごとの同強度サンプルは独立であり、形だけを変えた対応付き比較ではない。',
'- metrics.csv: 平均、p90、1/2 mm以内率、回答率。paired.json: 距離間・REI/固定位置との差。',
'- region_strength_metrics.csv: d3基準の内/端/外×強度別集計。predictions.csv: 全テスト予測。',
'- distance_comparison.png/pdf/svg: 距離別比較図。protocol.json: 事前設定。',
'- モデルによる選択効果・3seed集合の差は、イベントbootstrapだけでは全て評価できない。追加の学習seed/独立データが必要。']
(root/'RESULT_JA.md').write_text('\n\n'.join(report)+'\n')
print(root/'RESULT_JA.md')
