"""Read-only analysis of predictions and rendering of this medium-only study."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree
import pandas as pd


def table(df):
    def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def main(root):
    read=lambda n:json.loads((root/n).read_text())
    s=read('selection.json');audit=read('audit.json');paired=read('paired.json')
    m=pd.read_csv(root/'metrics.csv').set_index('model');primary=next(a for a in paired if a['primary'])
    selected=s['primary'];delta=primary['difference_mm'];ci=primary['ci95'];percent=-delta/m.loc['reference','mean_mm']*100
    result='改善側' if ci[1]<0 else '悪化側' if ci[0]>0 else '0を含む（差は不明確）'
    grids=[]
    for n in [288,576,864]:
        for h in [4,8,16]:
            grids.append(dict(train_events=n,observation_ms=h,SVR=m.loc[f'n{n}_h{h}_SVR','mean_mm'],
                binomial=m.loc[f'n{n}_h{h}_binomial','mean_mm'],GLS=m.loc[f'n{n}_h{h}_GLS','mean_mm'],validation_selected=s['selected'][f'n{n}_h{h}']))
    headline=['reference','old_allband_SVR4','old_allband_template4',selected]
    tails=m.loc[headline].reset_index()
    for c in ['within_1mm','over_3mm']:tails[c]*=100
    tails=tails.rename(columns={'within_1mm':'within_1mm_percent','over_3mm':'over_3mm_percent'})
    comparisons=pd.DataFrame([dict(model=a['model'],reference=a['reference'],difference_mm=a['difference_mm'],
        ci95=f"[{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]",primary=a['primary']) for a in paired])
    seeds=pd.read_csv(root/'seed_metrics.csv');seeds=seeds[seeds.model.isin(headline)][['seed','model','mean_mm','p90_mm','within_1mm']]
    tests=ElementTree.parse(root/'tests.xml').getroot()[0].attrib
    assert tests['failures']=='0' and tests['errors']=='0'
    text=f'''# 中イベント：照合方法・観測時間・学習量の実験

2026-09-11。新規の中イベント180件・360ショット・3生成rootを、全方式で共通評価。mmは物理座標のユークリッド誤差、低い方が良い。

## 主結果

**平均誤差と大外れには改善があったが、1mm以内率の大幅な改善はまだ確認できない。** 主比較は平均2.0842→1.8252mm（約12.4%減）、p90は4.1414→3.5664mm、3mm超率は21.67→15.83%。1mm以内率は26.39→29.44%だが、差の探索的95%区間は[-2.22,+8.33]ポイントで0を含む。

3テストroot全てで主方式の平均は基準より改善した。なお、過去の別テストの2.28mm等とは直接差を取らず、今回は同一の新規180イベント内で比較する。

検証で事前選択した16ms・864学習イベントの方式は **{selected}**。

4ms・288中イベントの温度1固定binomial照合に対し、平均誤差は **{m.loc['reference','mean_mm']:.4f}→{m.loc[selected,'mean_mm']:.4f}mm**、約{percent:.1f}%減。
対応付き平均差は{delta:+.4f}mm、95%区間[{ci[0]:+.4f}, {ci[1]:+.4f}]で、{result}。

{table(tails)}

旧全強度モデルは参考比較であり、学習分布・学習数が違う。4msから16msへの変更には約12.3msの追加待ち時間がある。モデル構造だけの改善ではない。

## 何を実験したか

- d5 rotated surface-code memory-Z、24check、1μs/round。観測は二値detector eventsで、生のスタビライザー測定値とは区別する。
- 中のみの新規学習864イベント、検証96、テスト180。各2ショットを同一分割に置く。288⊂576⊂864で学習量を比較。
- 円/楕円×ballistic/diffusive×位置領域の12層は各学習量で均等。位置ラベルoutsideは生成器の参照領域に対する区分で、実際のチップ外を意味しない。
- 同じ16.384ms記録の先頭4.096/8.192/16.384msを比較。先頭の時間区切りを保存し、未来の情報は短時間特徴に入れない。
- 各学習量×時間で、相対空間分布SVR、binomial物理照合、GLS物理照合。計27構成。物理照合の温度は0.1/1/10/100から検証のみで選択。
- **binomial**：各check・時間区間の発火数を、学習した物理確率テンプレートと照合。check/時間間の相関は無視した複合スコア。
- **GLS**：学習時の観測率−真の確率の残差から縮小共分散を推定。ノイズ相関で補正した距離で照合し、候補の中心座標を重み付き平均。
- 潜在的な真の確率は教師・テンプレート作成用。推定時は観測率と学習済みの固定テンプレート・共分散のみで、問い合わせイベントの真の位置・形状・強度・時刻は与えない。

## 27構成の平均誤差

{table(pd.DataFrame(grids))}

各セルのvalidation_selectedは未使用テスト生成前に固定した方式。テストで最も低かった方式を選び直していない。

## 何が効いたか

1. **4→8msの延長が主要な改善要因**。864件binomialでは2.0310→1.8398mm（約9.4%減）、差95%区間[-0.2773,-0.1060]mm。学習量・方式を固定した対応比較。
2. **8→16msの追加利益は不明確**。同じ864件binomialは1.8398→1.8252mm、差[-0.0916,+0.0628]mm。GLSでも1.7975→1.7773mm、差[-0.0883,+0.0484]mm。待ち時間は約2倍なので、16msが必須とは言えない。
3. **学習量の効果は小さく、方式による**。288→864件で4ms binomialは2.0842→2.0310mm（約2.5%減、区間改善側）。16ms binomialは1.8726→1.8252mmだが差区間は0を含む。16ms GLSは1.8533→1.7773mm（約4.1%減、探索的区間改善側）。学習量を3倍にすれば精度が大幅に上がる、とは言えない。
4. **相関を考慮した照合は小幅に有望**。864件16msでbinomial1.8252→GLS1.7773mm、差95%[-0.0836,-0.0130]mm。これは主選択外の探索的結果で、テスト最良のGLSに主結果を差し替えない。

[8→16msと尾部の追加解析](tail_and_8to16_exploratory.json)は探索的な対応付きイベントbootstrap。主比較以外の有意性は記述的で、多数の比較を一括保証しない。主比較のp90差[-0.9573,-0.0393]mm、3mm超率の差[-9.72,-1.94]ポイントは探索的区間で改善側。

現時点では、**8ms＋中専用物理照合を中心に次の検証を行う**のが時間との折り合いとして有望。ただし1mm以内率は約3割にとどまり、精密推定が解決したとは言わない。8ms GLSを運用上の新たな選択として確定するには、別テストでの追試と強度判定ゲートを含む評価が必要。

## 1要因ずつの差

負が改善。主比較以外は多重比較補正なしの記述的95%区間であり、多数の探索から一番良いものだけを主張しない。

{table(comparisons)}

## 生成root別

各root60物理イベント。within_1mmは0–1表記。

{table(seeds)}

## 全指標

率は0–1表記。

{table(m.reset_index())}

## 解釈の限界

追加の検証診断（実用成績ではない）：[oracle_validation.csv](oracle_validation.csv)。864件のbinomial照合は、実測率を真の発火確率で置き換えると、4msで2.1910→1.2731mm、16msで1.7560→1.0276mm。最寄り位置候補は平均0.1174mmまで近い。観測の揺らぎだけでなく、位置と物理条件の照合にも誤差が残ることを示唆する。単純な候補間隔を達成可能な誤差下限とは扱わない。

[選択された温度と共分散縮小率](template_settings.csv)では全照合の温度は1。GLSの縮小率は約0.67–0.96と高く、推定共分散は強く単純化されている。このGLSが勝たなくても「ノイズ相関が重要でない」と一般化することはできない。

- **中イベントと分かっている条件付き評価**。中専用モデルの利用前に必要な強度判定ゲートは未実装。弱/強/無イベントでの運用成績ではない。
- 位置候補だけを増やす実験ではなく、学習増量では物理条件のカバーも増える。学習イベントは共通の1集合から入れ子で、独立学習集合3回ではない。
- GLSは共通共分散・Gaussian近似。有限サンプルで推定した相関が必ず正しいとは限らず、厳密な量子回路の同時尤度でもない。
- binomial/GLSにはシミュレータ由来の真の確率教師を使う。実測シンドロームだけから同じ学習ができるという主張はしない。
- 同一の現象論的伝播＋局所QP-ODE＋Pauli近似シミュレータ、同一デバイス。実機や未知ノイズ分布への汎化、オンライン復号への効果は未検証。
- 16msは放射線が存在する既知の観測窓全体を待って推定する。観測待ち時間を含めた実用性は別評価。既定モデルを自動置換していない。

## 監査と再現

モデル・ソース・較正入りprotocol・学習manifestを固定後にテスト生成。イベント・generation seedの旧データとの非重複を確認。2ショットを独立イベントとして数えず、イベント平均誤差差を10,000回bootstrap。

自動テスト{tests['tests']}件通過。入力が真値に依存しないスコア関数、短窓が未来を参照しないこと、GLSスコアとMahalanobis距離の一致、物理イベント1件につき1テンプレートであることもテスト。

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.medium_inverse prepare NEW_ROOT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.medium_inverse generate_fit NEW_ROOT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.medium_inverse train NEW_ROOT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.medium_inverse generate_test NEW_ROOT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.medium_inverse evaluate NEW_ROOT
.venv/bin/python examples/report_medium_inverse.py NEW_ROOT
```

prepareは兄弟のradical_pilot_20260911の固定protocol・較正を参照する。別の独立追試には生成seedも変更する必要がある。

シンドロームだけで推論する入口：

```text
.venv/bin/python -m qp_ode_simulator.medium_predict ROOT OBSERVED.npz NEW_OUTPUT.csv --model n864_h16_GLS
```

`OBSERVED.npz`の`raw`は[shot,24check,time]の二値配列。指定したモデルに必要な先頭ラウンドだけを読む。`--model`省略時は検証選択の主方式。出力はx_mm,y_mmで、座標や伝播条件の正解は入力しない。既存CSVの上書きは拒否する。

[事前計画](PLAN_JA.md) / [固定選択](selection.json) / [監査](audit.json) / [全指標](metrics.csv) / [予測](predictions.csv) / [対応区間](paired.json) / [テスト](tests.xml)

[真値なし推論の動作確認](inference_smoke.json)。SVR、binomial、GLSで直接計算と推論入口の一致を確認し、ラウンド不足の入力も拒否する。
'''
    (root/'RESULT_JA.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
