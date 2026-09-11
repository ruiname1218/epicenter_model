"""Render the frozen pilot's results; does not select or retrain models."""
import argparse
from pathlib import Path
import json
import pandas as pd


def table(frame):
    def fmt(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main(root):
    read=lambda name:json.loads((root/name).read_text())
    metrics=pd.read_csv(root/'metrics.csv');selection=read('selection.json');audit=read('audit.json');paired=read('paired.json')
    pivot=metrics.pivot(index='model',columns='group',values='mean_mm')
    software=['base_SVR','base_ET','rates_plain','coinc_plain','coinc_joint','coinc_aux','forward_inverse','template_inverse','Adapted_REI','Prior']
    def means(names):return table(pivot.loc[names,['weak','medium','strong','strong_circle','strong_ellipse']].reset_index())
    oracle=[n for n in pivot.index if n.startswith('oracle_')]
    layouts=['base_SVR','dense_SVR','pair_SVR','base_ET','dense_ET','pair_ET']
    tails=metrics[metrics.model.isin(['base_SVR',selection['selected'],'dense_SVR','pair_SVR'])&metrics.group.isin(['medium','strong'])]
    tails=tails[['model','group','within_1mm','p90_mm','p95_mm','over_3mm']].copy()
    comparisons=[]
    for v in paired:
        if v['primary'] or (v['model'] in ['coinc_plain','coinc_joint','coinc_aux','forward_inverse'] and v['reference'] in ['rates_plain','coinc_plain','template_inverse']):
            interval=v['ci_primary'] if v['primary'] else v['ci95']
            comparisons.append(dict(model=v['model'],reference=v['reference'],group=v['group'],difference_mm=v['difference_mm'],
                interval=f'[{interval[0]:+.4f}, {interval[1]:+.4f}]',level='99.1667%' if v['primary'] else '95% exploratory'))
    blocks=pd.read_csv(root/'seed_metrics.csv')
    blocks=blocks[blocks.model.isin(['base_SVR',selection['selected'],'dense_SVR','pair_SVR'])&blocks.group.isin(['medium','strong'])][['test_seed','model','group','mean_mm']]
    epochs=pd.DataFrame([dict(model=n,epoch=v['best_epoch'],seconds=v['seconds']) for n,v in selection['cost'].items() if isinstance(v,dict)])
    txt=f'''# 抜本的改善4方向：新規データの横断パイロット

2026-09-11。全数値は同一の新規テスト540物理イベント（各2ショット）の位置誤差。単位mm、低い方が良い。

## 結論

- **通常の観測のまま、中イベントには物理テンプレート照合が有望**。通常SVRの2.6391→2.2759mm、約13.8%減。主比較99.1667%対応区間は差[-0.5178,-0.2071]mmで改善側、3テストrootとも改善。1mm以内率5.28→15.28%。ただしp90は4.0752→4.1154mmで改善せず、強は0.8020→1.1094mmに悪化。万能な置き換えではない。
- **同じ面積の高密度化は中に効果**。49→97qubitで中2.6391→2.4208mm、約8.3%減。主差区間[-0.3494,-0.0839]mm、3rootとも改善。中p90は3.8855mm。強0.8363mmは改善せず、平均差区間も0を含む。ハードウェア資源増加と仮定変更を含む。
- **隣接2パッチは今回の配置・学習では悪化**。中2.7489・強0.8671mm。遠い情報を足せば必ず良くなるわけではないが、パッチ配置や特徴正規化・学習量の違いを残しているので、多パッチ一般の否定ではない。
- **時間CNN・物理補助学習は通常SVRを上回らず**。中2.717–2.772、強0.996–1.057mm。補助課題そのものはある程度学べているが、位置精度には十分つながらなかった。
- **学習forward近似の逆探索は不成功**。中3.2424・強2.5368mm。強帯の検証確率RMSEは0.0551で一定確率対照0.0538より悪く、今回の近似では強イベントの応答を十分表現できていない。これは物理逆問題そのものの限界とは言えない。
- **真の伝播条件を追加するだけでは改善せず**。一方、真のdetector確率を渡すET診断は弱0.7825・中0.7356・強0.7628mm。有限のシンドロームから応答を推定する段階が重要と示唆する。ただしoracleは実用成績でも達成可能な下限の証明でもない。

弱イベントの精密な位置推定は依然未解決。全条件・尾部を同時に改善する新モデルは得ていないので、既定モデルは維持する。今回の最優先の次実験は、**中に照合、強にSVRを使い分ける推定ゲート**。真の強度を運用時に与えず、学習イベント内のOOF予測でゲートを学習し、別の新規テストで評価する必要がある。

## 実施したこと

1. 真の発生時刻・主要17伝播パラメータ・真の条件付きdetector確率を与えるoracle診断。
2. 学習したforward近似からの逆探索、および学習時の物理応答テンプレートからの逆探索。
3. 時間CNNで、発火率のみ／同時発火追加／伝播条件の補助学習／物理応答の補助学習を比較。
4. 通常d5、高密度d7、隣接する2つのd5を同じ物理イベントで比較。

回帰14学習＋時間CNN12学習（4アーム×3seed）＋forward MLP1学習、計27学習。逆探索の温度は検証のみで選択。

## 実験条件とカンニング対策

- 学習{audit['train_events']}、検証{audit['validation_events']}、テスト{audit['test_events']}イベント。各2ショットを同じ分割に置く。学習は弱288・中288・強144、検証は48・48・24。テストは各強度180、3生成root。
- 強い楕円は学習・検証に含めず、テストには90イベント含める。強い円も90イベント。
- 約4.096ms、1μs/round。入力は生の測定値そのものではなく、4095内部ラウンドの二値detector events。
- 通常配置の推論はシンドロームだけ。真の座標は学習ラベル、伝播条件・物理応答は補助教師に限定。oracleは明示的な別条件。
- 位置と17条件からのforward近似は、学習由来192位置×64条件を照合。問い合わせイベントの真の条件を渡していない。
- モデル・前処理・較正・候補ライブラリ・ソースのhashを固定してからテスト生成。旧実験のgeneration seedとの非重複も確認。
- 中・強の平均を等重みで評価した検証選択は **{selection['selected']}**。oracleと配置変更はこの選択から除外。テストを見た選び直しはしない。
- 同一の学習イベントを使うため、3ニューラルseedは初期化再現性のみ。3テストrootは新規イベント生成の再現性。独立した学習データ3集合での再学習ではない。

## 通常配置の実用モデル

CNNの各行は3seedの座標平均。SVRとETはそれぞれ1fit。入力は、SVR/ETが相対空間分布120特徴、CNNが128時間区間×24checkの発火率と同時発火。入力表現が異なるため、モデル構造だけの優劣とは解釈しない。

{means(software)}

Adapted_REIは同じ通常配置の観測を使った適用版、終端history1024。推定不能時は学習座標平均に置き換える。回答率{audit['rei_answer_rate']:.2%}。原著条件・原著実装での優越性を示す比較ではない。

## 配置の変更：ハードウェア資源の増加を含む

通常：49qubit・24check。高密度：97qubit・48checkで外接箱[-5,5]mmを維持。2パッチ：通常＋右11mmに独立d5、98qubit・48checkで観測面積を拡大。同一の物理場を全配置の座標和集合でシミュレートし、2パッチの左側は通常と同じ記録。

{means(layouts)}

配置ごとの量子故障は独立サンプル。高密度化によるクロストーク・ゲート速度・配線条件の悪化は含めない。固定デバイスは座標和集合上で定義され、過去の別デバイス配置のスコアと直接比較しない。2パッチは右側1配置のみで、最適配置探索ではない。

## Oracle診断：実用モデルの性能ではない

onsetは真の発生時刻、nuisanceは17パラメータ、nuisance_onlyはシンドロームなしの負の対照、meanは真の条件付きdetector確率。いずれもテスト真値を要求する。

{means(oracle)}

oracle_meanは確率を与える診断で、有限ショットの実測シンドロームとは異なる。これが良くても、それを単一ショットから復元できるとは限らない。oracle_nuisanceの非改善も、他の推定器や大規模学習で改善しないことの証明ではない。

## 対応付き誤差差と不確かさ

差はmodel−referenceで負が改善。イベント内2ショットを平均してから10,000回bootstrap。主比較は通常配置の検証選択、高密度SVR、2パッチSVRの各中/強（6比較）に対するBonferroni水準99.1667%。選択が基準自身なら自己比較を省略し、残りにも保守的に同じ水準を使う。それ以外は記述的95%。研究全体の反復探索に対する一括保証ではない。

{table(pd.DataFrame(comparisons))}

## 1mm以内率・尾部

率は0–1表記。

{table(tails)}

## テスト生成root別

{table(blocks)}

## 時間モデルの学習

全アームは同じConv1d構造・ヘッド数。rates_plainは同時発火入力を0にする。coinc_plainは座標損失のみ、coinc_jointは標準化17条件のMSEを重み0.1で追加、coinc_auxはさらに標準化384物理応答のMSEを0.1で追加。近傍は幾何学的に近い3checkで、厳密な回路相互作用グラフではない。

最大80epochから検証で選択。学習は座標MSE、選択は中/強のユークリッド誤差平均。学習イベント数720の小規模条件であり、構造の性能限界ではない。

[補助教師の診断](auxiliary_diagnostic.csv)では、coinc_auxの物理応答RMSEは一定の学習平均対照から、弱0.00740→0.00282、中0.00614→0.00293、強0.05970→0.02824に減る。しかし位置誤差はSVRより大きい。「物理応答のMSEを減らす」ことと「中心の識別に必要な差を保つ」ことは同じではない。伝播条件の標準化RMSEは大部分が平均予測の水準付近であり、条件を精密に復元できたとは言えない。

{table(epochs)}

## 逆探索の限界と検証

forwardの検証確率RMSE：{selection['forward_validation_probability_rmse']:.6f}。選択温度：{selection['inverse_settings']}。

強度別の補足診断は[forward_validation_diagnostic.json](forward_validation_diagnostic.json)。特に強帯では一定の平均確率を返す対照のRMSEも確認し、全強度RMSEだけで近似の成功とは扱わない。

forward教師はシミュレータの条件付きdetector確率だが、17パラメータは全ての物理状態を表さない。形状指数などの未入力条件・確率的空間模様が残る。16時間区間、192位置候補、近似MLPの誤差、相関を無視したbinomial複合スコアも制約である。逆探索が失敗しても物理モデルによる推定全体の不可能性とは扱わない。

## 再現と成果物

実装：`src/qp_ode_simulator/radical_data.py`、`radical_models.py`。リポジトリで環境変数`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`を設定し、以下を新しい出力先に順番に実行する。

```text
.venv/bin/python -m qp_ode_simulator.radical_data prepare NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data generate_fit NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data cache_fit NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_models train NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data generate_test NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data cache_test NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_models evaluate NEW_ROOT
.venv/bin/python examples/report_radical_pilot.py NEW_ROOT
```

シンドロームのみの推論入口（oracleは拒否）：

```text
.venv/bin/python -m qp_ode_simulator.radical_predict ROOT OBSERVED.npz NEW_OUTPUT.csv --model base_SVR
```

`OBSERVED.npz`の`base`は[shot,24check,4095round]の二値配列。出力は`x_mm,y_mm`。`--model`省略時は検証選択方式。高密度方式には`dense`48check、2パッチ方式には`base`と`right`各24checkを渡す。正解座標・発生時刻・伝播条件・物理応答は渡さない。既存出力CSVの上書きは拒否する。

prepareは兄弟ディレクトリの既存`feature_selection_20260910/protocol.json`を参照。新しい独立追試にはseedも新しくし、既存テスト再利用を避ける。

- [計画](PLAN_JA.md)、[固定した選択](selection.json)、[監査](audit.json)
- [全指標](metrics.csv)、[イベント別予測](predictions.csv)、[対応区間](paired.json)、[root別指標](seed_metrics.csv)
- 自動テスト131件通過：[tests.xml](tests.xml)。[真値なし推論の動作確認](inference_smoke.json)。既定モデルは自動置換していない。
'''
    (root/'RESULT_JA.md').write_text(txt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
