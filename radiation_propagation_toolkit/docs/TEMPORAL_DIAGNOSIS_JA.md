# 時間モデル・発生時刻診断の比較

`qp_ode_simulator.temporal_diagnosis` は、固定配置のsurface codeにおける
単一放射線イベントの位置推定を比較します。通常モデルに渡す入力は
二値detector eventのみです。物理配置は固定チェック順序に対応します。
GNN、無イベント分類、複数源、オンライン検出は対象外です。

## 条件

| arm | 入力 | 出力 |
|---|---|---|
| original | シンドローム | x, y |
| original_joint | シンドローム | x, y, 発生時刻 |
| multiscale | シンドローム | x, y |
| multiscale_joint | シンドローム | x, y, 発生時刻 |
| oracle_onset | シンドローム＋真の発生時刻 | x, y（診断専用） |
| oracle_parameters | シンドローム＋真の発生時刻・形状・伝播条件等 | x, y（診断専用） |

originalは従来のTemporalCNNと同じ構造・初期化で、等価性をテストしています。
multiscaleは短時間branch（stride 4、16区間）と長時間branch（平均化＋拡張畳み込み、
8区間）を使います。両者とも1チェック当たり128特徴と全期間の平均発火率を
位置回帰headに渡します。パラメータ数は近いですが完全に等しくありません。

jointは位置headとは別に時刻headを持ちます。損失は、正規化した座標のMSEと
`0.1 × 正規化した発生時刻のMSE`の和です。座標中心、時刻平均・標準偏差は
学習データだけで計算します。epoch選択は従来どおり検証の平均位置誤差です。
時刻予測は連続値で、範囲外の値も隠さず評価します。誤差をラウンド換算します。
物理的な発生時刻と最初に観測できた異常の時刻は同一ではありません。

oracle_parametersは軸比、角度、速度、拡散係数、初期/最大lambda、最大距離、
front幅、source lifetime、対数QP生成強度も使用します。無関係な伝播係数のNaNは
0に置換し、学習集合で標準化します。位置・位置領域・真のT1・観測後の達成強度は
入力しません。全ての物理乱数を与えるoracleでも、理論的最適推定器でもありません。
oracleが改善しなくても情報不足の証明にはなりません。

## REI比較の範囲

出典: https://arxiv.org/html/2506.16834v1 Algorithm 1。

同じ観測窓の全interior detector eventを履歴として、窓末端で中心を求めます。
各チェックを、そのチェックが置かれた物理位置に対応付けます。境界測定は含めません。
擬似コードにあるalpha、3点以上の条件、頻度min-max正規化、最近傍距離の空間条件、
二乗重みを実装しています。本文と擬似コードにはしきい値の不一致があるため、
v1擬似コードの「平均最近傍距離の2倍」を使用します。

`rei_v1_adapted`は検出なしをNaNにし、coverageと返答した例の位置誤差を報告します。
`rei_v1_fallback`は検出なしのときだけ学習座標平均に戻す、こちらで追加した比較条件です。
著者の放射線モデル、元のシンドローム定義、動的FIFO検出を全面再現した結果ではなく、
Q3DE/Chadwickの時空間窓法を実装したものでもありません。
無イベントでの偽陽性やオンラインの検出性能を、この位置比較から主張できません。

## 実行

リポジトリ直下で、新しい出力ディレクトリを指定します。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.temporal_diagnosis \
  --dataset ../localization_runs/window_calibration_20260908/dataset \
  --plan ../localization_runs/window_calibration_20260908/split_plan.json \
  --output ../localization_runs/temporal_diagnosis_example/exploration
```

6条件×3seed、最大40epoch、patience 8、batch 64、AdamW、学習率1e-3、weight decay
1e-3です。全モデルを学習し終え、通常4条件から検証seed平均の位置誤差で選択を
凍結した後、既存テストを採点します。既存テストを再利用した結果は探索扱いです。
診断モデルを通常モデルの選択に含めません。

新しいイベントseedと測定seedでデータを生成した後、再学習せずに評価します。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.cli.dataset generate \
  --profile examples/localization_window_simulator.json \
  --stim-config examples/localization_window_stim.json \
  --sampling examples/localization_sampling.json \
  --n-events 720 --batch-size 36 --workers 8 \
  --seed 20260930 --device-seed 123 --syndrome-seed 20260931 \
  --output ../localization_runs/temporal_diagnosis_example/fresh_dataset

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.temporal_diagnosis \
  --frozen-experiment ../localization_runs/temporal_diagnosis_example/exploration \
  --training-dataset ../localization_runs/window_calibration_20260908/dataset \
  --dataset ../localization_runs/temporal_diagnosis_example/fresh_dataset \
  --output ../localization_runs/temporal_diagnosis_example/confirmation
```

デバイスと回路を検査し、event_uid・生成seed・測定seedの重複を拒否します。
新しいイベントは全件テスト専用です。学習時の保留条件でID/OODに分類します。
確認結果からモデルを選び直しません。IDとOODは強度構成が異なるため、絶対誤差の差を
汎化劣化量とは呼びません。平均は単一ショット・単一モデルの誤差をseed平均した値で、
アンサンブルではありません。対応付き区間はイベント単位bootstrapで、
多重比較補正済みの確証試験や物理的限界の証明ではありません。

## 推論・保存物

通常モデルのラベルなし推論:

```python
from qp_ode_simulator.temporal_diagnosis import predict_model
predictions = predict_model("experiment/multiscale_joint_seed41", "observations.npz")
# 列: x_mm, y_mm, event_onset_ms（jointでないモデルは最初の2列）
```

重みchecksum、回路・配置・時刻を検査します。oracleモデルはこの入口では拒否します。
`model.json`、`weights.pt`、`validation_runs.csv`、`selected_model.json`、
`test_predictions.csv`、`metrics.json`、`paired_comparisons.json`に設定と結果を保存します。
元の既定モデル・シミュレータ設定は変更しません。

## 左右・上下だけを当てる追加診断

`qp_ode_simulator.position_information`は、真の強度帯別に、位置がデバイス中心の
左右/上下どちらにあるかをロジスティック回帰で分類します。通常の16区間の
シンドローム特徴と、真の発生時刻から計算する前後平均・差分の2種類を比較します。
特徴の種類・次元も異なるため、両者の差を時刻情報だけの効果とは解釈しません。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.position_information \
  --dataset ../localization_runs/window_calibration_20260908/dataset \
  --plan ../localization_runs/window_calibration_20260908/split_plan.json \
  --output ../localization_runs/temporal_diagnosis_example/side_diagnostic

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.position_information \
  --model ../localization_runs/temporal_diagnosis_example/side_diagnostic \
  --training-dataset ../localization_runs/window_calibration_20260908/dataset \
  --dataset ../localization_runs/temporal_diagnosis_example/fresh_dataset \
  --output ../localization_runs/temporal_diagnosis_example/side_confirmation
```

学習/検証分割は位置モデルと同じで、正則化Cは事前の4候補から検証AUCで選びます。
新しいテストでのAUC、balanced accuracy、イベント単位のbootstrap区間を保存します。
AUC 0.5がランダム順位付けの目安、1が完全な順位付けです。
低強度で0.5付近でも「この特徴・線形分類器では拾えない」という結果であり、
あらゆるモデルに対する情報限界の証明ではありません。真の強度帯でモデルを
振り分けるため、この診断全体も本番推論としては使用しません。
