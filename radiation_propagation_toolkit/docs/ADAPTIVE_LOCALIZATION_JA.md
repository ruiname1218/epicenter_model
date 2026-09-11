# 区切り方・弱信号縮小・経験的見本照合

`qp_ode_simulator.adaptive_localization`はシンドロームだけでの位置推定について、
ニューラルネットを大きくする以外の候補を比較します。固定デバイス・同じ距離3
surface code・単一イベント存在の前提で、無イベント分類やオンライン推定ではありません。

## 特徴

1チェック当たり、区切り前と区切り後3区間の発火率から独立平常時発火率を引きます。
区切り後3区間では正の超過率を空間方向に正規化した特徴も追加します。
最後に区切り位置/観測長を付け、8チェックでは計57特徴です。
分母には固定値0.003を加え、弱い信号の比率が発散しないようにします。

区切りは、学習時刻中央値の固定値、観測された全チェック平均の上昇変化点、
真の発生時刻（診断専用）の3種類。後者を本番入力に使うことは拒否します。
固定区切りも、対象イベントの正解時刻を参照せず使えます。
変化点は観測全体を見たオフライン推定であり、オンラインの検出遅延は評価しません。

Ridgeはalpha={10,100,1000}。ExtraTreesは128 trees、min_samples_leaf={4,12}で
seed 41/42/43の予測平均。経験的見本照合は学習イベントだけの2ショットを平均し、
標準化した特徴に対してk={8,24,64}近傍の距離加重回帰を使います。
物理モデルの最尤推定ではありません。

縮小処理は、保存済み3 CNNの座標平均について、
`center + w*(prediction-center)`、`w=s²/(s²+threshold²)`を使います。
sは観測発火率と平常較正の差の正の部分。これは信頼確率ではありません。
threshold={0.0005,0.001,0.002,0.004,0.008,0.016}を検証で比較します。

学習・検証は元のイベント分割を検査して維持。CNN重みも元のデータ・分割と一致する
ことを検査します。通常25候補から検証平均位置誤差で選択し、oracle 8候補は診断のみ。
新規確認データではモデル選択を変更しません。

## 実行

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.adaptive_localization \
  --dataset ../localization_runs/window_calibration_20260908/dataset \
  --plan ../localization_runs/window_calibration_20260908/split_plan.json \
  --cnn-experiment ../localization_runs/temporal_diagnosis_20260908/exploration \
  --calibration ../localization_runs/window_calibration_20260908/quiet \
  --output ../localization_runs/adaptive_example/experiment

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.cli.dataset generate \
  --profile examples/localization_window_simulator.json \
  --stim-config examples/localization_window_stim.json \
  --sampling examples/localization_sampling.json \
  --n-events 360 --batch-size 36 --workers 8 \
  --seed 20261002 --device-seed 123 --syndrome-seed 20261003 \
  --output ../localization_runs/adaptive_example/fresh_dataset

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.adaptive_localization \
  --model ../localization_runs/adaptive_example/experiment \
  --training-dataset ../localization_runs/window_calibration_20260908/dataset \
  --dataset ../localization_runs/adaptive_example/fresh_dataset \
  --output ../localization_runs/adaptive_example/confirmation
```

ラベルなし推論:

```python
from qp_ode_simulator.adaptive_localization import predict
xy_mm = predict("experiment", "observations.npz")
```

比較用の他候補を除き、選択済み候補だけを保存する場合:

```python
from qp_ode_simulator.adaptive_localization import export_selected
export_selected("experiment", "selected_candidate")
xy_mm = predict("selected_candidate", "observations.npz")
```

回帰モデルが選ばれている場合、CNN重みを分離artifactには含めず、推論でもCNNを実行しません。

artifact checksum・回路・配置・時刻を検査します。入力には観測されたビットと固定の
回路/配置情報だけが必要で、平常較正値と学習済みモデルはartifactに含まれます。
出力は(x_mm, y_mm)のみ。今回の変化点は特徴抽出用で、発生時刻を正確に推定できたとは
主張しません。大量データに対する性能・メモリ最適化は別課題です。
