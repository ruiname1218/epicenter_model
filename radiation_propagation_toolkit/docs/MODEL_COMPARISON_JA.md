# 同じ特徴量でのモデル比較

`qp_ode_simulator.model_comparison`は、adaptive_localization実験の57特徴を固定し、
座標への変換モデルだけを比較します。回路・学習/検証分割・独立平常較正・区切りは
親実験から取得して検査し、再調整しません。推論時に真の時刻・強度は入力しません。

| モデル | 探索範囲 |
|---|---|
| Ridge | alpha=10,100,1000 |
| MLP | (32,) alpha=1 / (64,32) alpha=0.1 / (64,32) alpha=10 |
| Random Forest | 128 trees、min_samples_leaf=4,12 |
| ExtraTrees | 128 trees、min_samples_leaf=4,12 |
| HistGradientBoosting | 150 iterations、max_leaf_nodes=8,16、lr=.05、L2=10 |
| RBF Kernel Ridge | alpha=1,10 × gamma=.001,.01 |
| RBF SVR | C=1,10 × gamma=.001,.01、epsilon=.1 mm |

計20設定、7モデル群。MLP/Random Forest/ExtraTreesは3seedの予測座標平均です。
決定的手法は1モデル。探索・容量・計算量を完全に一致させた比較ではありません。
MLPは最大150epoch、patience15、外部検証の平均位置誤差でepochを選択します。
scikit-learn内蔵のランダムな内部検証分割は使わず、イベントをまたぐ漏洩を避けます。
標準化は学習集合だけでfit。各familyの代表と全体選択は新しいテスト前に固定します。

## 実行

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.model_comparison \
  --dataset ../localization_runs/window_calibration_20260908/dataset \
  --parent-experiment ../localization_runs/adaptive_features_20260908/experiment \
  --output ../localization_runs/model_comparison_example/experiment

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.cli.dataset generate \
  --profile examples/localization_window_simulator.json \
  --stim-config examples/localization_window_stim.json \
  --sampling examples/localization_sampling.json \
  --n-events 360 --batch-size 36 --workers 8 \
  --seed 20261010 --device-seed 123 --syndrome-seed 20261011 \
  --output ../localization_runs/model_comparison_example/fresh_dataset

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.adaptive_localization \
  --model ../localization_runs/model_comparison_example/experiment \
  --training-dataset ../localization_runs/window_calibration_20260908/dataset \
  --dataset ../localization_runs/model_comparison_example/fresh_dataset \
  --output ../localization_runs/model_comparison_example/confirmation
```

学習出力はadaptive_localizationのartifact形式を再利用するため、評価とラベルなし
推論は共通です。CNNは別入力を用いる参考基準として残します。

```python
from qp_ode_simulator.adaptive_localization import export_selected, predict
export_selected("experiment", "selected_candidate")
xy_mm = predict("selected_candidate", "observations.npz")
```

`grid.json`、`validation.csv`、`mlp_histories.json`、`model.json`、`models.joblib`を保存します。
MLPも別プロセスから読み込める形式で保存し、未使用テストを変更しても学習結果が
変わらないことをテストしています。近傍法や物理最尤推定、GNN、Transformerは今回の範囲外です。
