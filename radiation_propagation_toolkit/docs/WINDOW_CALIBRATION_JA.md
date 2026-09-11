# 観測窓と平常時補正の比較

時間CNNの位置推定について、次の2要因を同じイベント・ショットで比較します。

| 観測窓 | 従来の二値入力 | 平常時基準で補正 |
|---|---|---|
| 1.024 ms | 基準条件 | 補正のみ変更 |
| 2.048 ms | 観測窓のみ変更 | 両方変更 |

CNNは従来と同じ68,530パラメータです。GNNや発生ラウンドの同時推定は追加していません。短い入力は長い入力と同一の観測ビットの先頭部分で、初期・最終境界の検出器を除きます。短い窓用に架空の終了測定を追加しません。

## 平常時補正とは

別に取得した放射線なしの256ショットから、チェックごとの検出器発火率 `p` を推定します。学習・評価イベントの正解時刻、T1や強度は使いません。1チェック当たり256×2,047＝524,032個の二値観測を平均します。時系列相関はあり、この数を独立試行数とした信頼区間は主張しません。

補正ありでは最初の畳み込みへ `(bit-p)/sqrt(p*(1-p))` を渡します。補正なしは従来どおり `10*bit` です。これは平常時との差を取ることと分散スケールの正規化を合わせた方法で、**差を引くだけの効果を単独に検証する実験ではありません**。

`p` は全イベントで同じ外部校正値です。推論対象の波形を使って更新せず、真のイベント前区間を切り出して推定することもしません。同じデバイス・回路設定での安定した校正が取得できるという前提です。実機では別の平常運転データから測り、ドリフト時には更新と再検証が必要です。

補正は新たなイベント情報を生むわけではなく、信号が弱すぎる場合の識別性を保証しません。また、CNNの最終時間プーリング数は固定のため、長い窓では最終表現の時間分解能が粗くなります。評価するのは「この固定CNNの運用で窓を延ばすと改善するか」であり、観測情報量の理論限界ではありません。

## 実行方法

リポジトリ直下、既存の仮想環境を使います。出力先は新しいディレクトリにしてください。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.cli.dataset generate \
  --profile examples/localization_window_simulator.json \
  --stim-config examples/localization_window_stim.json \
  --sampling examples/localization_sampling.json \
  --n-events 720 --batch-size 36 --workers 8 \
  --seed 20260920 --device-seed 123 --syndrome-seed 20260921 \
  --output ../localization_runs/window_example/dataset

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.window_calibration calibrate \
  --dataset ../localization_runs/window_example/dataset \
  --output ../localization_runs/window_example/quiet --shots 256

env -u PYTHONPATH .venv/bin/python -m qp_ode_simulator.learning_plan \
  --dataset ../localization_runs/window_example/dataset \
  --output ../localization_runs/window_example/split_plan.json \
  --sizes 360 --seed 90211

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.window_calibration train \
  --dataset ../localization_runs/window_example/dataset \
  --plan ../localization_runs/window_example/split_plan.json \
  --calibration ../localization_runs/window_example/quiet \
  --output ../localization_runs/window_example/experiment
```

校正は生成manifestの作成後なら本データ生成と同時に行えます。学習はデータセット完成後のみ可能です。生成再開は `--resume`、学習途中の再開は未対応です。

## 評価と保存物

720イベントの場合、学習360・検証120・通常テスト120・未学習の強い楕円形テスト24・不使用96です。新しいseedのため以前のCNN比較で評価したイベントを再利用しません。4条件それぞれを3seedで学習します。同じイベント・ショット・学習分割・検証分割で比較します。

epochと条件の選択には検証データのみを使い、全学習完了後に新しいテストを評価します。代表モデルのseedは最初に指定した41で固定します。条件別の誤差に加え、「長い窓−短い窓」「補正あり−なし」の**対応付き誤差差**をイベント単位のbootstrapで集計します。負の差が改善です。複数比較の区間は記述的なもので、多重性補正済みの確証試験ではありません。

通常・未学習テストの強度分布は異なります。絶対誤差を直接比べて未知条件への汎化劣化とは呼びません。前回より学習・テスト件数も少ないため、前回の3.18 mmとの直接比較ではなく、今回の4条件間で効果を判断します。

- `quiet/quiet_syndromes.npz`、`quiet/calibration.json`：観測ビットと校正値・生成条件。
- `experiment/experiment.json`、`split_plan.json`：設定・ソース/データハッシュ・固定分割。
- `validation_runs.csv`、`selected_model.json`：検証結果と事前ルールで選んだモデル。
- `test_predictions.csv`、`metrics.json`：各ショットの予測、条件別誤差、対応付き区間。
- `cnn_r*_*/weights.pt`、`model.json`：各モデルと推論に必要な校正値。

## 推論

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.window_calibration predict \
  --model ../localization_runs/window_example/experiment/cnn_r1024_calibrated_seed41 \
  --input ../localization_runs/window_example/dataset/events_00000000_00000036.npz \
  --output ../localization_runs/window_example/predictions.csv
```

正解ラベルやT1は不要です。校正値はモデルに保存してあるため、イベントごとに校正し直す必要もありません。短いモデルは入力の先頭1,024ラウンドだけを使い、将来のビットは参照しません。完全なNPZだけでなく、同じ親回路IDを保持した観測プレフィックスも受け付けます。別の回路・配置・時刻設定は拒否します。

これは同一デバイス内の合成データ実験です。実機への適合、強いT1 collapseにおけるPauli近似、予測の信頼領域、無イベント検出、GNNは引き続き別課題です。
