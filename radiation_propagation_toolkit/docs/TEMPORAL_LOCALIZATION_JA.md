# 時間CNNと固定評価での学習量比較

## モデルと適用範囲

単一イベントがある観測窓について、X/Z検出器イベントの**1ショット分**から連続座標 `(x, y)` を推定します。復号器は不要です。現在は固定配置・固定回路専用で、X/Zチェックの違いと位置は固定されたチェック順序に対応します。GNNや符号をまたぐ汎用モデルではありません。

時間CNNは `(ショット, チェック位置, ラウンド)` の二値入力を使います。初期・最終境界の検出器だけを除き、時間ビンで平均せず入力します。各位置で共通の3層の時間畳み込みを行い、時間方向のプーリング後に位置順に結合し、64ユニットの全結合層から2座標を出します。観測窓全体を使うため、窓終了後の推定であり逐次オンライン推定ではありません。

CNNは学習済みの畳み込みと間引きを使います。「入力が生の時系列」であることは、出力まで1ラウンド分解能を完全に保持するという意味ではありません。実装APIは [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) を使用しています。

比較用MLPは従来と同じ128/64ユニットで、16時間ビンの発火率・相対発火率・時間差分・位置別平均を入力します。今回は分割キャッシュから全学習サンプルの順序をシャッフルして訓練します。両者で同じイベント、ショット、目標座標の正規化、検証集合を使いますが、モデルと入力表現の両方が違います。差をCNN構造だけの効果とは断定できません。

真の発生時刻・強度・T1・伝播パラメータは推定入力に使いません。発生時刻で時系列を整列させることもしません。出力に座標範囲の切り詰めはなく、外部発生も回帰対象です。

## 環境準備

既存環境を使います。CNN用PyTorchは独立した追加依存です（従来の `all` には含めていません）。CPU版を使う場合：

```bash
env -u PYTHONPATH .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
env -u PYTHONPATH .venv/bin/pip install -e '.[temporal,qec,plot,dev]'
```

`env -u PYTHONPATH` はこの環境のROS用Pythonパスとの混在防止です。必要なライブラリのバージョンは各実験に保存します。

## 1. データ生成

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.cli.dataset generate \
  --profile examples/localization_training_simulator.json \
  --stim-config examples/localization_training_stim.json \
  --sampling examples/localization_sampling.json \
  --n-events 1440 --batch-size 36 --workers 8 \
  --seed 20260909 --device-seed 123 --syndrome-seed 20260910 \
  --output ../localization_runs/cnn_example/dataset
```

36区分を各40イベント生成します。中心・形状・軸比・向き・速度/拡散係数・range・強度・発生時刻・寿命などが変わり、デバイスは固定です。発生時刻は0.05〜0.65 msに広げています。観測窓は1.024 msなので、遅いイベントや外部発生の影響が十分に到達しないケースも含みます。

`--workers` は通常のプロセス並列で、イベント乱数系列や順序を変えません。並列・直列のビット一致をテストしています。`--batch-size` が処理待ちの範囲を制限し、全イベントの物理場をメモリに持ちません。生成再開は同じコマンドに `--resume` を追加します。予定イベント総数は途中で変更できないので、拡大した実験は新しい出力に生成します。

## 2. 固定分割の作成

```bash
env -u PYTHONPATH .venv/bin/python -m qp_ode_simulator.learning_plan \
  --dataset ../localization_runs/cnn_example/dataset \
  --output ../localization_runs/cnn_example/split_plan.json \
  --sizes 180 360 720 --seed 90210
```

形状・伝播則・位置領域・強度帯ごとに60/20/20の候補分割を作り、「楕円形かつ最大強度帯」を学習・検証から除きます。今回の件数では以下になります。

| 用途 | 独立イベント数 |
|---|---:|
| 学習候補（包含関係のある180/360/720件を使用） | 720 |
| 検証 | 240 |
| 既知の条件種類に属する未知イベントのテスト | 240 |
| 未学習の強度×形状組合せのテスト | 48 |
| 学習・検証候補に割り当たった除外条件（不使用） | 192 |

同じイベントの別ショットを分割しません。小さい学習集合は大きい集合の部分集合で、評価集合は全学習量・全モデルで同じです。保存したplanと生成manifestのハッシュを照合し、別データでの流用や重複を拒否します。`family_id` が重複するペア比較データは、この独立イベント用plannerでは拒否します。

未学習テストは「楕円形そのもの」でも「強いイベントそのもの」でもなく、その組合せです。通常テストと条件分布が違うので、両者の誤差差だけを未知条件による劣化量とは解釈しません。

独立イベント数をさらに増やす場合は、同じ設定・seed・コード・実行環境で、予定総数だけ増やしたデータを新しいディレクトリに生成します。その後、`--extend 元のsplit_plan.json --sizes 180 360 720 1800` のように指定してplanを拡張します。元の学習順序、検証・両テストのイベントとショットの生成条件を維持し、新しい適格イベントだけを学習候補へ追加します。通常の新規分割を作り直して、過去のテストイベントを学習へ移すことは避けてください。

拡張planも過去のテスト結果を忘れさせるものではありません。テストを見てモデルを調整した場合は、拡張前のテストを最終的な未知評価とは呼ばず、別seedの未使用評価データを追加してください。

## 3. 学習と評価

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u -m qp_ode_simulator.temporal_localization train \
  --dataset ../localization_runs/cnn_example/dataset \
  --plan ../localization_runs/cnn_example/split_plan.json \
  --output ../localization_runs/cnn_example/experiment \
  --seeds 41 42 43 --epochs 40 --patience 8 --batch-size 64 --threads 2
```

2モデル×3学習量×3seed＝18回の学習を行います。各epochの検証平均距離誤差で重みを選び、8epoch改善がなければ停止します。MLPの入力標準化、目標座標の中心は学習部分だけで決めます。最終的なモデル種類・学習量はseed平均の検証誤差で選び、代表ファイルのseedは最初の指定seedに固定します。テストを見てseedを選びません。

すべての訓練と選択の完了後に初めてテスト予測を計算します。事前指定した全学習量のテスト曲線を報告しますが、結果を見て追加調整した場合には、新しい最終評価集合が必要です。

評価は平均・中央値・90パーセンタイルの距離誤差、1/2 mm以内の割合、条件別誤差、重心法・学習座標平均との比較です。95%区間は独立イベント単位でbootstrapし、ショットやseedを独立位置として水増ししません。これは訓練済みモデルに条件付けたテスト位置の不確かさで、モデルの個々の予測に対する信頼領域ではありません。

生入力とMLP特徴量は `cache/*.npy` にディスク保存し、メモリマップからバッチごとに読みます。ラベル・予測はRAM使用量がイベント数に比例します。学習の途中再開は未対応ですが、完了した各モデルはその都度保存します。既存出力への上書きは拒否します。

## 4. 保存物と推論

- `experiment.json`：事前設定、ソースハッシュ、ライブラリバージョン。
- `split_plan.json`：再利用する固定分割。
- `validation_runs.csv`、`selected_model.json`：検証結果と選択。
- `test_predictions.csv`、`metrics.json`：各ショットの予測と条件別評価。
- `cnn_n*_seed*/weights.pt` と `model.json`：CNN重み、回路・時間設定、正規化。
- `mlp_n*_seed*/model.joblib`：従来の推論関数で読み込めるMLP。

```bash
env -u PYTHONPATH .venv/bin/python -m qp_ode_simulator.temporal_localization predict \
  --model ../localization_runs/cnn_example/experiment/cnn_n720_seed41 \
  --input ../localization_runs/cnn_example/dataset/events_00000000_00000036.npz \
  --output ../localization_runs/cnn_example/predictions.csv

env -u PYTHONPATH .venv/bin/python examples/plot_learning_curve.py \
  --experiment ../localization_runs/cnn_example/experiment \
  --output ../localization_runs/cnn_example/learning_curve.png
```

推論には正解CSVは不要です。回路ID・物理配置・検出器順序・観測時間が異なれば拒否します。モデルファイルは信頼できる生成元のものだけを読み込んでください。

## 現時点で実装していないもの

GNN、混合分布出力・予測信頼領域、イベント検出、複数源の分離、別デバイスへの汎化、BB code対応は含みません。まず固定配置の時間CNNが重心法とMLPを上回るかを確かめます。強いイベントにおけるPauli近似や物理パラメータの実機校正も別課題として残ります。
