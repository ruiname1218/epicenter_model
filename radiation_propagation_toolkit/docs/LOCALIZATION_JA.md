# シンドロームからエピセンター座標を推定する

`qp-ode-localize` は、Stim の二値検出器イベントから連続座標 `(x_mm, y_mm)` を出す教師あり回帰モデルです。まず動作と評価を確認できるCPU向けのベースラインとして実装しています。

対象は、単一の放射線イベントを含む切り出し済みの観測窓。同じ回路・配置・時間設定に対し、**1ショットごと**に座標を推定します。イベント検出、複数発生源、推定の信頼領域、実機データへの適応は未実装です。入力にイベントがなくても座標は出るため、未知の連続ストリームにそのまま適用しないでください。

## 最短の実行手順

リポジトリのルートで実行します。仮想環境作成済みなら最初の行は不要です。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'

qp-ode-stim \
  --profile examples/localization_simulator.json \
  --stim-config examples/localization_stim.json \
  --output ../localization_runs/example/dataset

qp-ode-localize train \
  --dataset ../localization_runs/example/dataset \
  --output ../localization_runs/example/model

qp-ode-localize predict \
  --model ../localization_runs/example/model/model.joblib \
  --input ../localization_runs/example/dataset/stim_syndrome_events.npz \
  --output ../localization_runs/example/predictions.csv
```

最後のコマンドはデータ全件への推論方法の例です。未学習イベントの性能は `model/metrics.json` と `model/test_predictions.csv` で確認してください。モデル保存先には新規または空のフォルダ、推論CSVには新規ファイルを指定します。

この環境でROSのpytestプラグインが干渉する場合のテストコマンド:

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

## 入出力

学習は `qp-ode-stim` の保存先を指定し、次を読み込みます。

- `stim_syndrome_events.npz`: 観測ビットと回路の座標・時間情報。
- `true_parameters.csv`: 学習と評価のための発生位置、control、発生源数。

CSVの `event` はNPZのイベント軸と同順の `0..N-1` とし、行の並べ替えは拒否します。controlは位置損失・分割・評価から除外し、非controlの複数発生源は拒否します。少なくとも15件の非controlイベントが必要ですが、これは実行上の下限であり、十分な学習量を意味しません。

推論時には正解CSVは不要です。NPZに必要な配列は以下です。

| 配列 | 役割 |
|---|---|
| `detector_events` | `(event, shot, detector)` の0/1。1次元の単一観測、または `(sample, detector)` も受け付ける |
| `detector_coords` | Stim検出器順に並んだ `(grid_x, grid_y, round)` |
| `circuit_qubit_ids` | 回路量子ビットID |
| `circuit_grid_coords` | 回路量子ビットの格子座標 |
| `circuit_physical_coords_mm` | 対応する実座標、mm |
| `circuit_qubit_roles` | data/X-ancilla/Z-ancillaの種別コード |
| `round_start_time_ms`, `round_time_ms` | 観測時間設定 |
| `gate_slice_duration_ms` | ゲート区間の長さ |

これらのメタデータは、観測系について既知の情報です。真のイベント発生時刻、T1/T2、QP密度、注入Pauli確率、形状、強度、伝播則は特徴量として読みません。NPZ内にあっても無視します。生のancilla測定記録を使う場合は、対応するStim回路の `compile_m2d_converter()` などで検出器イベントに変換してから渡します。

出力CSV:

```text
event,shot,x_mm,y_mm
```

`event` は入力内での0始まりの行番号です。座標は量子ビットへの丸めをしない連続値です。既存シミュレータの命名との対応は **`x_mm = epicenter_row`, `y_mm = epicenter_col`** です。

Pythonからも使えます。

```python
from qp_ode_simulator.localization import predict_epicenters

prediction = predict_epicenters("model/model.joblib", "unlabelled_syndromes.npz")
print(prediction[["x_mm", "y_mm"]])
```

## モデルと分割

初期・最終境界の検出器を除き、各検出器位置の二値時系列を既定16区間に分割して発火率を計算します。発火率、各時刻での空間的な相対発火率、時間差分、全期間の位置別平均をExtraTreesに入力します。distance-3の同梱例は8検出器位置、384特徴量です。ショット間平均は取りません。

元イベントを乱数seed付きで60%/20%/20%に分割してから、各イベントのショットを展開します。同じ潜在場から作ったショットはすべて同じ集合に残ります。2種類の葉サイズをvalidationで比較し、選択後にtestを評価します。testを用いたモデル選択・学習は行いません。

この分割は同一機器・同一生成分布の未学習イベントに対する評価です。別機器や未知のパラメータ範囲への汎化評価ではありません。

比較対象は、学習イベントの平均発生位置を常に返すモデルと、最初の時間区間からの発火率増加で重み付けした検出器重心です。後者は観測のみを用いる簡易法で、最初の区間が常にイベント前であるという保証は置いていません。

## 保存される評価

- `model.joblib`: 回帰器、特徴抽出仕様、座標・時間設定。
- `metrics.json`: 平均・中央値・90/95パーセンタイルの位置誤差、1/2 mm以内の割合、比較対象、形状・伝播則・領域別集計、バージョン、入力ハッシュ。
- `splits.json`: train/validation/testの元イベント番号。
- `validation_predictions.csv`, `test_predictions.csv`: 各ショットの正解、推定、距離誤差。軸比・伝播係数・範囲・生成率・最小T1なども、推定後の誤差分析専用として添付。

独立な標本数はイベント数です。例えば64イベント×4ショットの256予測を256独立イベントとは扱いません。モデルファイルはjoblib/pickle形式のため信頼できる生成元のものだけを読み込んでください。保存時と同じscikit-learn版を要求します。

図を作るには:

```bash
python examples/plot_localization.py \
  --dataset ../localization_runs/example/dataset \
  --model-dir ../localization_runs/example/model \
  --output ../localization_runs/example/evaluation.png
```

## パラメータを変更するとき

`examples/localization_simulator.json` はQP-ODE既定設定を継承します。形状、速度・拡散係数、range、位置、QP生成率はイベントごとに変わります。既定の例は320イベント、内部70%・境界20%・外部10%、発生時刻0.05〜0.15 msです。ハードウェアは1回の生成内で固定します。

内部・境界・外部は量子ビット座標の外接矩形に対する分類で、実際のチップ外形を表すものではありません。ExtraTreesの出力は学習データの位置範囲を越えた外挿には向かないため、推定対象にしたい外部領域も学習分布に含めてください。

`examples/localization_stim.json` はdistance-3、1,024ラウンド、1 µs/ラウンド、4ショット/イベント。全観測窓は1.024 msなので、遅い拡散や遠方の発生源は波面が十分届かない場合も含みます。検出しやすいイベントだけを正解情報で選別する処理は入れていません。

`--n-events` と `--seed` で生成件数と乱数を変更できます。設定範囲を変えたら新しい出力先に生成して学習し直します。回路・配置・時間が変わった入力はモデルが拒否します。回路の意味や測定ノイズを変えた場合も、配列形状が同じだから適用可能とは限りません。

従来の`qp-ode-stim`は物理診断・回路ノイズ配列も保持するので、メモリと生成時間が増えます。大量生成には、新しい`qp-ode-dataset`の分割保存・途中再開と、MLPによる分割学習を利用できます。[DATASET_JA.md](DATASET_JA.md)を参照してください。

## 次の改善を判断する基準

まずtestに対する単純重心・定数モデルとの差、ballistic/diffusive別、内外別の誤差を確認します。弱い・遅いイベントで悪化するなら観測時間とイベント検出、楕円形で悪化するなら時空間表現、全条件で悪化するなら生成件数とモデル容量を調べます。

強いT1低下ではStimのPauli近似誤差も残ります。このモデルの評価は同じ合成器内の位置推定であり、実機のradiation位置推定精度ではありません。伝播場単独のベンチマークとも区別します。

実装参照: [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)、[グループ単位の分割](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)。
