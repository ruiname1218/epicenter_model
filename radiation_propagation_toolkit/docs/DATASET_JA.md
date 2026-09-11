# 大量生成・分割学習

`qp-ode-dataset` は、小分け保存、途中再開、イベントID、機器の固定、条件の層別サンプリングを備えたデータ生成コマンドです。`qp-ode-localize train` は分割形式を検出すると、ファイルを順に読み込むMLP学習へ切り替わります。

## 現在の符号

同梱例 `examples/localization_stim.json` は `surface_code:rotated_memory_z`、distance-3です。データ量子ビット9個、補助量子ビット8個、X/Z両方のチェックを使い、論理Zのメモリを評価する回路です。`memory_z`はZチェックだけを測るという意味ではありません。

- 1,024ラウンド、1ラウンド1 µs、4ショット/イベント。
- 放射線によるT1/T2ノイズはdataとancillaの両方へ挿入。
- 推定器の入力は二値の検出器イベント。生の補助量子ビット測定値や復号結果とは異なります。
- 学習器は固定回路・固定物理配置用。符号や時間設定を変更したら再学習が必要です。

BB（bivariate bicycle）codeの回路生成は未実装です。`syndrome.py`の`custom_css`は任意のX/Zチェックを扱う代数的な代理モデルですが、BB符号のゲートスケジュール・補助量子ビット測定回路を実行する機能ではありません。回路として対応するには、具体的な符号、物理座標、チェック測定のゲート列、時間設定、検出器の対応を追加する必要があります。

分割データには符号名、distance、回路ハッシュ、物理配置を記録します。学習・推論は回路IDと座標・時間設定を照合し、異なるものを混ぜません。符号名だけを変更してBB codeのシンドロームにすることはできません。

## 生成と再開

```bash
source .venv/bin/activate
python -m pip install -e '.[all]'

qp-ode-dataset generate \
  --profile examples/localization_simulator.json \
  --stim-config examples/localization_stim.json \
  --sampling examples/localization_sampling.json \
  --n-events 36 --batch-size 6 \
  --seed 42 --device-seed 123 --syndrome-seed 456 \
  --output ../localization_runs/shards_example/dataset
```

36イベントは動作と分布確認のための小規模例です。十分な学習量・推定精度を保証する件数ではありません。独立イベントを増やすには`--n-events`を増やします。`shots_per_event`だけを増やしてもイベント条件の多様性は増えません。

中断した場合は、同じコマンドに`--resume`を付けます。保存済みの分割ファイルをハッシュ検証し、未完了のイベントから再開します。完了済みなら再生成しません。

- 1分割を完了するごとにNPZ/CSVとmanifestを保存します。
- 中断時に未保存の分割だけは再計算します。保存済み分割の破損はエラーとし、自動で上書きしません。
- 途中再開では設定・seed・予定イベント数・生成コードが一致する必要があります。
- `--batch-size`は変更可能です。イベントID・元イベント・シンドロームは変わりません。
- 同一フォルダへの生成の二重起動はロックで拒否します。
- 全イベントの重い物理配列を保持せず、イベントごとに計算して、観測ビットだけを分割単位でまとめます。
- 既定は直列生成です。`--workers 8` などを指定すると独立イベントを並列生成します。並列数・分割サイズを変えてもイベント乱数系列と保存順序は同じです。

## 乱数と機器

`--seed`はイベント群、`--device-seed`は固定機器の校正と感度分布、`--syndrome-seed`は測定乱数を制御します。各イベントのseedは全体seedとイベントIDから決まり、分割サイズに依存しません。生成コマンドのseed指定を使用し、profile内の`seed`と`n_events`は使用しません。

機器seedを同じにすると、同じ物理配置・機器設定に対するT1/T2・周波数の校正マップが分割をまたいで維持されます。発生位置・形状・イベント固有の感度変動などはイベントごとに変わります。元イベントのIDは、生成条件に基づく`source_id`とイベント番号から作ります。

通常の`run_simulation`にも任意の`device_seed`を追加しました。省略した従来設定では、以前の乱数系列を維持します。

## パラメータ分布

`examples/localization_sampling.json`では、以下の直積36通りを1巡ごとにシャッフルして生成します。

- 円形／楕円形。
- ballistic／diffusive。
- 内部／境界／外部。
- QP生成率の3帯域。

各帯域のQP生成率は対数一様分布です。36件ごとに各組合せが同数となり、端数は次の巡回の一部になります。controlが指定されている場合は、各イベントに独立に指定確率で割り当てるため、非controlだけの件数は厳密な均等になりません。

このsampling設定はprofileの形状選択・伝播則選択・領域確率・QP生成率を上書きします。速度や拡散係数の値、軸比、角度、range、領域内の位置、発生時刻などはprofileの範囲からサンプルします。強度帯域の境界は探索用の指定であり、実機での発生頻度や最適な学習分布を主張するものではありません。

内部・境界・外部は量子ビット座標の外接矩形に対する分類です。境界は現在も境界線上で、帯状の境界領域への変更は行っていません。

## 保存形式

```text
dataset_manifest.json
events_00000000_00000006.npz
events_00000000_00000006.csv
events_00000006_00000012.npz
events_00000006_00000012.csv
...
```

NPZには生の二値検出器データ、イベントID、controlフラグ、回路・物理座標・時間設定を保存します。CSVには正解座標とイベントパラメータ、強度帯域、seedを保存します。全時刻のQP密度・T1/T2・ゲートノイズ確率は保存せず、最小T1などの要約のみ残します。詳細な物理場を調べるイベントは、記録した条件で通常のシミュレータから再生成できます。

## 観測時間と強度の診断

```bash
qp-ode-dataset audit \
  --dataset ../localization_runs/shards_example/dataset \
  --windows-ms 0.25 0.5 1.0 \
  --output ../localization_runs/shards_example/audit
```

同じ観測時系列の先頭部分を使って比較します。検出器発火率、イベント前後の発火率、空間的な発火率のばらつきをCSVに出し、強度帯域と観測時間ごとに集計します。初期・最終境界の検出器は除外します。イベント前後の区分に使う真の発生時刻は診断専用で、推定モデルには渡しません。

`full_field_minimum_t1_us`は生成した物理場全体の最小値で、各短い観測窓内の最小値ではありません。イベント前または後の測定がない窓では対応する発火率を空欄にします。

長い窓を比較する場合は、Stimのラウンド数とシミュレータの終了時刻を両方延長した別データセットを生成します。診断値だけから「長いほど位置精度がよい」とは判断できません。最適な観測時間・強度範囲の決定には、位置推定の比較実験が別途必要です。

## 分割ファイルからの学習・推論

```bash
qp-ode-localize train \
  --dataset ../localization_runs/shards_example/dataset \
  --output ../localization_runs/shards_example/model \
  --bins 16 --epochs 20 --batch-size 256

qp-ode-localize predict \
  --model ../localization_runs/shards_example/model/model.joblib \
  --input ../localization_runs/shards_example/dataset \
  --output ../localization_runs/shards_example/predictions.csv
```

分割形式では128/64ユニットのMLPを逐次学習します。従来の単一NPZフォルダには従来のExtraTreesを使用します。`--trees`と`--jobs`はExtraTrees用、`--epochs`と`--batch-size`はMLP用です。

元イベント単位で60/20/20に分割します。controlを位置学習から除外し、複数発生源を拒否します。入力の標準化はtrainの特徴量だけで決め、validationで最良epochを選択し、最後にtestを評価します。ショットを別々の集合へ分けることはありません。

生のシンドロームは1分割ずつ読み込みます。特徴量は一時フォルダに分割キャッシュし、学習を繰り返した後に削除します。特徴量を全件RAMに保持しませんが、ラベル・予測・評価値はイベント数に比例するメモリを使用します。一時ディスクには特徴量キャッシュ分の空き容量が必要です。

推論はNPZ単体でも可能です。その場合は正解CSVが不要です。フォルダ全体の推論ではmanifestとラベルとの整合性も検査しますが、座標の真値は予測の特徴量には使いません。回路ID、検出器順序、物理配置、観測時間が学習時と異なる入力は拒否します。

保存済みファイルを読み直して学習できますが、MLPの学習途中のcheckpoint再開は未実装です。大量生成の再開機能と区別してください。

生成の再開では、設定・生成コードと依存コードのハッシュ・Python/NumPy/SciPy/Stimのバージョンの一致も必要です。更新前のデータは読み込めますが、途中から異なるコードや環境のデータを追加することは拒否します。

観測時間・時間ビン幅・強度・rangeを位置推定誤差で比較する方法は [比較実験ガイド](LOCALIZATION_STUDY_JA.md) を参照してください。

固定分割・未学習条件の評価・時間CNNの学習量比較は [時間CNNガイド](TEMPORAL_LOCALIZATION_JA.md) を参照してください。
