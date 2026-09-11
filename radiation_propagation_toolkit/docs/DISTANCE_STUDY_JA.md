# 符号距離を変えたエピセンター推定実験

`qp_ode_simulator.distance_study` は、固定した放射線発生領域・量子ビット間隔で
rotated surface-code memory-Z の距離 3、5、7 を比較する実験モジュールです。
既存の学習済みモデルや過去のデータは上書きしません。

## 公平性のために固定するもの

- 同じ1,440物理イベントとイベント単位の学習/検証/テスト分割。
- x/y基準領域 [-3,3] mm。外側イベントもこの基準に対して定義。
- 量子ビット間隔、観測時間2.048 ms、回路ノイズ、物理源パラメータ。
- 全距離の座標の和集合上でQP/T1/T2を計算し、共通座標では同じ物理場を使う。
- 推定時は検出イベントのみ（REIは既知の固定配置も利用）。
- 全モデルの調整後、全距離の設定を凍結してから新規テストを採点。

符号を大きくすると観測範囲が広がります。局所的な量子ビット密度は増えません。
この実験のregionラベルはd3基準であり、大きい符号の内外を表しません。

## 実行

リポジトリ内の仮想環境を使う例です。生成先の親ディレクトリには既存の
`window_calibration_20260908/dataset/dataset_manifest.json` が必要です。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.distance_study generate ../localization_runs/distance_study_20260909 --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m qp_ode_simulator.distance_study fit ../localization_runs/distance_study_20260909
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m qp_ode_simulator.distance_study evaluate ../localization_runs/distance_study_20260909
.venv/bin/python examples/report_distance_study.py ../localization_runs/distance_study_20260909
```

学習720イベント、通常検証240、テスト通常240+強い楕円48。
学習候補の強い楕円144と検証の強い楕円48はモデルの学習・選択に使いません。
CNNは3seedの座標アンサンブル。Ridge/SVR/ExtraTreesは同じ検証基準で調整します。
REIの未回答には固定位置を返し、回答率と位置誤差を両方保存します。

## 実装追加

`epicenter.reference_bounds_mm = [[xmin,xmax],[ymin,ymax]]` を指定すると、
量子ビット配置とは独立に発生位置の基準領域を固定できます。
省略時の従来挙動（配置のbounding boxを使用）は維持しています。

線形Pauli fault-response cacheの対応範囲をmemory-Zのd3/d5/d7へ拡張しました。
各距離で、初期/定常/終端の単一故障応答、64通りの決定論的複合故障のStim一致、
20,000ショットの周辺確率とランダムな複数検出器のパリティ一致、乱数再現性を検証します。
これは現在のPauli/Clifford回路モデルに対する検証であり、実機再現の保証ではありません。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_fault_response.py tests/test_distance_study.py
```

## 限界

- 物理源は局所QP ODEと伝播近似。完全な空間拡散PDEではありません。
- REI原著の完全再現、イベント有無の検出、BB code、実機の検証ではありません。
- 距離ごとに別モデルを学習します。未学習距離へのゼロショット転移ではありません。
- 大きいCNNはパラメータ数も増えます。固定モデル容量の比較ではありません。
- イベントbootstrapの信頼区間に多重比較補正と学習集合変動は含めません。

## d5の学習量増加実験

`qp_ode_simulator.distance_growth` は上記実験のd5学習720イベントだけを再利用し、
1,440・2,880イベントへ段階的に増量します。旧検証・旧テストは学習に使いません。
新規3,456イベントから学習用追加分と、新規検証432・テスト432を作ります。
前回の座標和集合と固定ハードウェアを維持して、d5検出イベントだけを保存します。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.distance_growth generate ../localization_runs/distance_growth_20260909 --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m qp_ode_simulator.distance_growth train ../localization_runs/distance_growth_20260909 --size 720
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m qp_ode_simulator.distance_growth train ../localization_runs/distance_growth_20260909 --size 1440
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m qp_ode_simulator.distance_growth train ../localization_runs/distance_growth_20260909 --size 2880
.venv/bin/python -m qp_ode_simulator.distance_growth freeze ../localization_runs/distance_growth_20260909
.venv/bin/python -m qp_ode_simulator.distance_growth evaluate ../localization_runs/distance_growth_20260909
.venv/bin/python examples/report_distance_growth.py audit ../localization_runs/distance_growth_20260909
.venv/bin/python examples/report_distance_growth.py report ../localization_runs/distance_growth_20260909
```

3種類の`train`は別プロセスで並行実行できます。全学習が終わってから`freeze`し、
その後にのみ`evaluate`します。特徴量はバッチ処理してメモリ使用量を抑えています。
前回選択済みのSVR/ExtraTrees設定を固定した比較を主比較とし、同じ検証探索予算で
再調整する比較を副比較とします。CNNは各サイズ3seedの座標アンサンブルです。
