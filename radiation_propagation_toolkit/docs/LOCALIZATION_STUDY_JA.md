# 大量生成前の比較実験

目的は、シンドロームからの単一発生中心推定について、観測時間・時間分解能・強度・到達範囲の影響を切り分けることです。実機精度の検証や最適な物理モデルの決定ではありません。

## 再現方法

リポジトリ直下で実行します。既存の出力がないディレクトリを指定してください。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m qp_ode_simulator.localization_study \
  --profile examples/localization_simulator.json \
  --stim-config examples/localization_stim.json \
  --study-config examples/localization_study.json \
  --workers 4 --output ../localization_runs/study_example
```

同じ設定・コードでの生成再開には `--resume`、保存したデータの再評価には `--evaluate-only` を追加します。生成は1条件ずつファイルに確定し、再開時にハッシュを確認します。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python examples/check_time_resolution.py \
  --study-dir ../localization_runs/study_example \
  --output ../localization_runs/study_example/time_resolution.csv
```

## 実験の設計

- 距離3の rotated surface-code memory-Z 回路。X/Z両方の検出器の生データを使います。放射線による変化の対象はデータ・補助量子ビット両方です。
- 48個のイベントファミリーを生成し、同じ位置・形状・向き・伝播条件・デバイスに対して、強度3水準とrange倍率2水準を比較します。合計288条件、各2ショットです。
- 12ファミリーごとに円形/楕円形、ballistic/diffusive、内部/境界/外部の12組合せを網羅します。学習24・検証12・テスト12ファミリーに分けます。別強度の同じイベントも、別ショットも、分割をまたぎません。
- 最長2.048 msの時系列から先頭0.512/1.024/2.048 msを取り出します。仮の終了測定を作らず、初期・最終境界の検出器を特徴量から除外します。
- 時間ビンの**幅**を64/16 µsで固定して比較します。観測窓を長くする際、ビン数固定によって時間分解能が粗くなる問題を避けます。
- 通常の評価に加え、「最強度かつ広範囲」の組合せを学習・検証から除き、未学習の組合せへの汎化を調べます。これは未学習デバイスや未学習伝播則への汎化とは異なります。
- ExtraTreesの葉サイズは検証データだけで選びます。推定入力は検出器時系列由来の特徴量だけで、真の発生時刻やT1、強度、rangeは渡しません。

`variants.csv` に条件・最小T1・T1が30 µs未満のラウンド/量子ビット割合・計算資源を記録します。30 µsは診断用の目安で、物理的に有効/無効を決める境界ではありません。`study_predictions.csv` は各ショットの予測、`validation_selection.csv` は検証誤差、`study_summary.json` は条件別のテスト誤差です。

95%区間はファミリー単位のbootstrapです。独立したテスト位置は12個しかなく、576ショットを576個の独立イベントとして扱ってはいけません。条件別の順位は探索的な結果です。最終設定は別の十分大きい評価集合で確認してください。

## 調査に伴う修正

`radiation_channel.targets=data` が補助量子ビットの平常時T1/T2ノイズまで消していたため、**放射線による増分だけ**を対象外にするよう修正しました。固定復号器にも平常時ノイズを残します。APIからは基準T1/T2を渡し、低レベル関数の直接利用では基準値またはイベント前のサンプルを必要とします。参照回路との測定・検出器・復号結果の一致をテストしています。

既定の `targets=all` は変えていません。従来の `targets=data` データで学習していた場合は、データ再生成と再学習が必要です。

大量生成用datasetでは、再開時に確認するハッシュにAPI・配置・ノイズ変換の依存コードを追加し、Python/NumPy/SciPy/Stimのバージョンも固定します。また、次のイベント生成前に前イベントの大きな物理配列への参照を解放するようにしました。

## この比較では解決しない点

Pauli近似による振幅減衰の非対称性の欠落は残ります。強いT1 collapseを実機相当とみなすには別途校正・検証が必要です。QP生成率はエネルギーや線量ではありません。また、外部発生や弱いイベントではシンドロームに位置を識別する情報が少なく、モデルの複雑化だけでは解決しません。

現在は「放射線イベントが1個ある」とした条件付き位置推定です。無イベントの検出、複数源の分離、別デバイスへの適用、BB code対応を実装済みとはしていません。
