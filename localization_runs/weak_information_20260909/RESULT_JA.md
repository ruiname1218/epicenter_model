# d5弱イベント: 観測・時刻・物理情報の切り分け

## 新規テスト576イベントの結果

全て弱イベント。イベント存在・弱強度帯が既知の評価であり、全強度で運用した性能ではない。学習1,728・検証288・新規テスト576物理イベント、各2ショット。

| 条件 | 選択モデル | 平均mm | p90 mm | 1mm以内率 |
| --- | --- | --- | --- | --- |
| 旧全強度SVR+ET | parent | 3.23115 | 4.50647 | 0.03733 |
| 前回の弱重視切り替え | previous_weak_routed | 3.22059 | 4.45528 | 0.03733 |
| 固定座標基準 | train_mean | 3.23268 | 4.18668 | 0.03299 |
| 通常ノイズ2ms | nom2_template/composite/4.0 | 3.22157 | 4.17451 | 0.03385 |
| 通常ノイズ4ms | nom4_template/gls/1.0 | 3.19256 | 4.21429 | 0.03733 |
| 背景1/10・2ms | quiet2_template/composite/1.0 | 3.16932 | 4.72987 | 0.04253 |
| 背景1/10・4ms | quiet4_template/composite/1.0 | 3.00642 | 4.87044 | 0.06684 |

- 通常ノイズ2ms: 旧モデルとの差 -0.00958 mm、95%区間 [-0.04165,+0.02198]（明確な差なし）。固定座標との差 -0.01111 mm、区間 [-0.01616,-0.00593]。

- 通常ノイズ4ms: 旧モデルとの差 -0.03860 mm、95%区間 [-0.06810,-0.00922]（改善側）。固定座標との差 -0.04012 mm、区間 [-0.05619,-0.02314]。

- 背景1/10・2ms: 旧モデルとの差 -0.06183 mm、95%区間 [-0.12912,+0.00469]（明確な差なし）。固定座標との差 -0.06336 mm、区間 [-0.12098,-0.00603]。

- 背景1/10・4ms: 旧モデルとの差 -0.22473 mm、95%区間 [-0.30669,-0.14003]（改善側）。固定座標との差 -0.22626 mm、区間 [-0.30131,-0.14973]。

## 特権情報による診断（シンドロームだけの実用性能ではない）

| 診断 | モデル | 平均mm | p90 mm |
| --- | --- | --- | --- |
| nom2_oracle_time | nom2_oracle_time/R1000 | 3.24949 | 4.2887 |
| oracle_t1 | oracle_t1/E16 | 0.59573 | 1.15713 |
| oracle_marginals | oracle_marginals/E16 | 0.61108 | 1.13767 |

- nom2_oracle_time: 真の発生時刻を与えて4区間に分割。学習・推定時とも真の時刻を使う診断。

- oracle_t1: ノイズのないd5内49量子ビットのT1過剰緩和率を2.048ms内で4分割平均（196特徴量）。座標・真の時刻を入力に足さない。

- oracle_marginals: 2.048msの24チェックの正確な条件付きdetector発生確率を16分割平均（384特徴量）。有限ショットのゆらぎがないため、実観測では得られない。

- 物理情報で当たってもシンドロームから当たる証拠ではない。逆にモデルが外れても数学的な推定不可能性の証明ではない。

## 試した改善方法

- 通常2ms: 固定切り出し、観測シンドロームの集計から推定した変化点、5つの時間分割候補を結合した特徴量。

- 同じ4.096msシンドロームから2.048msのprefixを取り出して比較。時間窓の比較で別イベントにしない。

- 背景1/10: T1/T2の平常緩和率と回路背景ノイズを1/10にする仮想ハードウェア介入。放射線による過剰緩和率は保存。ソフトウェアのみの改善ではない。

- 平常較正は条件ごとに256独立ショット。旧モデルの評価には旧較正をそのまま使う。

- 9特徴量ケースそれぞれRidge2、SVR2、ExtraTrees2設定（54回の回帰学習）。各ケースの設定は検証で選択。

- 学習イベントの物理的なdetector発生確率をテンプレートにし、観測ビットの時間窓発生率と照合。Bernoulli複合スコア／平常空間共分散で補正するGLSスコア、温度0.25/1/4（4観測条件×6設定）。座標はテンプレート重み付き平均。

- テンプレートに真の中心座標を付けるのは学習イベントだけ。問い合わせの真の確率・中心・時刻はテンプレート推定には使わない。チェックの相関を完全には扱わず、スコアを正確な同時尤度とは呼ばない。

- 固定座標は学習座標平均／既知の幾何中心(0,0)を検証で選択。これを超えるか確認する。

## 全候補のテスト値（テスト順位で選び直さない）

| model | events | mean_mm | p90_mm | within_1mm |
| --- | --- | --- | --- | --- |
| geometry_center | 576 | 3.23181 | 4.17798 | 0.03299 |
| nom2_estimated/E16 | 576 | 3.23489 | 4.21941 | 0.03472 |
| nom2_fixed/E64 | 576 | 3.23056 | 4.19745 | 0.03559 |
| nom2_multi/E64 | 576 | 3.23564 | 4.18431 | 0.03125 |
| nom2_oracle_time/R1000 | 576 | 3.24949 | 4.2887 | 0.03299 |
| nom2_template/composite/0.25 | 576 | 3.39201 | 5.35938 | 0.04861 |
| nom2_template/composite/1.0 | 576 | 3.20511 | 4.31018 | 0.03906 |
| nom2_template/composite/4.0 | 576 | 3.22157 | 4.17451 | 0.03385 |
| nom2_template/gls/0.25 | 576 | 3.23453 | 4.63883 | 0.0408 |
| nom2_template/gls/1.0 | 576 | 3.21237 | 4.20451 | 0.03646 |
| nom2_template/gls/4.0 | 576 | 3.22649 | 4.1744 | 0.03559 |
| nom4_fixed/E64 | 576 | 3.23078 | 4.18028 | 0.03472 |
| nom4_template/composite/0.25 | 576 | 3.46598 | 5.6944 | 0.05469 |
| nom4_template/composite/1.0 | 576 | 3.185 | 4.3717 | 0.03819 |
| nom4_template/composite/4.0 | 576 | 3.2096 | 4.17461 | 0.03646 |
| nom4_template/gls/0.25 | 576 | 3.25925 | 4.91517 | 0.04514 |
| nom4_template/gls/1.0 | 576 | 3.19256 | 4.21429 | 0.03733 |
| nom4_template/gls/4.0 | 576 | 3.21977 | 4.16819 | 0.03559 |
| oracle_marginals/E16 | 576 | 0.61108 | 1.13767 | 0.84375 |
| oracle_t1/E16 | 576 | 0.59573 | 1.15713 | 0.84722 |
| parent | 576 | 3.23115 | 4.50647 | 0.03733 |
| previous_weak_routed | 576 | 3.22059 | 4.45528 | 0.03733 |
| quiet2_fixed/R1000 | 576 | 3.17535 | 4.43489 | 0.03906 |
| quiet2_template/composite/0.25 | 576 | 3.61389 | 6.20336 | 0.0816 |
| quiet2_template/composite/1.0 | 576 | 3.16932 | 4.72987 | 0.04253 |
| quiet2_template/composite/4.0 | 576 | 3.16719 | 4.19346 | 0.03646 |
| quiet2_template/gls/0.25 | 576 | 3.43508 | 5.68686 | 0.07031 |
| quiet2_template/gls/1.0 | 576 | 3.1308 | 4.34807 | 0.04167 |
| quiet2_template/gls/4.0 | 576 | 3.19072 | 4.15589 | 0.03733 |
| quiet4_fixed/S1 | 576 | 3.04278 | 4.68422 | 0.05729 |
| quiet4_template/composite/0.25 | 576 | 3.43747 | 6.28036 | 0.09635 |
| quiet4_template/composite/1.0 | 576 | 3.00642 | 4.87044 | 0.06684 |
| quiet4_template/composite/4.0 | 576 | 3.07084 | 4.2203 | 0.0434 |
| quiet4_template/gls/0.25 | 576 | 3.2556 | 5.782 | 0.09115 |
| quiet4_template/gls/1.0 | 576 | 2.97876 | 4.43834 | 0.05295 |
| quiet4_template/gls/4.0 | 576 | 3.13425 | 4.14746 | 0.03559 |
| train_mean | 576 | 3.23268 | 4.18668 | 0.03299 |

## 対応付き比較

| model | reference | difference_mm | ci95 |
| --- | --- | --- | --- |
| nom2_oracle_time/R1000 | nom2_fixed/E64 | 0.01893 | [0.001541, 0.036479] |
| nom2_oracle_time/R1000 | nom2_template/composite/4.0 | 0.027913 | [0.009658, 0.046083] |
| nom2_oracle_time/R1000 | parent | 0.018333 | [-0.017733, 0.053754] |
| nom2_oracle_time/R1000 | previous_weak_routed | 0.028893 | [-0.002952, 0.060374] |
| nom2_oracle_time/R1000 | train_mean | 0.016806 | [-0.002293, 0.035881] |
| nom2_template/composite/4.0 | nom2_fixed/E64 | -0.008983 | [-0.015384, -0.002727] |
| nom2_template/composite/4.0 | nom2_template/composite/4.0 | 0.0 | [0.0, 0.0] |
| nom2_template/composite/4.0 | parent | -0.00958 | [-0.04165, 0.02198] |
| nom2_template/composite/4.0 | previous_weak_routed | 0.00098 | [-0.026078, 0.02785] |
| nom2_template/composite/4.0 | train_mean | -0.011107 | [-0.016162, -0.005926] |
| nom4_template/gls/1.0 | nom2_fixed/E64 | -0.037999 | [-0.054216, -0.020979] |
| nom4_template/gls/1.0 | nom2_template/composite/4.0 | -0.029016 | [-0.042502, -0.014907] |
| nom4_template/gls/1.0 | parent | -0.038596 | [-0.068105, -0.009224] |
| nom4_template/gls/1.0 | previous_weak_routed | -0.028036 | [-0.053206, -0.002873] |
| nom4_template/gls/1.0 | train_mean | -0.040123 | [-0.056194, -0.023138] |
| oracle_marginals/E16 | nom2_fixed/E64 | -2.619473 | [-2.696472, -2.541709] |
| oracle_marginals/E16 | nom2_template/composite/4.0 | -2.61049 | [-2.687227, -2.532783] |
| oracle_marginals/E16 | parent | -2.62007 | [-2.700542, -2.539354] |
| oracle_marginals/E16 | previous_weak_routed | -2.60951 | [-2.68959, -2.530202] |
| oracle_marginals/E16 | train_mean | -2.621597 | [-2.698314, -2.544576] |
| oracle_t1/E16 | nom2_fixed/E64 | -2.634827 | [-2.710474, -2.559487] |
| oracle_t1/E16 | nom2_template/composite/4.0 | -2.625844 | [-2.701196, -2.550639] |
| oracle_t1/E16 | parent | -2.635424 | [-2.713529, -2.556682] |
| oracle_t1/E16 | previous_weak_routed | -2.624864 | [-2.702277, -2.547506] |
| oracle_t1/E16 | train_mean | -2.636951 | [-2.711932, -2.562072] |
| quiet2_template/composite/1.0 | nom2_fixed/E64 | -0.061235 | [-0.118552, -0.004115] |
| quiet2_template/composite/1.0 | nom2_template/composite/4.0 | -0.052251 | [-0.110007, 0.005423] |
| quiet2_template/composite/1.0 | parent | -0.061832 | [-0.12912, 0.004695] |
| quiet2_template/composite/1.0 | previous_weak_routed | -0.051271 | [-0.116426, 0.013108] |
| quiet2_template/composite/1.0 | train_mean | -0.063359 | [-0.120976, -0.006029] |
| quiet4_template/composite/1.0 | nom2_fixed/E64 | -0.224131 | [-0.298731, -0.146925] |
| quiet4_template/composite/1.0 | nom2_template/composite/4.0 | -0.215148 | [-0.290126, -0.138731] |
| quiet4_template/composite/1.0 | parent | -0.224728 | [-0.306695, -0.140034] |
| quiet4_template/composite/1.0 | previous_weak_routed | -0.214168 | [-0.294493, -0.131731] |
| quiet4_template/composite/1.0 | train_mean | -0.226255 | [-0.301314, -0.149733] |

## 検証時の選択

| candidate | mean_mm |
| --- | --- |
| oracle_t1/E16 | 0.60395 |
| oracle_marginals/E16 | 0.61542 |
| quiet4_template/composite/1.0 | 2.91203 |
| quiet4_template/gls/1.0 | 2.93365 |
| quiet4_fixed/S1 | 2.98357 |
| quiet4_template/composite/4.0 | 3.03242 |
| quiet2_template/composite/1.0 | 3.08555 |
| quiet4_template/gls/0.25 | 3.08601 |
| quiet2_template/gls/1.0 | 3.0931 |
| quiet4_template/gls/4.0 | 3.10639 |
| quiet2_template/composite/4.0 | 3.13497 |
| quiet2_fixed/R1000 | 3.14724 |
| quiet2_template/gls/4.0 | 3.16691 |
| nom4_template/gls/1.0 | 3.19633 |
| nom4_template/composite/4.0 | 3.20005 |
| nom4_template/gls/4.0 | 3.20554 |
| nom2_oracle_time/R1000 | 3.20663 |
| nom4_template/composite/1.0 | 3.20772 |
| nom2_template/composite/4.0 | 3.20912 |
| nom2_template/gls/1.0 | 3.21027 |
| nom2_template/gls/4.0 | 3.21039 |
| train_mean | 3.2122 |
| geometry_center | 3.21231 |
| nom2_multi/E64 | 3.21293 |
| nom2_fixed/E64 | 3.2174 |
| nom4_fixed/E64 | 3.21758 |
| nom2_estimated/E16 | 3.21988 |
| nom2_template/composite/1.0 | 3.22157 |
| previous_weak_routed | 3.26128 |
| quiet4_template/composite/0.25 | 3.27102 |
| parent | 3.28636 |
| nom2_template/gls/0.25 | 3.30818 |
| quiet2_template/gls/0.25 | 3.31767 |
| nom4_template/gls/0.25 | 3.34622 |
| quiet2_template/composite/0.25 | 3.44863 |
| nom2_template/composite/0.25 | 3.54394 |
| nom4_template/composite/0.25 | 3.59314 |

## なぜ弱信号が難しいか（総検出数の診断）

| 条件 | 平常総数期待値 | 平常総数の標準偏差 | 放射線による増加期待値の中央値 | 増加/平常標準偏差の中央値 |
| --- | --- | --- | --- | --- |
| nom2 | 1536.24 | 57.73 | 5.15 | 0.089 |
| nom4 | 3073.23 | 76.43 | 13.63 | 0.178 |
| quiet2 | 158.34 | 20.18 | 5.49 | 0.272 |
| quiet4 | 316.75 | 29.36 | 14.53 | 0.495 |

通常2msでは放射線による検出数増加の期待値は中央値約5件なのに対し、平常総数のショット間標準偏差は約58件。背景低減・観測延長でこの比は改善する。ただし全チェック合計の尺度であり、空間・時間パターンを全て評価した情報量上限ではない。診断用の潜在確率を使用し、モデル入力には渡さない。

## 弱強度帯の前半・後半（副評価）

境界は対数強度帯の中央sqrt(1e-10×1e-9) /us。テストに合わせて境界を探索しない。

| model | group | events | mean_mm |
| --- | --- | --- | --- |
| nom4_template/gls/1.0 | lower_weak | 282 | 3.24667 |
| nom4_template/gls/1.0 | upper_weak | 294 | 3.14065 |
| quiet2_template/composite/1.0 | lower_weak | 282 | 3.34547 |
| quiet2_template/composite/1.0 | upper_weak | 294 | 3.00036 |
| nom2_template/composite/4.0 | lower_weak | 282 | 3.26879 |
| nom2_template/composite/4.0 | upper_weak | 294 | 3.17628 |
| train_mean | lower_weak | 282 | 3.27436 |
| train_mean | upper_weak | 294 | 3.1927 |
| parent | lower_weak | 282 | 3.31032 |
| parent | upper_weak | 294 | 3.15522 |
| quiet4_template/composite/1.0 | lower_weak | 282 | 3.296 |
| quiet4_template/composite/1.0 | upper_weak | 294 | 2.72867 |

## 注意事項

- 2ms通常条件の検証選択を主比較とし、他は副評価。今回のテストを使った再調整はしない。

- 弱専用学習は1,728イベント。旧全強度モデルは2,880（うち弱1,152）なので、旧モデルとの差は純粋な特徴量だけの効果ではない。同じ新学習のnom2_fixedへの対応付き差も保存。

- 真の時刻と潜在物理特徴は診断専用。潜在値はlatent_という別名で保存し、観測推定関数はビットからの特徴量のみを受け取る。

- 95%区間は物理イベント単位の10,000回paired bootstrap。2ショットを独立イベントとは数えない。多重比較、学習データ集合・学習seedの不確実性は未補正。

- d5固定、量子ビット間隔・source領域・公称ハードウェアは前回と同一。BB code・複数イベント・連続検出・未知強度の統合運用には未検証。

- 背景1/10・4msの平均改善は弱帯の上半分に偏る。下半分の平均3.296 mmは固定位置3.274 mmを超えていない。全体p90も前回弱モデル4.455→4.870 mmで悪化方向。平均だけで高精度化・大外れ解消とは主張しない。

- 実装テスト113件通過。全2,592イベントのhash・ビット形状・12層均等・過去seed非重複・物理ハードウェア同一・学習限定テンプレートを監査。

## 成果物

`protocol.json`, `selection.json`, `validation_scores.csv`, `metrics.csv`, `paired.json`, `predictions.csv`, `strength_subgroups.csv`, `audit.json`, `tests.xml`。学習済み回帰器は各ケースのディレクトリ、テンプレートはbank_*.npy。

再実行: リポジトリで `.venv/bin/python -m qp_ode_simulator.weak_information_study <stage> <新出力先>`。stageはprepare→generate_fit→train→freeze→generate_test→evaluate。生成は--workers 8、学習は--workers 3を推奨。OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。
