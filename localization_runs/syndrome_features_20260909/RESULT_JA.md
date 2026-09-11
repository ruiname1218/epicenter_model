# d5: 時間窓・相関・背景補正の比較実験

## 結論の読み方

通常検証で最良だったのは既存モデルparentであり、新方式を全体用モデルとして採用する根拠は得られなかった。弱信号向け候補は別の副評価として扱う。平均誤差が減っても、固定位置基準を超えなければ位置情報の抽出に成功したとは主張しない。

弱信号候補−旧モデルの新規テスト差: 弱 -0.0077 mm、95%区間 [-0.0208, +0.0056]。通常 +0.0136 mm、95%区間 [+0.0064, +0.0206]。

## 検証で固定したモデル

- 主評価モデル: `parent`。通常検証の平均位置誤差で選択。

- 弱イベント重視モデル: `baseline/learned_logistic_expert_0.5_1`。通常検証が旧モデル+0.01 mm以内という制約で、弱検証の平均誤差を最小化。

- `parent` は前回の2,880イベント学習SVR＋ExtraTreesを再学習せず使用。`prior` は学習座標の平均を常に答える基準。

## 新規テスト結果

全数値は今回の同じ未使用720イベント（各2ショット）で比較。小さいほど良い。通常600、弱240、強い円120、強い楕円120イベント。通常集合に弱・円を含む。

| 条件 | parent (mm) | baseline/learned_logistic_expert_0.5_1 (mm) | prior (mm) | REI (mm) |
| --- | --- | --- | --- | --- |
| 通常テスト | 2.5696 | 2.5831 | 3.2533 | 2.7975 |
| 弱いイベント | 3.2227 | 3.2150 | 3.2223 | 3.2161 |
| 強い円形 | 0.7403 | 0.7475 | 3.2281 | 1.5342 |
| 強い楕円形 | 1.0006 | 1.0072 | 3.1524 | 1.6274 |

## 対応付き差と95%区間

負の差は改善。物理イベント単位で2ショットをまとめて10,000回bootstrap。記述的区間であり、多重比較補正・学習集合再抽出は行っていない。主検定対象は主評価モデル−parentの通常集合。

| 条件 | モデル | 基準 | 差 mm | 95%区間 | 判断 |
| --- | --- | --- | --- | --- | --- |
| ordinary | baseline/learned_logistic_expert_0.5_1 | parent | +0.0136 | [+0.0064, +0.0206] | 悪化側 |
| ordinary | baseline/learned_logistic_expert_0.5_1 | prior | -0.6702 | [-0.7584, -0.5837] | 改善側 |
| ordinary | baseline/learned_logistic_expert_0.5_1 | REI | -0.2144 | [-0.2507, -0.1788] | 改善側 |
| ordinary | parent | parent | +0.0000 | [+0.0000, +0.0000] | 差は明確でない |
| ordinary | parent | prior | -0.6838 | [-0.7729, -0.5947] | 改善側 |
| ordinary | parent | REI | -0.2279 | [-0.2665, -0.1900] | 改善側 |
| strong_ellipse | baseline/learned_logistic_expert_0.5_1 | parent | +0.0066 | [+0.0027, +0.0113] | 悪化側 |
| strong_ellipse | baseline/learned_logistic_expert_0.5_1 | prior | -2.1452 | [-2.3131, -1.9750] | 改善側 |
| strong_ellipse | baseline/learned_logistic_expert_0.5_1 | REI | -0.6202 | [-0.7016, -0.5382] | 改善側 |
| strong_ellipse | parent | parent | +0.0000 | [+0.0000, +0.0000] | 差は明確でない |
| strong_ellipse | parent | prior | -2.1518 | [-2.3197, -1.9825] | 改善側 |
| strong_ellipse | parent | REI | -0.6268 | [-0.7080, -0.5449] | 改善側 |
| weak | baseline/learned_logistic_expert_0.5_1 | parent | -0.0077 | [-0.0208, +0.0056] | 差は明確でない |
| weak | baseline/learned_logistic_expert_0.5_1 | prior | -0.0073 | [-0.0512, +0.0387] | 差は明確でない |
| weak | baseline/learned_logistic_expert_0.5_1 | REI | -0.0011 | [-0.0311, +0.0288] | 差は明確でない |
| weak | parent | parent | +0.0000 | [+0.0000, +0.0000] | 差は明確でない |
| weak | parent | prior | +0.0005 | [-0.0550, +0.0573] | 差は明確でない |
| weak | parent | REI | +0.0066 | [-0.0313, +0.0439] | 差は明確でない |
| strong_circle | baseline/learned_logistic_expert_0.5_1 | parent | +0.0071 | [+0.0025, +0.0133] | 悪化側 |
| strong_circle | baseline/learned_logistic_expert_0.5_1 | prior | -2.4806 | [-2.6376, -2.3197] | 改善側 |
| strong_circle | baseline/learned_logistic_expert_0.5_1 | REI | -0.7867 | [-0.8762, -0.6958] | 改善側 |
| strong_circle | parent | parent | +0.0000 | [+0.0000, +0.0000] | 差は明確でない |
| strong_circle | parent | prior | -2.4878 | [-2.6447, -2.3270] | 改善側 |
| strong_circle | parent | REI | -0.7938 | [-0.8843, -0.7019] | 改善側 |

## 実験した変更

- baseline: 従来の169特徴量。新しい学習手法との対照。

- multiscale: 4/16等分時間窓のチェック別発生率を追加（649特徴量）。16分割は約128 round。

- whitened: multiscaleの空間24チェックの共分散を独立平常データから推定し、縮小共分散で補正（649特徴量）。

- correlation: 近隣チェックの同時発火、±1 roundの遅延ペア、自己の1 round遅延を4窓で集計（1,009特徴量）。平常時のペア発生率を差し引く。

- combined: 背景補正付き時間窓＋相関（1,489特徴量）。

- 各入力でRidge 2設定、SVR 3設定、ExtraTrees 3設定、弱イベント4倍重みのExtraTreesを学習。計45学習。SVR/ETの座標平均も比較。

- 初期35学習の探索に旧選択ハイパーパラメータが含まれていなかったため、検証段階で全5入力に等しく追加した（validation_amendment.json、追加10学習）。探索的な検証変更であり、未使用テスト生成前に全設定を固定した。

- 検証で改善しなかったため、multiscale/combinedに学習データのみでfitするPCA 32/128次元＋SVRを8学習追加。さらにbaseline特徴から弱信号を推定する分類器（ロジスティック回帰/ExtraTrees 2設定）を3学習し、旧予測と固定座標/弱専門家を混ぜる24設定を比較した。合計56学習。追加はrefinement_amendment.jsonに記録、最終テスト生成前に固定。

- 観測信号が小さいときに学習平均座標へ寄せる3閾値のゲートを検証。閾値は学習信号の分位点から固定。真の強度は推定時に使わない。

## 検証スコア（テストでモデルを選び直さない）

| model | ordinary | weak |
| --- | --- | --- |
| parent | 2.58821 | 3.29804 |
| baseline/blend | 2.58882 | 3.29885 |
| baseline/learned_logistic_expert_0.5_2 | 2.59 | 3.2878 |
| baseline/learned_ET16_expert_0.5_2 | 2.59203 | 3.28878 |
| baseline/learned_ET64_expert_0.5_2 | 2.5927 | 3.28926 |
| baseline/learned_logistic_center_0.5_2 | 2.59303 | 3.28425 |
| baseline/learned_logistic_expert_0.5_1 | 2.59374 | 3.28024 |
| baseline/learned_logistic_expert_1.0_2 | 2.59424 | 3.28177 |
| baseline/learned_ET16_expert_1.0_2 | 2.59695 | 3.28093 |
| baseline/learned_ET16_center_0.5_2 | 2.59717 | 3.28575 |
| baseline/learned_ET64_expert_1.0_2 | 2.59822 | 3.28162 |
| baseline/learned_ET64_center_0.5_2 | 2.59919 | 3.28637 |
| baseline/SVR_parent | 2.59957 | 3.37187 |
| baseline/learned_ET16_expert_0.5_1 | 2.60022 | 3.27991 |
| baseline/learned_logistic_center_0.5_1 | 2.60136 | 3.27457 |
| baseline/learned_ET64_expert_0.5_1 | 2.60167 | 3.28027 |
| baseline/learned_logistic_center_1.0_2 | 2.60195 | 3.2772 |
| baseline/learned_logistic_expert_1.0_1 | 2.60429 | 3.27026 |
| baseline/SVR10 | 2.60578 | 3.36155 |
| baseline/learned_ET16_center_1.0_2 | 2.60853 | 3.27584 |
| parent@gate0 | 2.61102 | 3.31317 |
| baseline/blend@gate0 | 2.61136 | 3.31334 |
| multiscale/PCA128_SVR10 | 2.61195 | 3.37651 |
| multiscale/blend | 2.61295 | 3.28678 |
| baseline/learned_ET64_center_1.0_2 | 2.613 | 3.27669 |
| whitened/blend | 2.61518 | 3.28597 |
| baseline/learned_ET16_expert_1.0_1 | 2.61724 | 3.26667 |
| baseline/learned_ET16_center_0.5_1 | 2.61917 | 3.27436 |
| baseline/learned_ET64_expert_1.0_1 | 2.62056 | 3.26698 |
| baseline/learned_logistic_center_1.0_1 | 2.62323 | 3.26408 |
| correlation/blend | 2.62479 | 3.27229 |
| multiscale/SVR_parent | 2.62493 | 3.35294 |
| baseline/learned_ET64_center_0.5_1 | 2.62607 | 3.27481 |
| whitened/SVR_parent | 2.62877 | 3.35352 |
| combined/blend | 2.62929 | 3.27458 |
| multiscale/SVR10 | 2.63267 | 3.35745 |
| baseline/SVR100 | 2.63573 | 3.3945 |
| multiscale/blend@gate0 | 2.63741 | 3.30416 |
| parent@gate1 | 2.63744 | 3.27386 |
| baseline/blend@gate1 | 2.63768 | 3.27378 |
| baseline/Ridge1000 | 2.63792 | 3.31125 |
| correlation/SVR10 | 2.63792 | 3.3297 |
| whitened/SVR10 | 2.63914 | 3.36064 |
| whitened/blend@gate0 | 2.64181 | 3.31201 |
| baseline/Ridge100 | 2.64202 | 3.33057 |
| baseline/ET_parent | 2.64789 | 3.26634 |
| correlation/blend@gate0 | 2.64881 | 3.29862 |
| combined/SVR10 | 2.64936 | 3.32922 |
| combined/blend@gate0 | 2.6504 | 3.29167 |
| combined/SVR_parent | 2.659 | 3.32085 |
| combined/PCA128_SVR10 | 2.65918 | 3.39044 |
| baseline/ET8 | 2.6613 | 3.25564 |
| baseline/learned_ET16_center_1.0_1 | 2.66212 | 3.25916 |
| multiscale/PCA32_SVR10 | 2.66508 | 3.44931 |
| multiscale/blend@gate1 | 2.66536 | 3.27201 |
| whitened/blend@gate1 | 2.66686 | 3.2732 |
| correlation/SVR_parent | 2.66935 | 3.33272 |
| correlation/blend@gate1 | 2.67271 | 3.27001 |
| multiscale/Ridge1000 | 2.67474 | 3.32453 |
| combined/blend@gate1 | 2.67692 | 3.26504 |
| multiscale/ET_parent | 2.67716 | 3.26823 |
| whitened/ET_parent | 2.67875 | 3.26491 |
| multiscale/ET8 | 2.67932 | 3.25885 |
| baseline/learned_ET64_center_1.0_1 | 2.68011 | 3.25939 |
| whitened/Ridge1000 | 2.68042 | 3.32773 |
| whitened/ET8 | 2.68548 | 3.2563 |
| combined/ET_parent | 2.68967 | 3.26042 |
| combined/ET8 | 2.69037 | 3.25414 |
| baseline/ET16 | 2.69068 | 3.26902 |
| correlation/ET8 | 2.69715 | 3.25558 |
| correlation/ET_parent | 2.69718 | 3.27144 |
| multiscale/ET16 | 2.70275 | 3.26383 |
| parent@gate2 | 2.70648 | 3.25274 |
| baseline/blend@gate2 | 2.70667 | 3.25264 |
| whitened/ET16 | 2.70963 | 3.26251 |
| correlation/Ridge1000 | 2.71112 | 3.32542 |
| correlation/ET16 | 2.71397 | 3.2653 |
| multiscale/Ridge100 | 2.71409 | 3.36203 |
| combined/ET16 | 2.71548 | 3.26116 |
| whitened/Ridge100 | 2.71642 | 3.36313 |
| multiscale/SVR100 | 2.7231 | 3.43588 |
| multiscale/blend@gate2 | 2.72716 | 3.25216 |
| whitened/SVR100 | 2.72785 | 3.44069 |
| whitened/blend@gate2 | 2.72872 | 3.2525 |
| correlation/blend@gate2 | 2.72962 | 3.25231 |
| combined/blend@gate2 | 2.73497 | 3.25066 |
| baseline/ET_weak | 2.73985 | 3.24789 |
| combined/Ridge1000 | 2.74984 | 3.33485 |
| baseline/ET_weak@gate0 | 2.75404 | 3.26368 |
| multiscale/ET_weak | 2.76401 | 3.25866 |
| combined/PCA32_SVR10 | 2.76823 | 3.42243 |
| whitened/ET_weak | 2.76973 | 3.25245 |
| multiscale/ET_weak@gate0 | 2.77331 | 3.26778 |
| baseline/ET_weak@gate1 | 2.77364 | 3.2544 |
| multiscale/PCA128_SVR1 | 2.77443 | 3.39084 |
| correlation/ET_weak | 2.77656 | 3.24508 |
| combined/ET_weak | 2.77831 | 3.2487 |
| whitened/ET_weak@gate0 | 2.77889 | 3.26404 |
| multiscale/ET_weak@gate1 | 2.78771 | 3.25475 |
| correlation/ET_weak@gate0 | 2.78852 | 3.25824 |
| combined/ET_weak@gate0 | 2.79047 | 3.26216 |
| whitened/ET_weak@gate1 | 2.79267 | 3.25301 |
| combined/ET_weak@gate1 | 2.80445 | 3.25411 |
| correlation/ET_weak@gate1 | 2.80488 | 3.25132 |
| correlation/SVR100 | 2.80755 | 3.46861 |
| baseline/ET_weak@gate2 | 2.81893 | 3.24778 |
| multiscale/ET_weak@gate2 | 2.82687 | 3.24785 |
| whitened/ET_weak@gate2 | 2.83079 | 3.24702 |
| correlation/Ridge100 | 2.83952 | 3.38127 |
| combined/ET_weak@gate2 | 2.84104 | 3.24783 |
| correlation/ET_weak@gate2 | 2.84198 | 3.24673 |
| combined/SVR100 | 2.85898 | 3.45636 |
| combined/PCA128_SVR1 | 2.89309 | 3.39279 |
| combined/Ridge100 | 2.90192 | 3.39393 |
| multiscale/PCA32_SVR1 | 2.94001 | 3.41317 |
| combined/PCA32_SVR1 | 2.99549 | 3.33433 |
| prior | 3.25209 | 3.24479 |

## 各入力の検証最良設定をテストで評価（副評価）

| group | model | events | mean_mm | p90_mm | within_1mm |
| --- | --- | --- | --- | --- | --- |
| ordinary | REI | 600 | 2.7975 | 4.17521 | 0.0825 |
| ordinary | baseline/blend | 600 | 2.56913 | 4.22848 | 0.1875 |
| ordinary | baseline/learned_logistic_expert_0.5_1 | 600 | 2.58313 | 4.18469 | 0.18583 |
| ordinary | combined/blend | 600 | 2.62337 | 4.17441 | 0.17333 |
| ordinary | correlation/blend | 600 | 2.61523 | 4.15035 | 0.17583 |
| ordinary | multiscale/PCA128_SVR10 | 600 | 2.60488 | 4.44409 | 0.165 |
| ordinary | parent | 600 | 2.56957 | 4.22886 | 0.18917 |
| ordinary | prior | 600 | 3.25332 | 4.23944 | 0.02833 |
| ordinary | whitened/blend | 600 | 2.59678 | 4.20514 | 0.17917 |
| strong_ellipse | REI | 120 | 1.62736 | 2.74859 | 0.25833 |
| strong_ellipse | baseline/blend | 120 | 1.00136 | 1.95317 | 0.57917 |
| strong_ellipse | baseline/learned_logistic_expert_0.5_1 | 120 | 1.00716 | 1.95036 | 0.575 |
| strong_ellipse | combined/blend | 120 | 1.10791 | 2.0705 | 0.53333 |
| strong_ellipse | correlation/blend | 120 | 1.07468 | 2.10866 | 0.575 |
| strong_ellipse | multiscale/PCA128_SVR10 | 120 | 1.14385 | 2.03103 | 0.49583 |
| strong_ellipse | parent | 120 | 1.00059 | 1.93731 | 0.57917 |
| strong_ellipse | prior | 120 | 3.15236 | 4.22978 | 0.05 |
| strong_ellipse | whitened/blend | 120 | 1.09739 | 2.06617 | 0.5125 |
| weak | REI | 240 | 3.21613 | 4.31527 | 0.0375 |
| weak | baseline/blend | 240 | 3.22195 | 4.38732 | 0.03333 |
| weak | baseline/learned_logistic_expert_0.5_1 | 240 | 3.21499 | 4.29056 | 0.03333 |
| weak | combined/blend | 240 | 3.20418 | 4.2936 | 0.03958 |
| weak | correlation/blend | 240 | 3.20583 | 4.28871 | 0.0375 |
| weak | multiscale/PCA128_SVR10 | 240 | 3.2776 | 4.78139 | 0.03125 |
| weak | parent | 240 | 3.22271 | 4.40388 | 0.03333 |
| weak | prior | 240 | 3.22226 | 4.20216 | 0.0375 |
| weak | whitened/blend | 240 | 3.20639 | 4.32579 | 0.0375 |
| strong_circle | REI | 120 | 1.53416 | 2.42122 | 0.25417 |
| strong_circle | baseline/blend | 120 | 0.73983 | 1.52323 | 0.77917 |
| strong_circle | baseline/learned_logistic_expert_0.5_1 | 120 | 0.74746 | 1.53306 | 0.775 |
| strong_circle | combined/blend | 120 | 0.83745 | 1.78124 | 0.71667 |
| strong_circle | correlation/blend | 120 | 0.83143 | 1.77162 | 0.72917 |
| strong_circle | multiscale/PCA128_SVR10 | 120 | 0.90745 | 1.718 | 0.63333 |
| strong_circle | parent | 120 | 0.74033 | 1.52848 | 0.78333 |
| strong_circle | prior | 120 | 3.22809 | 4.19043 | 0.01667 |
| strong_circle | whitened/blend | 120 | 0.79406 | 1.52545 | 0.75 |

## 再現性・制約

- d5、49量子ビット、24チェック、2.048 ms、通常背景ノイズ、同じ固定ハードウェア・発生領域。入力は1ショットの[24,2047] detector events。出力は2次元座標。

- 独立学習2,880イベント・検証360イベントは前回と同じ。過去テストは学習・選択に不使用。検証集合は再利用しているため、検証改善のみを成功とはしない。

- 今回の平常較正は独立512ショット。旧モデルの128ショット較正は変更せず維持。追加較正資源を伴うので、全てを純粋なモデル構造改善とは呼ばない。

- 強い楕円は学習・検証選択から除外した形状×強度の組合せ外挿。弱/中強度の楕円は学習に含む。

- 1つの放射線イベントがあると既知の観測窓。発生有無・オンライン時刻検出・複数イベント・別ハードウェア・BB codeには未検証。

- REIは前回選択済みの履歴長1024を固定した適用版。原著実装・原著条件を上回るとの主張ではない。

- モデル/特徴量/データハッシュ、36層均等、全旧データとのseed非重複、ハードウェア同一、旧holdout非混入を監査。

- 実装テスト106件通過。tests.xmlに保存。

## 成果物

- `protocol.json`, `selection.json`: 探索計画と新規テスト生成前の設定固定。

- `validation_scores.csv`, `metrics.csv`, `paired.json`, `predictions.csv`: 検証・テスト・対応付き差・予測。

- `calibration_bits.npz`, `calibration.joblib`: 独立平常較正。

- 各入力名ディレクトリの `.joblib`: 学習済みモデル。`audit.json`: 監査。
