# 誤差を直接減らす混合と中担当SVRの改善実験（2026-09-10）

## 結論

中担当を中専用SVRへ変更すると、新規テストの中平均が2.7857→2.7674 mm（約0.66%減）になり、主比較97.5%区間は改善側。3つのテスト生成rootすべてで平均が改善した。強は0.8434→0.8365 mmだが、主比較区間が0を含み明確な改善とは言えない。中p90は4.1448→4.1932 mm、強p90は1.6357→1.6730 mmで、尾部改善は確認できなかった。

副候補の誤差直接学習は中2.7495、強0.8285 mm。平均では有望だが、中p90と弱平均の悪化があり、検証で主候補にしなかった。全強度SVRを中担当に入れる変更は中2.8117 mmで悪化。難しい層の重み付けSVRも中の数値は改善するが、強・弱や尾部とのトレードオフが残る。大幅な精度向上ではなく、限定的な改善と失敗条件の整理という結果。

モデルと予測は新規ディレクトリに保存した。既存の既定モデルは自動置換していない。

## 事前選択

主候補: `MiddleOnlySVR`。大外れ制約を外した副候補: `ErrorGate_r0.01_t0`。選択後に新規1,080イベントを生成。テスト最良への差し替えはしていない。

同じ学習2,880イベント・検証360イベント、d5、24チェック×2,047内部round、約2.048ms、通常ノイズ。テストは弱/中/強各360物理イベント、各2ショット。テスト生成rootは3個。

## 共通未使用テスト結果

| 強度・形状 | モデル | イベント数 | 平均mm | 1mm以内% | p90 mm | p95 mm | 3mm超% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak | SVR | 360 | 3.2645 | 2.7778 | 4.8326 | 5.1995 | 58.6111 |
| medium | SVR | 360 | 2.7481 | 8.1944 | 4.3035 | 4.7125 | 43.3333 |
| strong | SVR | 360 | 0.9242 | 66.25 | 1.7985 | 2.1991 | 1.3889 |
| weak | ThreeHard | 360 | 3.2065 | 3.3333 | 4.4186 | 4.6748 | 63.8889 |
| medium | ThreeHard | 360 | 2.7857 | 7.0833 | 4.1448 | 4.438 | 45.2778 |
| strong | ThreeHard | 360 | 0.8434 | 69.3056 | 1.6357 | 2.1132 | 1.1111 |
| weak | MiddleGlobalSVR | 360 | 3.2099 | 3.3333 | 4.3984 | 4.6654 | 63.4722 |
| medium | MiddleGlobalSVR | 360 | 2.8117 | 6.8056 | 4.1734 | 4.444 | 47.0833 |
| strong | MiddleGlobalSVR | 360 | 0.8524 | 68.8889 | 1.6865 | 2.1523 | 1.25 |
| weak | MiddleOnlySVR | 360 | 3.2137 | 3.4722 | 4.4337 | 4.6867 | 63.6111 |
| medium | MiddleOnlySVR | 360 | 2.7674 | 7.6389 | 4.1932 | 4.4599 | 45.0 |
| strong | MiddleOnlySVR | 360 | 0.8365 | 69.7222 | 1.673 | 2.1004 | 1.1111 |
| weak | ErrorGate_r0.01_t0 | 360 | 3.239 | 2.7778 | 4.6538 | 5.0051 | 60.8333 |
| medium | ErrorGate_r0.01_t0 | 360 | 2.7495 | 7.7778 | 4.2274 | 4.605 | 44.3056 |
| strong | ErrorGate_r0.01_t0 | 360 | 0.8285 | 70.1389 | 1.6044 | 2.0418 | 1.25 |
| weak | Overlap | 360 | 3.215 | 3.3333 | 4.5268 | 4.8218 | 61.3889 |
| medium | Overlap | 360 | 2.7749 | 5.9722 | 4.1835 | 4.5442 | 44.8611 |
| strong | Overlap | 360 | 0.8838 | 68.1944 | 1.7462 | 2.1163 | 1.25 |
| weak | REI | 360 | 3.2184 | 2.9167 | 4.384 | 4.6173 | 64.5833 |
| medium | REI | 360 | 2.9787 | 3.6111 | 4.112 | 4.4214 | 52.9167 |
| strong | REI | 360 | 1.64 | 23.1944 | 2.6628 | 2.979 | 4.7222 |

## 主比較と解釈

medium: 主候補−従来ThreeHardの平均差 -0.01829 mm、97.5%対応イベントbootstrap区間 [-0.02926, -0.00748]。負が改善。中・強の2主比較にBonferroni対応した近似区間。

strong: 主候補−従来ThreeHardの平均差 -0.00685 mm、97.5%対応イベントbootstrap区間 [-0.01582, 0.00213]。負が改善。中・強の2主比較にBonferroni対応した近似区間。

検証でp90悪化を制限しても、新規テストでのp90改善を保証しない。平均・1mm以内率・p90/p95・3mm超率を別々に判断する。

## 何を実装したか

- ThreeHardの中担当を、従来の中のみRidge／全強度SVR／中のみSVRの3通りで比較。強度は観測特徴から分類し、真の強度は渡さない。

- ErrorGate: Ridge・SVR・ExtraTrees・ThreeHardの4座標を、シンドローム169特徴＋その予測座標だけからsoftmaxで凸混合。分類精度ではなく位置のEuclidean誤差を直接学習。L2正則化3通り×上位10%平均誤差への重み2通り、各3初期値。学習目的の強度重みは弱.2/中.4/強.4。

- 3-foldの物理イベント単位out-of-fold予測で混合器を学習。同イベントの2ショットは必ず同fold。各fold内でスケーラーと回帰器・分類器を再学習し、予測対象イベントは基礎モデルの学習から除外。基礎ET系は3seedの座標平均。

- OOFの強度×領域×形状別誤差で難しい学習層に重みを付けたSVRを2種類比較。弱は重み1。追加データ生成ではなく、既存学習データの重み付けという低コストの予備実験。

- 新規学習はOOF基礎モデル48学習、中専用/重み付きSVR3学習、混合器18学習の計69学習。検証選択の候補は既存方式込み15構成。最終基礎モデルは前回の凍結済みモデルを再利用。

- 検証選択は中・強の平均を等重みで最小化。ただし各帯p90がThreeHard+.02mm以内、弱平均が+.03mm以内。従来モデルも候補に残し、改善不成立なら維持する。

## 検証での失敗条件

validation_failure_analysis.csv: 中央/端/外側、円/楕円、伝播則、発生時刻・到達範囲・軸比の3分位別。各分位境界は分析対象集合から作る記述的分析で、モデル入力やテスト再選択には使わない。training_difficult_strata.csvはOOF学習誤差。

| model | strength_band | condition | value | events | mean_mm | median_mm | p90_mm | p95_mm | within_1mm | over_3mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ThreeHard | 1 | epicenter_region | edge | 48 | 2.9101 | 3.0235 | 3.9172 | 4.1346 | 0.0208 | 0.5312 |
| ThreeHard | 1 | epicenter_region | inside | 48 | 2.0745 | 2.0395 | 3.3869 | 3.6979 | 0.1146 | 0.1354 |
| ThreeHard | 1 | epicenter_region | outside | 48 | 3.3625 | 3.5684 | 4.5357 | 4.6491 | 0.0417 | 0.7292 |

## 全候補・形状別結果

| 強度・形状 | モデル | イベント数 | 平均mm | 1mm以内% | p90 mm | p95 mm | 3mm超% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak | Ridge | 360 | 3.2204 | 2.7778 | 4.5116 | 4.8953 | 61.3889 |
| medium | Ridge | 360 | 2.8312 | 5.8333 | 4.1701 | 4.604 | 46.25 |
| strong | Ridge | 360 | 1.0599 | 55.4167 | 2.0114 | 2.3854 | 1.5278 |
| strong_circle | Ridge | 180 | 0.9985 | 59.1667 | 1.9999 | 2.3954 | 1.3889 |
| strong_ellipse | Ridge | 180 | 1.1214 | 51.6667 | 2.016 | 2.3698 | 1.6667 |
| weak | SVR | 360 | 3.2645 | 2.7778 | 4.8326 | 5.1995 | 58.6111 |
| medium | SVR | 360 | 2.7481 | 8.1944 | 4.3035 | 4.7125 | 43.3333 |
| strong | SVR | 360 | 0.9242 | 66.25 | 1.7985 | 2.1991 | 1.3889 |
| strong_circle | SVR | 180 | 0.8256 | 73.3333 | 1.6386 | 2.1587 | 1.1111 |
| strong_ellipse | SVR | 180 | 1.0227 | 59.1667 | 1.9262 | 2.201 | 1.6667 |
| weak | ExtraTrees | 360 | 3.1956 | 2.9167 | 4.3394 | 4.5627 | 65.6944 |
| medium | ExtraTrees | 360 | 2.9315 | 3.8889 | 4.1426 | 4.3953 | 50.6944 |
| strong | ExtraTrees | 360 | 1.0221 | 58.8889 | 2.054 | 2.5253 | 2.3611 |
| strong_circle | ExtraTrees | 180 | 0.9535 | 65.5556 | 2.1161 | 2.5247 | 2.2222 |
| strong_ellipse | ExtraTrees | 180 | 1.0907 | 52.2222 | 1.9925 | 2.5253 | 2.5 |
| weak | Blend | 360 | 3.2092 | 2.9167 | 4.4614 | 4.7759 | 61.3889 |
| medium | Blend | 360 | 2.8066 | 4.8611 | 4.1749 | 4.5195 | 45.6944 |
| strong | Blend | 360 | 0.8919 | 65.0 | 1.7824 | 2.1633 | 1.5278 |
| strong_circle | Blend | 180 | 0.8081 | 69.7222 | 1.6097 | 2.1562 | 1.3889 |
| strong_ellipse | Blend | 180 | 0.9757 | 60.2778 | 1.8065 | 2.1638 | 1.6667 |
| weak | ThreeHard | 360 | 3.2065 | 3.3333 | 4.4186 | 4.6748 | 63.8889 |
| medium | ThreeHard | 360 | 2.7857 | 7.0833 | 4.1448 | 4.438 | 45.2778 |
| strong | ThreeHard | 360 | 0.8434 | 69.3056 | 1.6357 | 2.1132 | 1.1111 |
| strong_circle | ThreeHard | 180 | 0.7484 | 75.2778 | 1.4928 | 2.1445 | 0.8333 |
| strong_ellipse | ThreeHard | 180 | 0.9383 | 63.3333 | 1.6949 | 2.0953 | 1.3889 |
| weak | MiddleGlobalSVR | 360 | 3.2099 | 3.3333 | 4.3984 | 4.6654 | 63.4722 |
| medium | MiddleGlobalSVR | 360 | 2.8117 | 6.8056 | 4.1734 | 4.444 | 47.0833 |
| strong | MiddleGlobalSVR | 360 | 0.8524 | 68.8889 | 1.6865 | 2.1523 | 1.25 |
| strong_circle | MiddleGlobalSVR | 180 | 0.7615 | 75.0 | 1.5735 | 2.1667 | 1.1111 |
| strong_ellipse | MiddleGlobalSVR | 180 | 0.9433 | 62.7778 | 1.731 | 2.102 | 1.3889 |
| weak | MiddleOnlySVR | 360 | 3.2137 | 3.4722 | 4.4337 | 4.6867 | 63.6111 |
| medium | MiddleOnlySVR | 360 | 2.7674 | 7.6389 | 4.1932 | 4.4599 | 45.0 |
| strong | MiddleOnlySVR | 360 | 0.8365 | 69.7222 | 1.673 | 2.1004 | 1.1111 |
| strong_circle | MiddleOnlySVR | 180 | 0.7443 | 75.5556 | 1.5165 | 2.1008 | 0.8333 |
| strong_ellipse | MiddleOnlySVR | 180 | 0.9288 | 63.8889 | 1.7023 | 2.0948 | 1.3889 |
| weak | ErrorGate_r0.001_t0.2 | 360 | 3.2253 | 3.3333 | 4.4748 | 4.7695 | 62.9167 |
| medium | ErrorGate_r0.001_t0.2 | 360 | 2.7738 | 7.2222 | 4.2019 | 4.4866 | 45.8333 |
| strong | ErrorGate_r0.001_t0.2 | 360 | 0.8391 | 70.2778 | 1.6468 | 2.0567 | 1.25 |
| strong_circle | ErrorGate_r0.001_t0.2 | 180 | 0.7349 | 77.5 | 1.5304 | 2.0798 | 1.1111 |
| strong_ellipse | ErrorGate_r0.001_t0.2 | 180 | 0.9433 | 63.0556 | 1.6947 | 2.0449 | 1.3889 |
| weak | ErrorGate_r0.001_t0 | 360 | 3.2535 | 3.0556 | 4.6938 | 5.0273 | 61.5278 |
| medium | ErrorGate_r0.001_t0 | 360 | 2.748 | 8.0556 | 4.222 | 4.5892 | 43.6111 |
| strong | ErrorGate_r0.001_t0 | 360 | 0.8315 | 69.8611 | 1.6248 | 2.0491 | 1.25 |
| strong_circle | ErrorGate_r0.001_t0 | 180 | 0.7243 | 76.9444 | 1.4934 | 2.0278 | 1.1111 |
| strong_ellipse | ErrorGate_r0.001_t0 | 180 | 0.9387 | 62.7778 | 1.6707 | 2.0541 | 1.3889 |
| weak | ErrorGate_r0.01_t0.2 | 360 | 3.2188 | 3.3333 | 4.4387 | 4.7997 | 61.6667 |
| medium | ErrorGate_r0.01_t0.2 | 360 | 2.7682 | 7.6389 | 4.1847 | 4.5292 | 45.1389 |
| strong | ErrorGate_r0.01_t0.2 | 360 | 0.8332 | 70.4167 | 1.6218 | 2.0498 | 1.25 |
| strong_circle | ErrorGate_r0.01_t0.2 | 180 | 0.7332 | 77.5 | 1.5035 | 2.0831 | 1.1111 |
| strong_ellipse | ErrorGate_r0.01_t0.2 | 180 | 0.9332 | 63.3333 | 1.6633 | 2.0229 | 1.3889 |
| weak | ErrorGate_r0.01_t0 | 360 | 3.239 | 2.7778 | 4.6538 | 5.0051 | 60.8333 |
| medium | ErrorGate_r0.01_t0 | 360 | 2.7495 | 7.7778 | 4.2274 | 4.605 | 44.3056 |
| strong | ErrorGate_r0.01_t0 | 360 | 0.8285 | 70.1389 | 1.6044 | 2.0418 | 1.25 |
| strong_circle | ErrorGate_r0.01_t0 | 180 | 0.7248 | 77.7778 | 1.5168 | 2.0454 | 1.1111 |
| strong_ellipse | ErrorGate_r0.01_t0 | 180 | 0.9322 | 62.5 | 1.6837 | 2.0338 | 1.3889 |
| weak | ErrorGate_r0.1_t0.2 | 360 | 3.2099 | 3.4722 | 4.432 | 4.7744 | 62.2222 |
| medium | ErrorGate_r0.1_t0.2 | 360 | 2.7717 | 7.2222 | 4.1612 | 4.47 | 45.5556 |
| strong | ErrorGate_r0.1_t0.2 | 360 | 0.8321 | 69.7222 | 1.6167 | 2.0488 | 1.25 |
| strong_circle | ErrorGate_r0.1_t0.2 | 180 | 0.7364 | 76.1111 | 1.509 | 2.0679 | 1.1111 |
| strong_ellipse | ErrorGate_r0.1_t0.2 | 180 | 0.9279 | 63.3333 | 1.6805 | 2.038 | 1.3889 |
| weak | ErrorGate_r0.1_t0 | 360 | 3.2274 | 3.0556 | 4.5936 | 4.9413 | 60.8333 |
| medium | ErrorGate_r0.1_t0 | 360 | 2.7543 | 7.6389 | 4.2092 | 4.6005 | 44.4444 |
| strong | ErrorGate_r0.1_t0 | 360 | 0.8342 | 69.8611 | 1.6087 | 2.0573 | 1.25 |
| strong_circle | ErrorGate_r0.1_t0 | 180 | 0.7378 | 76.6667 | 1.5108 | 2.0655 | 1.1111 |
| strong_ellipse | ErrorGate_r0.1_t0 | 180 | 0.9306 | 63.0556 | 1.682 | 1.9732 | 1.3889 |
| weak | StratumSVR1 | 360 | 3.2701 | 3.0556 | 4.8367 | 5.2345 | 59.0278 |
| medium | StratumSVR1 | 360 | 2.7443 | 8.3333 | 4.3199 | 4.7592 | 43.3333 |
| strong | StratumSVR1 | 360 | 0.9243 | 66.25 | 1.8114 | 2.1279 | 1.3889 |
| strong_circle | StratumSVR1 | 180 | 0.8247 | 73.3333 | 1.6217 | 2.1261 | 1.1111 |
| strong_ellipse | StratumSVR1 | 180 | 1.0238 | 59.1667 | 1.9119 | 2.1284 | 1.6667 |
| weak | StratumSVR2 | 360 | 3.2772 | 2.9167 | 4.8531 | 5.2644 | 58.75 |
| medium | StratumSVR2 | 360 | 2.7399 | 8.3333 | 4.3581 | 4.7858 | 43.6111 |
| strong | StratumSVR2 | 360 | 0.9306 | 66.1111 | 1.7858 | 2.0983 | 1.3889 |
| strong_circle | StratumSVR2 | 180 | 0.8288 | 73.3333 | 1.666 | 2.0842 | 1.1111 |
| strong_ellipse | StratumSVR2 | 180 | 1.0323 | 58.8889 | 1.9078 | 2.1165 | 1.6667 |
| weak | Overlap | 360 | 3.215 | 3.3333 | 4.5268 | 4.8218 | 61.3889 |
| medium | Overlap | 360 | 2.7749 | 5.9722 | 4.1835 | 4.5442 | 44.8611 |
| strong | Overlap | 360 | 0.8838 | 68.1944 | 1.7462 | 2.1163 | 1.25 |
| strong_circle | Overlap | 180 | 0.7902 | 75.0 | 1.6416 | 2.1175 | 1.1111 |
| strong_ellipse | Overlap | 180 | 0.9774 | 61.3889 | 1.8121 | 2.0768 | 1.3889 |
| weak | REI | 360 | 3.2184 | 2.9167 | 4.384 | 4.6173 | 64.5833 |
| medium | REI | 360 | 2.9787 | 3.6111 | 4.112 | 4.4214 | 52.9167 |
| strong | REI | 360 | 1.64 | 23.1944 | 2.6628 | 2.979 | 4.7222 |
| strong_circle | REI | 180 | 1.6152 | 26.3889 | 2.7031 | 3.056 | 6.1111 |
| strong_ellipse | REI | 180 | 1.6648 | 20.0 | 2.6326 | 2.9102 | 3.3333 |
| weak | Prior | 360 | 3.2204 | 3.0556 | 4.2196 | 4.5641 | 70.2778 |
| medium | Prior | 360 | 3.2833 | 1.9444 | 4.2814 | 4.4409 | 72.5 |
| strong | Prior | 360 | 3.1777 | 2.7778 | 4.1841 | 4.4406 | 70.0 |
| strong_circle | Prior | 180 | 3.1675 | 3.8889 | 4.1969 | 4.5365 | 70.0 |
| strong_ellipse | Prior | 180 | 3.1879 | 1.6667 | 4.1403 | 4.3181 | 70.0 |

## 尾部・成功率の差（記述的95%区間）

| group | model | reference | metric | difference | ci95 | unit |
| --- | --- | --- | --- | --- | --- | --- |
| weak | MiddleOnlySVR | ThreeHard | within1 | 0.13889 | [0.0, 0.4166666666666669] | percentage points |
| weak | MiddleOnlySVR | ThreeHard | over3 | -0.27778 | [-1.1111111111111183, 0.5555555555555536] | percentage points |
| weak | MiddleOnlySVR | ThreeHard | p90 | 0.01514 | [-0.014992942340263937, 0.0576687164255895] | mm |
| weak | MiddleOnlySVR | ThreeHard | p95 | 0.01189 | [-0.0046202816062228536, 0.08831947098402274] | mm |
| weak | MiddleOnlySVR | REI | within1 | 0.55556 | [-0.5555555555555557, 1.8055555555555554] | percentage points |
| weak | MiddleOnlySVR | REI | over3 | -0.97222 | [-3.472222222222221, 1.5277777777777724] | percentage points |
| weak | MiddleOnlySVR | REI | p90 | 0.04974 | [-0.02526700959052135, 0.12757379989511988] | mm |
| weak | MiddleOnlySVR | REI | p95 | 0.06944 | [-0.03499108789061584, 0.19267423737735015] | mm |
| medium | MiddleOnlySVR | ThreeHard | within1 | 0.55556 | [-0.4166666666666666, 1.527777777777778] | percentage points |
| medium | MiddleOnlySVR | ThreeHard | over3 | -0.27778 | [-1.250000000000001, 0.694444444444442] | percentage points |
| medium | MiddleOnlySVR | ThreeHard | p90 | 0.04842 | [-0.0172171863789897, 0.08807895207017226] | mm |
| medium | MiddleOnlySVR | ThreeHard | p95 | 0.02189 | [-0.0001802946795086413, 0.08171190272583116] | mm |
| medium | MiddleOnlySVR | REI | within1 | 4.02778 | [2.0833333333333335, 5.972222222222222] | percentage points |
| medium | MiddleOnlySVR | REI | over3 | -7.91667 | [-10.694444444444445, -5.138888888888893] | percentage points |
| medium | MiddleOnlySVR | REI | p90 | 0.08123 | [-0.024096945285479187, 0.13436143659488978] | mm |
| medium | MiddleOnlySVR | REI | p95 | 0.03846 | [-0.06742297751091098, 0.2197601556385235] | mm |
| strong | MiddleOnlySVR | ThreeHard | within1 | 0.41667 | [-0.5555555555555647, 1.5277777777777724] | percentage points |
| strong | MiddleOnlySVR | ThreeHard | over3 | 0.0 | [0.0, 0.0] | percentage points |
| strong | MiddleOnlySVR | ThreeHard | p90 | 0.03737 | [-0.044607296698954024, 0.08526370678187134] | mm |
| strong | MiddleOnlySVR | ThreeHard | p95 | -0.01275 | [-0.15302093201181632, 0.029947481220087495] | mm |
| strong | MiddleOnlySVR | REI | within1 | 46.52778 | [41.80555555555556, 51.38888888888889] | percentage points |
| strong | MiddleOnlySVR | REI | over3 | -3.61111 | [-5.277777777777778, -2.0833333333333335] | percentage points |
| strong | MiddleOnlySVR | REI | p90 | -0.98971 | [-1.1508758569498494, -0.9059162087878531] | mm |
| strong | MiddleOnlySVR | REI | p95 | -0.87858 | [-1.1187910696292973, -0.7502186915742313] | mm |

## 凸包内外の比較

REIの重心推定は出力可能領域に制限があるため、実際の物理qubitの凸包で内外を分けた。シミュレータのinside/edge/outside区分とは別。外側だけの利益を普遍的優位性としない。

重要: 今回の1,080イベントは全てd5の物理qubitの凸包内にある。生成ラベルのoutsideは基準領域[-3,3]mmの外側を意味し、量子ビット配置の外側ではない。物理qubitの座標範囲は各軸[-5,5]mm。今回、チップ外への外挿性能は検証できていない。

| model | strength_band | region | events | mean_mm | median_mm | p90_mm | p95_mm | within_1mm | over_3mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ErrorGate_r0.01_t0 | 0 | inside_hull | 360 | 3.239 | 3.3022 | 4.6538 | 5.0051 | 0.0278 | 0.6083 |
| ErrorGate_r0.01_t0 | 1 | inside_hull | 360 | 2.7495 | 2.7669 | 4.2274 | 4.605 | 0.0778 | 0.4431 |
| ErrorGate_r0.01_t0 | 2 | inside_hull | 360 | 0.8285 | 0.6804 | 1.6044 | 2.0418 | 0.7014 | 0.0125 |
| MiddleGlobalSVR | 0 | inside_hull | 360 | 3.2099 | 3.3154 | 4.3984 | 4.6654 | 0.0333 | 0.6347 |
| MiddleGlobalSVR | 1 | inside_hull | 360 | 2.8117 | 2.9024 | 4.1734 | 4.444 | 0.0681 | 0.4708 |
| MiddleGlobalSVR | 2 | inside_hull | 360 | 0.8524 | 0.6697 | 1.6865 | 2.1523 | 0.6889 | 0.0125 |
| MiddleOnlySVR | 0 | inside_hull | 360 | 3.2137 | 3.3214 | 4.4337 | 4.6867 | 0.0347 | 0.6361 |
| MiddleOnlySVR | 1 | inside_hull | 360 | 2.7674 | 2.8481 | 4.1932 | 4.4599 | 0.0764 | 0.45 |
| MiddleOnlySVR | 2 | inside_hull | 360 | 0.8365 | 0.6678 | 1.673 | 2.1004 | 0.6972 | 0.0111 |
| Overlap | 0 | inside_hull | 360 | 3.215 | 3.2626 | 4.5268 | 4.8218 | 0.0333 | 0.6139 |
| Overlap | 1 | inside_hull | 360 | 2.7749 | 2.8268 | 4.1835 | 4.5442 | 0.0597 | 0.4486 |
| Overlap | 2 | inside_hull | 360 | 0.8838 | 0.7293 | 1.7462 | 2.1163 | 0.6819 | 0.0125 |
| REI | 0 | inside_hull | 360 | 3.2184 | 3.3267 | 4.384 | 4.6173 | 0.0292 | 0.6458 |
| REI | 1 | inside_hull | 360 | 2.9787 | 3.0583 | 4.112 | 4.4214 | 0.0361 | 0.5292 |
| REI | 2 | inside_hull | 360 | 1.64 | 1.562 | 2.6628 | 2.979 | 0.2319 | 0.0472 |
| SVR | 0 | inside_hull | 360 | 3.2645 | 3.2983 | 4.8326 | 5.1995 | 0.0278 | 0.5861 |
| SVR | 1 | inside_hull | 360 | 2.7481 | 2.7749 | 4.3035 | 4.7125 | 0.0819 | 0.4333 |
| SVR | 2 | inside_hull | 360 | 0.9242 | 0.7579 | 1.7985 | 2.1991 | 0.6625 | 0.0139 |
| ThreeHard | 0 | inside_hull | 360 | 3.2065 | 3.3102 | 4.4186 | 4.6748 | 0.0333 | 0.6389 |
| ThreeHard | 1 | inside_hull | 360 | 2.7857 | 2.8649 | 4.1448 | 4.438 | 0.0708 | 0.4528 |
| ThreeHard | 2 | inside_hull | 360 | 0.8434 | 0.6708 | 1.6357 | 2.1132 | 0.6931 | 0.0111 |

## 再現性

test_seed_metrics.csv: 独立生成root3個の結果。seed_variation.csv: 従来系は木/分類器seed、ErrorGateは基礎アンサンブルを固定した混合器初期値の変動。両者を同じ種類の学習変動と扱わない。学習集合の再抽出や異なるデバイスへの再現性は未検証。

| model | group | events | mean_mm | median_mm | p90_mm | p95_mm | within_1mm | over_3mm | seed | variation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiddleOnlySVR | medium | 360 | 2.77844 | 2.85594 | 4.21176 | 4.44978 | 0.07639 | 0.45694 | 41 | base tree/classifier seed |
| MiddleOnlySVR | strong | 360 | 0.83678 | 0.66234 | 1.67231 | 2.08158 | 0.69583 | 0.01111 | 41 | base tree/classifier seed |
| MiddleOnlySVR | medium | 360 | 2.76607 | 2.82495 | 4.19078 | 4.45265 | 0.075 | 0.44722 | 42 | base tree/classifier seed |
| MiddleOnlySVR | strong | 360 | 0.83654 | 0.65628 | 1.66946 | 2.1024 | 0.69444 | 0.01111 | 42 | base tree/classifier seed |
| MiddleOnlySVR | medium | 360 | 2.76532 | 2.85625 | 4.18895 | 4.52997 | 0.07917 | 0.45139 | 43 | base tree/classifier seed |
| MiddleOnlySVR | strong | 360 | 0.83897 | 0.66944 | 1.66901 | 2.13013 | 0.69306 | 0.01111 | 43 | base tree/classifier seed |
| ErrorGate_r0.01_t0 | medium | 360 | 2.74933 | 2.76649 | 4.22759 | 4.61288 | 0.07778 | 0.44306 | 41 | gate initialization only; fixed base ensemble |
| ErrorGate_r0.01_t0 | strong | 360 | 0.82903 | 0.68083 | 1.59435 | 2.04043 | 0.70139 | 0.0125 | 41 | gate initialization only; fixed base ensemble |
| ErrorGate_r0.01_t0 | medium | 360 | 2.7496 | 2.76708 | 4.22664 | 4.6044 | 0.07778 | 0.44306 | 42 | gate initialization only; fixed base ensemble |
| ErrorGate_r0.01_t0 | strong | 360 | 0.82849 | 0.68214 | 1.61536 | 2.04421 | 0.7 | 0.0125 | 42 | gate initialization only; fixed base ensemble |
| ErrorGate_r0.01_t0 | medium | 360 | 2.74954 | 2.76701 | 4.22798 | 4.59769 | 0.07778 | 0.44306 | 43 | gate initialization only; fixed base ensemble |
| ErrorGate_r0.01_t0 | strong | 360 | 0.82849 | 0.68453 | 1.61631 | 2.04088 | 0.69861 | 0.0125 | 43 | gate initialization only; fixed base ensemble |
| ThreeHard | medium | 360 | 2.79434 | 2.88574 | 4.1724 | 4.44858 | 0.07083 | 0.45972 | 41 | base tree/classifier seed |
| ThreeHard | strong | 360 | 0.84538 | 0.67064 | 1.6285 | 2.09222 | 0.68889 | 0.0125 | 41 | base tree/classifier seed |
| ThreeHard | medium | 360 | 2.78543 | 2.84614 | 4.15295 | 4.44486 | 0.07083 | 0.45417 | 42 | base tree/classifier seed |
| ThreeHard | strong | 360 | 0.84493 | 0.66361 | 1.63567 | 2.13807 | 0.68889 | 0.01111 | 42 | base tree/classifier seed |
| ThreeHard | medium | 360 | 2.78317 | 2.88497 | 4.15787 | 4.49578 | 0.07083 | 0.45278 | 43 | base tree/classifier seed |
| ThreeHard | strong | 360 | 0.84726 | 0.67376 | 1.64018 | 2.14166 | 0.68889 | 0.01111 | 43 | base tree/classifier seed |
| MiddleGlobalSVR | medium | 360 | 2.81986 | 2.93608 | 4.206 | 4.44826 | 0.06806 | 0.47917 | 41 | base tree/classifier seed |
| MiddleGlobalSVR | strong | 360 | 0.8527 | 0.66464 | 1.67482 | 2.17173 | 0.68333 | 0.0125 | 41 | base tree/classifier seed |
| MiddleGlobalSVR | medium | 360 | 2.81116 | 2.91312 | 4.1722 | 4.43042 | 0.06944 | 0.47361 | 42 | base tree/classifier seed |
| MiddleGlobalSVR | strong | 360 | 0.8524 | 0.6608 | 1.68561 | 2.15112 | 0.68472 | 0.0125 | 42 | base tree/classifier seed |
| MiddleGlobalSVR | medium | 360 | 2.80916 | 2.91273 | 4.18169 | 4.46162 | 0.06944 | 0.47361 | 43 | base tree/classifier seed |
| MiddleGlobalSVR | strong | 360 | 0.85407 | 0.66944 | 1.69588 | 2.17822 | 0.68611 | 0.0125 | 43 | base tree/classifier seed |

## 1ショット推論時間

CPU単一スレッド、モデル読み込み後、特徴抽出込み、16物理イベント×3反復の48回/モデル。中央値とp90を表示。観測データ取得と学習時間は含めない。現在のPython実装の診断であり、最適化後の限界やリアルタイム保証ではない。予測が保存済み結果と一致することも確認した。

| model | median_ms | p90_ms | mean_ms |
| --- | --- | --- | --- |
| REI | 0.3535 | 0.3903 | 0.3622 |
| SVR | 2.805 | 3.3637 | 2.9449 |
| ThreeHard | 88.4298 | 96.7073 | 90.3777 |
| MiddleOnlySVR | 89.5494 | 103.6598 | 92.1521 |
| ErrorGate_r0.01_t0 | 88.79 | 101.9034 | 91.5523 |

精度改善には計算コストがある。約2.048msの観測窓に対して現行3seed専門家アンサンブルは約90ms/shotを要し、この実装のまま各窓を単一ワーカーで連続処理できるとは言えない。低遅延運用にはモデル圧縮・蒸留・推論の最適化などを別途検証する必要がある。

## 研究上の位置付けと未実施項目

REI原著は検出・中心・影響範囲・実行時間・復号への応用を扱う。今回はイベントがある観測窓からの位置推定だけを比較しており、検出性能・論理誤り率の改善を実証していない。[REI原著](https://arxiv.org/html/2506.16834v1)。

2026-09-10に過去の監査で記録した実装候補 https://github.com/HicrestLaboratory/REI は404で取得できなかった。arXiv v1のリポジトリ引用は公開予定のプレースホルダーだった。原著実装との完全同一性は未確認。現行適用版REIを比較対象と明記する。

両者の観測元は同じ内部detector bitsだが、学習方式は全2,047roundの特徴と独立平常較正・2,880正解イベントを利用し、REIは末尾1,024roundの固定履歴を使用。利用する履歴処理、較正情報、学習コストは等しくない。

強い楕円は学習・検証に含まない。円から楕円への形状×強度の未学習組合せを評価するが、完全未知の伝播則や別ハードウェア・別シミュレータへの汎化ではない。強度区分はシミュレータの生成率proxyであり実機での普遍的分類ではない。

大規模な失敗層追加生成、信頼区間付き位置出力/回答棄却、異なる物理モデルでの評価、影響領域推定、復号器への接続は今回は未実施。モデルの小幅な改善だけで新規性が十分とは断定しない。

## 監査と成果物

実装テスト118件通過。OOFイベント分離、学習/検証/新規テスト非混入、旧生成seed非重複、3root×36層均等、重み・コード・データhashを確認。REI回答率 1.0000。

timingはaudit.jsonに保存したバッチ壁時計の診断値。基礎モデル一括計測には比較用の余分なモデルと読み込み時間を含むため、REIとの速度倍率やオンライン遅延の主張には使わない。

protocol.json / selection.json / validation_metrics.csv / validation_failure_analysis.csv / oof_folds.csv / metrics.csv / paired.json / tail_intervals.json / test_failure_analysis.csv / convex_hull_metrics.csv / test_seed_metrics.csv / seed_variation.csv / audit.json。

実行: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.error_routing_study <prepare|train|generate|evaluate> <出力先>。レポート: examples/report_error_routing.py <出力先>。
