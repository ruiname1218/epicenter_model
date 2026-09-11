# SVR対適用版REI：同一観測時間での新規比較

2026-09-11。平均位置誤差は物理座標(x,y)のユークリッド距離、mm。以下の率は0–1表記。

## 主結果

**同一観測時間でも、通常条件ではSVRの平均位置誤差が適用版REIより小さかった。ただし、弱イベント・尾部誤差・未知の遅い伝播では限界が残る。**

- ID・4msの全強度平均はREI2.5713→SVR2.2140mm（約13.9%減）。3テストroot全てで平均が改善。
- 中2.9540→2.6337mm（約10.8%減）、強1.5622→0.7425mm（約52.5%減）。円・楕円の両方で平均は改善。強の1mm以内率は23.89→77.78%。
- 弱は3.1977→3.2659mmで改善なし。全体p90も4.0709→4.2096mmに悪化し、その差の探索的95%区間[+0.0238,+0.2706]mmは悪化側。平均改善を全指標の改善としない。
- 回路雑音2倍では4ms全体平均2.6030→2.2678mmと改善を維持。ただし未知の遅い伝播では2.5606→2.5080mm、差95%[-0.1467,+0.0424]mmで優位性は不明確。遅い伝播では短窓内の到達情報が少ない影響も含む。
- CNN/RidgeもIDではREIより平均が小さいが、SVRには及ばない。slow・4msではCNN2.4516、Ridge2.4802がSVR2.5080より点推定で小さく、SVRが全条件で最良ではない。
- 4/8msではREIの検証選択履歴も全履歴であり、主比較と履歴調整後の比較は一致。2msのみ検証選択は1024round。
- 較正なしSVRでもID・4ms平均2.2573mmでREI2.5713mmより小さい。優位性が独立平常較正の有無だけで生じた結果ではない。ただし学習情報や計算コストはREIと同一ではない。

事前固定した主比較は、**通常ID・4ms・全強度平均のSVR−REI_full**。
差は **-0.3573mm**、対応付きイベントbootstrap95%区間は **[-0.4139, -0.3014]mm**。SVRの平均誤差が小さい側。

| model | group | events | mean_mm | within_1mm | p90_mm | over_3mm | answer_rate |
|---|---|---|---|---|---|---|---|
| SVR | all | 540 | 2.2140 | 0.2935 | 4.2096 | 0.3380 | 1.0000 |
| SVR | weak | 180 | 3.2659 | 0.0278 | 4.6373 | 0.6250 | 1.0000 |
| SVR | medium | 180 | 2.6337 | 0.0750 | 4.2470 | 0.3889 | 1.0000 |
| SVR | strong | 180 | 0.7425 | 0.7778 | 1.3825 | 0.0000 | 1.0000 |
| Ridge | all | 540 | 2.3387 | 0.2120 | 4.1081 | 0.3630 | 1.0000 |
| Ridge | weak | 180 | 3.2244 | 0.0278 | 4.4042 | 0.6361 | 1.0000 |
| Ridge | medium | 180 | 2.7545 | 0.0583 | 4.1061 | 0.4444 | 1.0000 |
| Ridge | strong | 180 | 1.0372 | 0.5500 | 1.8677 | 0.0083 | 1.0000 |
| CNN | all | 540 | 2.3545 | 0.2056 | 4.1622 | 0.3602 | 1.0000 |
| CNN | weak | 180 | 3.2588 | 0.0278 | 4.5570 | 0.6278 | 1.0000 |
| CNN | medium | 180 | 2.7843 | 0.0556 | 4.2082 | 0.4444 | 1.0000 |
| CNN | strong | 180 | 1.0204 | 0.5333 | 1.7102 | 0.0083 | 1.0000 |
| REI_full | all | 540 | 2.5713 | 0.1065 | 4.0709 | 0.4185 | 1.0000 |
| REI_full | weak | 180 | 3.1977 | 0.0444 | 4.3025 | 0.6722 | 1.0000 |
| REI_full | medium | 180 | 2.9540 | 0.0361 | 4.1578 | 0.5417 | 1.0000 |
| REI_full | strong | 180 | 1.5622 | 0.2389 | 2.5741 | 0.0417 | 1.0000 |
| REI_valK | all | 540 | 2.5713 | 0.1065 | 4.0709 | 0.4185 | 1.0000 |
| REI_valK | weak | 180 | 3.1977 | 0.0444 | 4.3025 | 0.6722 | 1.0000 |
| REI_valK | medium | 180 | 2.9540 | 0.0361 | 4.1578 | 0.5417 | 1.0000 |
| REI_valK | strong | 180 | 1.5622 | 0.2389 | 2.5741 | 0.0417 | 1.0000 |
| Prior | all | 540 | 3.2154 | 0.0426 | 4.3032 | 0.6870 | 1.0000 |
| Prior | weak | 180 | 3.2115 | 0.0444 | 4.2833 | 0.7389 | 1.0000 |
| Prior | medium | 180 | 3.2416 | 0.0389 | 4.3571 | 0.6556 | 1.0000 |
| Prior | strong | 180 | 3.1932 | 0.0444 | 4.1671 | 0.6667 | 1.0000 |

REI_fullは元記録の4095内部round全てを使用。SVR/Ridgeも同じ先頭4095roundから特徴を作り、CNNも同じ記録の発火率を使う。表現・学習の有無は異なるが、使用可能な観測記録と時間はそろえている。

REI_valKは、検証だけで履歴長を選んだ適用版。主結果は全履歴比較だが、REIの履歴長調整後でも同じ結論かを副比較で示す。テストを見てREIが不利な履歴を選ぶことはしない。

## 実験の設計

- 新規学習864・検証180物理イベント、各2ショット。
- 強度3帯×円/楕円×ballistic/diffusive×位置領域3区分の36層を均等に含む。強い楕円も学習する。
- d5 rotated surface-code memory-Z、49qubit・24check、1μs/round。入力は生測定値ではなく二値detector events。
- 学習時も真の発火確率、物理テンプレート、強度分類教師、伝播条件教師は使わない。SVR/Ridge/CNNの教師は座標のみ。
- 特徴較正は独立の平常シンドローム128ショット。前処理・モデル・REI履歴設定を固定してからテスト生成。
- SVRはC0.1/1/10×gamma0.057/d,0.57/d、Ridgeはalpha1/10/100/1000。検証の全イベント平均誤差で選択。
- CNNは24check×128時間区間を処理するConv1d。座標MSE、最大80epoch、検証平均でepoch選択。3初期化seedの座標平均。3つの独立学習集合ではない。
- 計57学習（SVR18・較正なしSVR18・Ridge12・CNN9）、REI履歴12設定の検証。CNN/Ridgeは事前指定の副比較。

## 多様なテスト条件

- **ID**：同じ物理条件範囲の新規540イベント、3生成root×180。
- **noise2**：上記540イベントと同じ放射線の物理場で回路雑音を2倍にし、別の量子故障乱数で観測。モデルと通常較正はそのまま。独立540物理イベントの追加とは数えない。
- noise2で増やすのはゲートdepolarization、測定反転、リセット雑音。T1/T2由来の成分や全検出エラー率を一律に2倍にする実験ではない。
- **slow**：別の新規216イベント、3生成root×72。ballistic2–6m/s（学習12–40）、diffusion0.5–2mm²/ms（学習5–20）。それ以外の範囲は同じ。
- 固有物理イベント756、条件別イベント数{'ID': 540, 'noise2': 540, 'slow': 216}。観測2.048/4.096/8.192msは同じ8ms記録の先頭を使う。

| domain | horizon | model | mean_mm | within_1mm | p90_mm |
|---|---|---|---|---|---|
| ID | 2 | SVR | 2.3506 | 0.2648 | 4.4296 |
| noise2 | 2 | SVR | 2.3888 | 0.2481 | 4.3601 |
| slow | 2 | SVR | 2.6348 | 0.1620 | 4.5223 |
| ID | 2 | Ridge | 2.4300 | 0.1870 | 4.2327 |
| noise2 | 2 | Ridge | 2.5136 | 0.1713 | 4.2024 |
| slow | 2 | Ridge | 2.5556 | 0.1875 | 4.2631 |
| ID | 2 | CNN | 2.4420 | 0.2046 | 4.3209 |
| noise2 | 2 | CNN | 2.4767 | 0.1880 | 4.2467 |
| slow | 2 | CNN | 2.5939 | 0.1644 | 4.2011 |
| ID | 2 | REI_full | 2.6028 | 0.1176 | 4.1305 |
| noise2 | 2 | REI_full | 2.6377 | 0.1111 | 4.1436 |
| slow | 2 | REI_full | 2.6754 | 0.1181 | 4.0878 |
| ID | 2 | REI_valK | 2.5581 | 0.1204 | 4.1264 |
| noise2 | 2 | REI_valK | 2.6024 | 0.1204 | 4.1830 |
| slow | 2 | REI_valK | 2.5964 | 0.1435 | 4.1362 |
| ID | 4 | SVR | 2.2140 | 0.2935 | 4.2096 |
| noise2 | 4 | SVR | 2.2678 | 0.2741 | 4.1385 |
| slow | 4 | SVR | 2.5080 | 0.1759 | 4.3649 |
| ID | 4 | Ridge | 2.3387 | 0.2120 | 4.1081 |
| noise2 | 4 | Ridge | 2.4274 | 0.1778 | 4.0522 |
| slow | 4 | Ridge | 2.4802 | 0.1759 | 4.1882 |
| ID | 4 | CNN | 2.3545 | 0.2056 | 4.1622 |
| noise2 | 4 | CNN | 2.3598 | 0.2065 | 4.0537 |
| slow | 4 | CNN | 2.4516 | 0.1991 | 4.1943 |
| ID | 4 | REI_full | 2.5713 | 0.1065 | 4.0709 |
| noise2 | 4 | REI_full | 2.6030 | 0.1000 | 4.0549 |
| slow | 4 | REI_full | 2.5606 | 0.1273 | 3.9899 |
| ID | 4 | REI_valK | 2.5713 | 0.1065 | 4.0709 |
| noise2 | 4 | REI_valK | 2.6030 | 0.1000 | 4.0549 |
| slow | 4 | REI_valK | 2.5606 | 0.1273 | 3.9899 |
| ID | 8 | SVR | 2.1378 | 0.3139 | 4.1271 |
| noise2 | 8 | SVR | 2.2310 | 0.2694 | 4.0658 |
| slow | 8 | SVR | 2.4249 | 0.2014 | 4.3481 |
| ID | 8 | Ridge | 2.3118 | 0.2083 | 4.0508 |
| noise2 | 8 | Ridge | 2.4091 | 0.1852 | 4.0069 |
| slow | 8 | Ridge | 2.4978 | 0.1528 | 4.1687 |
| ID | 8 | CNN | 2.2612 | 0.2491 | 4.0511 |
| noise2 | 8 | CNN | 2.2982 | 0.2250 | 4.0063 |
| slow | 8 | CNN | 2.3779 | 0.2060 | 4.1608 |
| ID | 8 | REI_full | 2.6056 | 0.0889 | 3.9823 |
| noise2 | 8 | REI_full | 2.6353 | 0.0870 | 4.0408 |
| slow | 8 | REI_full | 2.5337 | 0.1667 | 3.9761 |
| ID | 8 | REI_valK | 2.6056 | 0.0889 | 3.9823 |
| noise2 | 8 | REI_valK | 2.6353 | 0.0870 | 4.0408 |
| slow | 8 | REI_valK | 2.5337 | 0.1667 | 3.9761 |

## 強度・形状・伝播法別（ID・4ms）

| model | group | events | mean_mm | within_1mm | p90_mm | over_3mm | answer_rate |
|---|---|---|---|---|---|---|---|
| SVR | ballistic | 270 | 2.2539 | 0.2759 | 4.1957 | 0.3315 | 1.0000 |
| SVR | diffusive | 270 | 2.1741 | 0.3111 | 4.2370 | 0.3444 | 1.0000 |
| SVR | weak_circular | 90 | 3.3199 | 0.0278 | 4.7647 | 0.6222 | 1.0000 |
| SVR | weak_elliptical | 90 | 3.2119 | 0.0278 | 4.5577 | 0.6278 | 1.0000 |
| SVR | medium_circular | 90 | 2.6813 | 0.0722 | 4.2629 | 0.3889 | 1.0000 |
| SVR | medium_elliptical | 90 | 2.5862 | 0.0778 | 4.1182 | 0.3889 | 1.0000 |
| SVR | strong_circular | 90 | 0.6529 | 0.8778 | 1.0997 | 0.0000 | 1.0000 |
| SVR | strong_elliptical | 90 | 0.8321 | 0.6778 | 1.5146 | 0.0000 | 1.0000 |
| REI_full | ballistic | 270 | 2.6114 | 0.0944 | 4.0334 | 0.4185 | 1.0000 |
| REI_full | diffusive | 270 | 2.5312 | 0.1185 | 4.0987 | 0.4185 | 1.0000 |
| REI_full | weak_circular | 90 | 3.2680 | 0.0222 | 4.3892 | 0.6833 | 1.0000 |
| REI_full | weak_elliptical | 90 | 3.1275 | 0.0667 | 4.1014 | 0.6611 | 1.0000 |
| REI_full | medium_circular | 90 | 3.0315 | 0.0278 | 4.1803 | 0.5722 | 1.0000 |
| REI_full | medium_elliptical | 90 | 2.8764 | 0.0444 | 4.0259 | 0.5111 | 1.0000 |
| REI_full | strong_circular | 90 | 1.5561 | 0.2556 | 2.5509 | 0.0222 | 1.0000 |
| REI_full | strong_elliptical | 90 | 1.5683 | 0.2222 | 2.5741 | 0.0611 | 1.0000 |

## 履歴長と較正の感度（4ms・全強度平均）

| domain | model | mean_mm | within_1mm | p90_mm | answer_rate |
|---|---|---|---|---|---|
| ID | SVR | 2.2140 | 0.2935 | 4.2096 | 1.0000 |
| noise2 | SVR | 2.2678 | 0.2741 | 4.1385 | 1.0000 |
| slow | SVR | 2.5080 | 0.1759 | 4.3649 | 1.0000 |
| ID | SVR_noquiet | 2.2573 | 0.2611 | 4.1549 | 1.0000 |
| noise2 | SVR_noquiet | 2.3095 | 0.2472 | 4.1474 | 1.0000 |
| slow | SVR_noquiet | 2.5213 | 0.1759 | 4.2675 | 1.0000 |
| ID | REI_full | 2.5713 | 0.1065 | 4.0709 | 1.0000 |
| noise2 | REI_full | 2.6030 | 0.1000 | 4.0549 | 1.0000 |
| slow | REI_full | 2.5606 | 0.1273 | 3.9899 | 1.0000 |
| ID | REI_1024 | 2.6456 | 0.0833 | 4.0966 | 1.0000 |
| noise2 | REI_1024 | 2.6798 | 0.0880 | 4.1460 | 1.0000 |
| slow | REI_1024 | 2.5893 | 0.1319 | 4.1293 | 1.0000 |
| ID | REI_valK | 2.5713 | 0.1065 | 4.0709 | 1.0000 |
| noise2 | REI_valK | 2.6030 | 0.1000 | 4.0549 | 1.0000 |
| slow | REI_valK | 2.5606 | 0.1273 | 3.9899 | 1.0000 |

REIの履歴選択：{'2': 1024, '4': 4095, '8': 8191}（round）。主REIは全履歴、REI_1024は従来の末尾1024固定。反復数1、空間相関倍率2を全てで固定。

SVR_noquietは平常較正を用いない対照。学習済みモデルとREIで事前に利用する情報量まで同じとは言わず、較正の有無と学習コストを明示する。

## 対応付き誤差差

差はmodel−reference、負がSVR側の改善。物理イベント内2ショットを平均してから10,000回bootstrap。1つの主比較以外は全て探索的95%区間で、多重比較補正や研究全体の反復探索に対する保証はない。

| domain | horizon | group | model | reference | difference_mm | ci95 | primary |
|---|---|---|---|---|---|---|---|
| ID | 2 | all | SVR | REI_full | -0.2522 | [-0.3061, -0.1965] | False |
| ID | 2 | weak | SVR | REI_full | 0.1171 | [+0.0416, +0.1905] | False |
| ID | 2 | medium | SVR | REI_full | -0.1879 | [-0.2635, -0.1123] | False |
| ID | 2 | strong | SVR | REI_full | -0.6858 | [-0.7769, -0.5936] | False |
| noise2 | 2 | all | SVR | REI_full | -0.2489 | [-0.2993, -0.1980] | False |
| noise2 | 2 | weak | SVR | REI_full | 0.1295 | [+0.0702, +0.1870] | False |
| noise2 | 2 | medium | SVR | REI_full | -0.1835 | [-0.2467, -0.1183] | False |
| noise2 | 2 | strong | SVR | REI_full | -0.6927 | [-0.7792, -0.6053] | False |
| slow | 2 | all | SVR | REI_full | -0.0406 | [-0.1236, +0.0443] | False |
| slow | 2 | weak | SVR | REI_full | 0.2676 | [+0.1761, +0.3595] | False |
| slow | 2 | medium | SVR | REI_full | -0.0947 | [-0.2279, +0.0330] | False |
| slow | 2 | strong | SVR | REI_full | -0.2948 | [-0.4603, -0.1209] | False |
| ID | 2 | all | SVR | REI_valK | -0.2075 | [-0.2585, -0.1538] | False |
| ID | 2 | weak | SVR | REI_valK | 0.1304 | [+0.0577, +0.2048] | False |
| ID | 2 | medium | SVR | REI_valK | -0.1207 | [-0.1874, -0.0527] | False |
| ID | 2 | strong | SVR | REI_valK | -0.6321 | [-0.7208, -0.5428] | False |
| noise2 | 2 | all | SVR | REI_valK | -0.2137 | [-0.2599, -0.1670] | False |
| noise2 | 2 | weak | SVR | REI_valK | 0.1220 | [+0.0638, +0.1800] | False |
| noise2 | 2 | medium | SVR | REI_valK | -0.1400 | [-0.1978, -0.0821] | False |
| noise2 | 2 | strong | SVR | REI_valK | -0.6229 | [-0.7071, -0.5391] | False |
| slow | 2 | all | SVR | REI_valK | 0.0383 | [-0.0417, +0.1194] | False |
| slow | 2 | weak | SVR | REI_valK | 0.2673 | [+0.1800, +0.3509] | False |
| slow | 2 | medium | SVR | REI_valK | -0.0497 | [-0.1786, +0.0783] | False |
| slow | 2 | strong | SVR | REI_valK | -0.1026 | [-0.2753, +0.0845] | False |
| ID | 4 | all | SVR | REI_full | -0.3573 | [-0.4139, -0.3014] | True |
| ID | 4 | weak | SVR | REI_full | 0.0681 | [-0.0001, +0.1376] | False |
| ID | 4 | medium | SVR | REI_full | -0.3203 | [-0.4045, -0.2347] | False |
| ID | 4 | strong | SVR | REI_full | -0.8197 | [-0.9133, -0.7261] | False |
| noise2 | 4 | all | SVR | REI_full | -0.3351 | [-0.3853, -0.2848] | False |
| noise2 | 4 | weak | SVR | REI_full | 0.0461 | [-0.0084, +0.1000] | False |
| noise2 | 4 | medium | SVR | REI_full | -0.2554 | [-0.3219, -0.1903] | False |
| noise2 | 4 | strong | SVR | REI_full | -0.7961 | [-0.8818, -0.7101] | False |
| slow | 4 | all | SVR | REI_full | -0.0526 | [-0.1467, +0.0424] | False |
| slow | 4 | weak | SVR | REI_full | 0.1966 | [+0.1071, +0.2871] | False |
| slow | 4 | medium | SVR | REI_full | -0.1725 | [-0.2877, -0.0584] | False |
| slow | 4 | strong | SVR | REI_full | -0.1818 | [-0.4071, +0.0530] | False |
| ID | 4 | all | SVR | REI_valK | -0.3573 | [-0.4143, -0.3006] | False |
| ID | 4 | weak | SVR | REI_valK | 0.0681 | [-0.0013, +0.1359] | False |
| ID | 4 | medium | SVR | REI_valK | -0.3203 | [-0.4066, -0.2371] | False |
| ID | 4 | strong | SVR | REI_valK | -0.8197 | [-0.9147, -0.7271] | False |
| noise2 | 4 | all | SVR | REI_valK | -0.3351 | [-0.3845, -0.2854] | False |
| noise2 | 4 | weak | SVR | REI_valK | 0.0461 | [-0.0094, +0.1013] | False |
| noise2 | 4 | medium | SVR | REI_valK | -0.2554 | [-0.3208, -0.1905] | False |
| noise2 | 4 | strong | SVR | REI_valK | -0.7961 | [-0.8804, -0.7095] | False |
| slow | 4 | all | SVR | REI_valK | -0.0526 | [-0.1458, +0.0394] | False |
| slow | 4 | weak | SVR | REI_valK | 0.1966 | [+0.1060, +0.2867] | False |
| slow | 4 | medium | SVR | REI_valK | -0.1725 | [-0.2853, -0.0571] | False |
| slow | 4 | strong | SVR | REI_valK | -0.1818 | [-0.4116, +0.0573] | False |
| ID | 8 | all | SVR | REI_full | -0.4678 | [-0.5294, -0.4070] | False |
| ID | 8 | weak | SVR | REI_full | 0.0516 | [-0.0127, +0.1157] | False |
| ID | 8 | medium | SVR | REI_full | -0.4671 | [-0.5597, -0.3747] | False |
| ID | 8 | strong | SVR | REI_full | -0.9880 | [-1.0866, -0.8865] | False |
| noise2 | 8 | all | SVR | REI_full | -0.4044 | [-0.4572, -0.3501] | False |
| noise2 | 8 | weak | SVR | REI_full | 0.0143 | [-0.0478, +0.0774] | False |
| noise2 | 8 | medium | SVR | REI_full | -0.3112 | [-0.3809, -0.2396] | False |
| noise2 | 8 | strong | SVR | REI_full | -0.9162 | [-1.0040, -0.8248] | False |
| slow | 8 | all | SVR | REI_full | -0.1088 | [-0.2151, -0.0004] | False |
| slow | 8 | weak | SVR | REI_full | 0.0873 | [+0.0029, +0.1722] | False |
| slow | 8 | medium | SVR | REI_full | -0.2513 | [-0.3764, -0.1239] | False |
| slow | 8 | strong | SVR | REI_full | -0.1623 | [-0.4312, +0.1240] | False |
| ID | 8 | all | SVR | REI_valK | -0.4678 | [-0.5297, -0.4063] | False |
| ID | 8 | weak | SVR | REI_valK | 0.0516 | [-0.0119, +0.1144] | False |
| ID | 8 | medium | SVR | REI_valK | -0.4671 | [-0.5596, -0.3752] | False |
| ID | 8 | strong | SVR | REI_valK | -0.9880 | [-1.0876, -0.8880] | False |
| noise2 | 8 | all | SVR | REI_valK | -0.4044 | [-0.4590, -0.3509] | False |
| noise2 | 8 | weak | SVR | REI_valK | 0.0143 | [-0.0479, +0.0788] | False |
| noise2 | 8 | medium | SVR | REI_valK | -0.3112 | [-0.3817, -0.2407] | False |
| noise2 | 8 | strong | SVR | REI_valK | -0.9162 | [-1.0051, -0.8255] | False |
| slow | 8 | all | SVR | REI_valK | -0.1088 | [-0.2148, +0.0003] | False |
| slow | 8 | weak | SVR | REI_valK | 0.0873 | [+0.0030, +0.1707] | False |
| slow | 8 | medium | SVR | REI_valK | -0.2513 | [-0.3791, -0.1256] | False |
| slow | 8 | strong | SVR | REI_valK | -0.1623 | [-0.4315, +0.1253] | False |

## 生成root別（4ms）

| domain | seed | model | mean_mm | p90_mm |
|---|---|---|---|---|
| ID | 20260911511 | SVR | 2.1502 | 4.0384 |
| ID | 20260911521 | SVR | 2.2399 | 4.3213 |
| ID | 20260911531 | SVR | 2.2520 | 4.1999 |
| noise2 | 20260911511 | SVR | 2.2356 | 4.0816 |
| noise2 | 20260911521 | SVR | 2.2128 | 4.1603 |
| noise2 | 20260911531 | SVR | 2.3551 | 4.2122 |
| slow | 20260911541 | SVR | 2.5175 | 4.3798 |
| slow | 20260911551 | SVR | 2.6459 | 4.4546 |
| slow | 20260911561 | SVR | 2.3606 | 4.0494 |
| ID | 20260911511 | REI_full | 2.4934 | 3.9919 |
| ID | 20260911521 | REI_full | 2.5232 | 4.0698 |
| ID | 20260911531 | REI_full | 2.6973 | 4.1578 |
| noise2 | 20260911511 | REI_full | 2.5333 | 3.9600 |
| noise2 | 20260911521 | REI_full | 2.5342 | 3.9706 |
| noise2 | 20260911531 | REI_full | 2.7414 | 4.2358 |
| slow | 20260911541 | REI_full | 2.5916 | 4.0108 |
| slow | 20260911551 | REI_full | 2.6289 | 4.1215 |
| slow | 20260911561 | REI_full | 2.4612 | 3.9101 |
| ID | 20260911511 | REI_valK | 2.4934 | 3.9919 |
| ID | 20260911521 | REI_valK | 2.5232 | 4.0698 |
| ID | 20260911531 | REI_valK | 2.6973 | 4.1578 |
| noise2 | 20260911511 | REI_valK | 2.5333 | 3.9600 |
| noise2 | 20260911521 | REI_valK | 2.5342 | 3.9706 |
| noise2 | 20260911531 | REI_valK | 2.7414 | 4.2358 |
| slow | 20260911541 | REI_valK | 2.5916 | 4.0108 |
| slow | 20260911551 | REI_valK | 2.6289 | 4.1215 |
| slow | 20260911561 | REI_valK | 2.4612 | 3.9101 |

## 公平性・監査

- 同じ条件内では全モデルに同じ物理イベント・同じショット・同じ観測時間を与える。REIとSVRに別のテストを与えない。
- 主比較・比較対象・設定選択規則をテスト生成前に固定。テストで一番良いモデルへの差し替えはしない。
- モデル・較正・前処理・ソースhashを評価時に確認。旧実験との生成seed重複なし。イベント分割で2ショットを分離しない。
- REIの元実装と、保存した発火率統計からの実装の数値一致を自動テスト。推定不能例を除外せず、事前固定した学習座標平均で埋め、回答率を併記。
- 自動テスト140件通過。未来ラウンドの非参照、較正なし特徴、CNNの観測のみの入力も確認。

[共通のシンドローム入力での動作確認](inference_smoke.json)では、REI元実装との一致とSVRの未来非参照も実データで確認。[幾何監査](geometry_audit.json)では、テスト756物理イベントは全てチェック配置・物理qubit配置の両方の凸包内。REIが重心として到達できない凸包外だけの改善ではない。

[副指標の対応区間](endpoint_intervals.json)：全体1mm以内率の差は+18.70ポイント、探索的95%[+15.37,+22.13]。中では+3.89ポイント[+0.83,+6.94]、強では+53.89ポイント[+47.22,+60.83]。中p90差+0.0891mmの区間は0を含む。主比較以外は多重性補正なし。

[実際に生成された速度・拡散係数](actual_parameter_ranges.json)も保存し、slow条件が学習範囲外であることを確認する。

## 主張できる範囲

**原著REIの完全再現ではなく、明示した実装・条件の適用版との比較**。原著の入力表現・回路設定・オンライン手順と同一だとは主張しない。

全て「イベントが存在する既知の観測窓」の位置推定。無イベント検出、誤警報、検出遅延、リアルタイム復号の優越性は未検証。slowでは到達が遅い分の観測情報不足も含め、難しい例を除外しない。

同じ現象論的伝播＋局所QP-ODE＋Pauli近似シミュレータの同一デバイス。速度範囲外と雑音増加は試したが、未知の物理法則、実機、BB codeへの汎用性は保証しない。位置のoutsideラベルは参照領域[-3,3]の外で、実チップ外を意味しない。

モデルのみのバッチ処理時間は[audit.json](audit.json)に保存するが、取得・特徴抽出を除きCNNは読込を含むため、単発オンライン速度の比較には使わない。既定モデルは変更していない。

## 成果物と再現

[事前計画](PLAN_JA.md) / [選択](selection.json) / [監査](audit.json) / [全指標](metrics.csv) / [予測](predictions.csv) / [対応区間](paired.json) / [root別](seed_metrics.csv) / [テスト](tests.xml)

```text
python -m qp_ode_simulator.rei_fair_data prepare NEW_ROOT
python -m qp_ode_simulator.rei_fair_data generate_fit NEW_ROOT
python -m qp_ode_simulator.rei_fair_data cache_fit NEW_ROOT
python -m qp_ode_simulator.rei_fair_models train NEW_ROOT
python -m qp_ode_simulator.rei_fair_data generate_test NEW_ROOT
python -m qp_ode_simulator.rei_fair_data cache_test NEW_ROOT
python -m qp_ode_simulator.rei_fair_models evaluate NEW_ROOT
python examples/report_rei_fair.py NEW_ROOT
```

リポジトリの.venv、OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1で実行。prepareは兄弟のradical_pilot_20260911/protocol.jsonのデバイス設定を参照する。独立追試には新しいseedを使う。

共通の推論関数は`qp_ode_simulator.rei_fair_predict.predict(root, raw, horizon=4, family='SVR')`。rawは[shot,24check,time]の二値配列。SVR/CNN/Ridge/各REIを同じ形式で呼び出せ、返り値は座標と回答フラグ。正解座標や伝播条件の入力は不要。
