# d5: 弱・中・強イベント専門家と切り替え

## 結論（新規テスト）

以下の選択は全て新規720イベントの生成前に固定。通常600、弱240、強い円120、強い楕円120イベント。各2ショットを別々に推定し、統計区間は物理イベント単位でまとめる。単位mm、小さいほど良い。

| 条件 | 旧SVR+ET | 検証全体選択 | 弱重視選択 | 強重視選択 | 固定位置 | 適用REI |
| --- | --- | --- | --- | --- | --- | --- |
| 通常全体 | 2.5798 | 2.5582 | 2.5845 | 2.5645 | 3.3129 | 2.8394 |
| 弱イベント | 3.3341 | 3.3322 | 3.3263 | 3.3236 | 3.3496 | 3.3446 |
| 強い円形 | 0.7811 | 0.7657 | 0.7729 | 0.7234 | 3.2698 | 1.5915 |
| 強い楕円形 | 0.9633 | 0.9480 | 0.9675 | 0.9406 | 3.1632 | 1.6457 |

- 主評価: `overlap_two/ET16/hard/0.5`。旧モデルとの差 -0.0216 mm、記述的95%区間 [-0.0302, -0.0131]（改善側の区間）。

- 弱重視・副評価: `three/ET16/soft/0.5`。旧モデルとの差 -0.0078 mm、記述的95%区間 [-0.0198, +0.0038]（改善は明確でない）。

- 強重視・副評価: `three/logistic1/soft/1.0`。旧モデルとの差 -0.0577 mm、記述的95%区間 [-0.0905, -0.0259]（改善側の区間）。

## 専門家の学習

旧モデルと同じ2,880学習イベント・169特徴量。専門家はそれぞれの強度範囲だけで学習し、その範囲の検証で選択。強い楕円は全学習・検証選択から除外。

| 専門家 | 学習イベント | 検証イベント | 選択モデル | 対象検証誤差mm |
| --- | --- | --- | --- | --- |
| weak | 1152 | 144 | ET16 | 3.23546 |
| medium | 1152 | 144 | Ridge1000 | 2.67702 |
| strong | 576 | 72 | blend | 0.70518 |
| low_overlap | 2304 | 288 | Ridge1000 | 3.04083 |
| high_overlap | 1728 | 216 | SVR1 | 2.06057 |

- 弱/中/強の真ラベルは生成強度帯。弱[1e-10,1e-9]、中[1e-9,1e-8]、強[1e-8,3e-7] /us（生成率proxy）。学習・採点には使うが実用版の推定入力には使わない。

- 弱のみ、中のみ、強のみ、弱＋中、中＋強の5学習プール。各プールでRidge 2設定、SVR 2設定、ExtraTrees 2設定（30学習）とSVR/ETの座標平均を比較。

- シンドローム169特徴量から弱/中/強を推定する分類器4学習（LogisticRegression C0.1/1、ExtraTrees leaf16/64）。合計34学習。

## 切り替え方式

- strict_two: 弱だけ/強だけで学習した2専門家。弱側重みはP(弱)+0.5P(中)。

- overlap_two: 弱＋中/中＋強で学習した2専門家。同じ重み。中程度を学習範囲で重ねる。

- three: 弱/中/強の3専門家。3クラス確率で座標を平均。

- softは推定確率で混ぜる。hardは最尤クラスに変換（2専門家で中と判定した場合は50:50）。

- さらに旧モデルと0.5:0.5で混ぜる場合／専門家のみの場合を比較。3構成×4分類器×hard/soft×2混合率＝48候補＋旧モデル。

- 主選択は通常検証平均最小。弱・強向けは通常検証が旧モデル+0.01 mm以内という制約で各対象の平均を最小化。

- 実用選択: 全体 `overlap_two/ET16/hard/0.5`、弱 `three/ET16/soft/0.5`、強 `three/logistic1/soft/1.0`。

## 真の強弱で切り替える診断（実用性能ではない）

同じ専門家の予測を真の強度クラスで切り替える。2専門家では中を50:50とする。これは発生中心そのものを教える診断ではないが、真の強度を使うため実用版と混同しない。また最適推定器の厳密な上限ではない。

| 条件 | 旧モデル | 3専門家・推定切替のみ | 3専門家・真の強弱で切替 | 弱専門家単体 | 固定位置 |
| --- | --- | --- | --- | --- | --- |
| 通常全体 | 2.5798 | 2.5810 | 2.5192 | 3.2533 | 3.3129 |
| 弱イベント | 3.3341 | 3.3288 | 3.3405 | 3.3405 | 3.3496 |
| 強い円形 | 0.7811 | 0.7399 | 0.7295 | 3.1125 | 3.2698 |
| 強い楕円形 | 0.9633 | 0.9956 | 0.9525 | 3.0037 | 3.1632 |

弱専門家単体の強イベント誤差は対象外への適用診断であり、通常運用の評価ではない。

## 切り替えだけの寄与（同じ専門家・同じ混合率の対応付き比較）

| 条件 | 真ラベル版 | 観測版 | 差mm | 95%区間 |
| --- | --- | --- | --- | --- |
| ordinary | oracle/strict_two | routed_only/strict_two | -0.0362 | [-0.0505, -0.0225] |
| ordinary | oracle_matched/strict_two | strict_two/logistic1/soft/0.5 | -0.01824 | [-0.02519, -0.01152] |
| ordinary | oracle/overlap_two | routed_only/overlap_two | -0.01224 | [-0.01995, -0.00453] |
| ordinary | oracle_matched/overlap_two | overlap_two/ET16/hard/0.5 | -0.00759 | [-0.01153, -0.00362] |
| ordinary | oracle/three | routed_only/three | -0.06176 | [-0.09316, -0.03073] |
| ordinary | oracle_matched/three | three/ET16/hard/0.5 | -0.02848 | [-0.04472, -0.01254] |
| strong_ellipse | oracle/strict_two | routed_only/strict_two | -0.10903 | [-0.15834, -0.06723] |
| strong_ellipse | oracle_matched/strict_two | strict_two/logistic1/soft/0.5 | -0.04966 | [-0.07371, -0.02949] |
| strong_ellipse | oracle/overlap_two | routed_only/overlap_two | -0.01778 | [-0.03718, -0.00173] |
| strong_ellipse | oracle_matched/overlap_two | overlap_two/ET16/hard/0.5 | -0.01283 | [-0.02327, -0.00401] |
| strong_ellipse | oracle/three | routed_only/three | -0.0431 | [-0.11596, 0.02231] |
| strong_ellipse | oracle_matched/three | three/ET16/hard/0.5 | 0.00557 | [-0.0289, 0.0381] |
| weak | oracle/strict_two | routed_only/strict_two | 0.01335 | [-0.00245, 0.02922] |
| weak | oracle_matched/strict_two | strict_two/logistic1/soft/0.5 | 0.00069 | [-0.00725, 0.00868] |
| weak | oracle/overlap_two | routed_only/overlap_two | -0.00228 | [-0.01195, 0.00715] |
| weak | oracle_matched/overlap_two | overlap_two/ET16/hard/0.5 | -0.00066 | [-0.00552, 0.00411] |
| weak | oracle/three | routed_only/three | 0.0117 | [-0.02838, 0.05245] |
| weak | oracle_matched/three | three/ET16/hard/0.5 | 0.00458 | [-0.01574, 0.02513] |
| strong_circle | oracle/strict_two | routed_only/strict_two | -0.09691 | [-0.14243, -0.05839] |
| strong_circle | oracle_matched/strict_two | strict_two/logistic1/soft/0.5 | -0.04356 | [-0.06562, -0.02493] |
| strong_circle | oracle/overlap_two | routed_only/overlap_two | -0.01298 | [-0.02744, 0.00014] |
| strong_circle | oracle_matched/overlap_two | overlap_two/ET16/hard/0.5 | -0.01177 | [-0.01936, -0.00498] |
| strong_circle | oracle/three | routed_only/three | -0.01048 | [-0.06236, 0.03778] |
| strong_circle | oracle_matched/three | three/ET16/hard/0.5 | 0.01586 | [-0.00943, 0.04103] |

## 全テスト結果（事前に固定した副評価を含む）

| group | model | events | mean_mm | p90_mm | within_1mm |
| --- | --- | --- | --- | --- | --- |
| ordinary | REI | 600 | 2.83935 | 4.18231 | 0.07667 |
| ordinary | expert/high_overlap | 600 | 2.57648 | 4.58764 | 0.19667 |
| ordinary | expert/low_overlap | 600 | 3.64927 | 5.19642 | 0.0675 |
| ordinary | expert/medium | 600 | 4.05897 | 6.17779 | 0.075 |
| ordinary | expert/strong | 600 | 2.56515 | 4.39241 | 0.185 |
| ordinary | expert/weak | 600 | 3.25329 | 4.19754 | 0.02417 |
| ordinary | oracle/overlap_two | 600 | 2.54219 | 4.31092 | 0.19167 |
| ordinary | oracle/strict_two | 600 | 2.65104 | 4.13029 | 0.16667 |
| ordinary | oracle/three | 600 | 2.5192 | 4.1845 | 0.1975 |
| ordinary | oracle_matched/overlap_two | 600 | 2.55063 | 4.26692 | 0.19 |
| ordinary | oracle_matched/strict_two | 600 | 2.60383 | 4.15184 | 0.17417 |
| ordinary | oracle_matched/three | 600 | 2.53123 | 4.19418 | 0.18833 |
| ordinary | overlap_two/ET16/hard/0.5 | 600 | 2.55823 | 4.27025 | 0.1875 |
| ordinary | parent | 600 | 2.57985 | 4.22646 | 0.18 |
| ordinary | prior | 600 | 3.31291 | 4.22161 | 0.02167 |
| ordinary | routed_only/overlap_two | 600 | 2.55443 | 4.30228 | 0.18833 |
| ordinary | routed_only/strict_two | 600 | 2.68724 | 4.13841 | 0.1625 |
| ordinary | routed_only/three | 600 | 2.58095 | 4.17979 | 0.19667 |
| ordinary | strict_two/logistic1/soft/0.5 | 600 | 2.62207 | 4.16207 | 0.17 |
| ordinary | three/ET16/hard/0.5 | 600 | 2.55971 | 4.20771 | 0.19083 |
| ordinary | three/ET16/soft/0.5 | 600 | 2.58452 | 4.21365 | 0.18583 |
| ordinary | three/logistic1/soft/1.0 | 600 | 2.56447 | 4.20206 | 0.18083 |
| strong_ellipse | REI | 120 | 1.64569 | 2.59455 | 0.24583 |
| strong_ellipse | expert/high_overlap | 120 | 0.97522 | 1.74856 | 0.59167 |
| strong_ellipse | expert/low_overlap | 120 | 6.36536 | 16.25911 | 0.16667 |
| strong_ellipse | expert/medium | 120 | 8.29482 | 20.07942 | 0.125 |
| strong_ellipse | expert/strong | 120 | 0.95252 | 1.87737 | 0.625 |
| strong_ellipse | expert/weak | 120 | 3.00368 | 3.99452 | 0.04583 |
| strong_ellipse | oracle/overlap_two | 120 | 0.97522 | 1.74856 | 0.59167 |
| strong_ellipse | oracle/strict_two | 120 | 0.95252 | 1.87737 | 0.625 |
| strong_ellipse | oracle/three | 120 | 0.95252 | 1.87737 | 0.625 |
| strong_ellipse | oracle_matched/overlap_two | 120 | 0.93518 | 1.67154 | 0.65417 |
| strong_ellipse | oracle_matched/strict_two | 120 | 0.94981 | 1.77488 | 0.6375 |
| strong_ellipse | oracle_matched/three | 120 | 0.94981 | 1.77488 | 0.6375 |
| strong_ellipse | overlap_two/ET16/hard/0.5 | 120 | 0.94801 | 1.73238 | 0.65417 |
| strong_ellipse | parent | 120 | 0.9633 | 1.8256 | 0.6375 |
| strong_ellipse | prior | 120 | 3.16319 | 4.10176 | 0.05 |
| strong_ellipse | routed_only/overlap_two | 120 | 0.993 | 1.78648 | 0.58333 |
| strong_ellipse | routed_only/strict_two | 120 | 1.06155 | 2.29311 | 0.59167 |
| strong_ellipse | routed_only/three | 120 | 0.99562 | 1.91188 | 0.61667 |
| strong_ellipse | strict_two/logistic1/soft/0.5 | 120 | 0.99947 | 2.00969 | 0.62083 |
| strong_ellipse | three/ET16/hard/0.5 | 120 | 0.94424 | 1.72925 | 0.63333 |
| strong_ellipse | three/ET16/soft/0.5 | 120 | 0.96749 | 1.84422 | 0.62917 |
| strong_ellipse | three/logistic1/soft/1.0 | 120 | 0.94064 | 1.87592 | 0.64167 |
| weak | REI | 240 | 3.34458 | 4.44464 | 0.01458 |
| weak | expert/high_overlap | 240 | 3.44277 | 5.04078 | 0.03125 |
| weak | expert/low_overlap | 240 | 3.334 | 4.59573 | 0.025 |
| weak | expert/medium | 240 | 3.397 | 4.9448 | 0.02917 |
| weak | expert/strong | 240 | 3.36957 | 4.7104 | 0.03333 |
| weak | expert/weak | 240 | 3.34046 | 4.29708 | 0.01667 |
| weak | oracle/overlap_two | 240 | 3.334 | 4.59573 | 0.025 |
| weak | oracle/strict_two | 240 | 3.34046 | 4.29708 | 0.01667 |
| weak | oracle/three | 240 | 3.34046 | 4.29708 | 0.01667 |
| weak | oracle_matched/overlap_two | 240 | 3.33154 | 4.54603 | 0.02292 |
| weak | oracle_matched/strict_two | 240 | 3.32102 | 4.38846 | 0.01458 |
| weak | oracle_matched/three | 240 | 3.32102 | 4.38846 | 0.01458 |
| weak | overlap_two/ET16/hard/0.5 | 240 | 3.3322 | 4.5507 | 0.02292 |
| weak | parent | 240 | 3.33408 | 4.51716 | 0.02083 |
| weak | prior | 240 | 3.34963 | 4.22256 | 0.01667 |
| weak | routed_only/overlap_two | 240 | 3.33629 | 4.67008 | 0.02708 |
| weak | routed_only/strict_two | 240 | 3.32711 | 4.31899 | 0.01667 |
| weak | routed_only/three | 240 | 3.32876 | 4.37492 | 0.02083 |
| weak | strict_two/logistic1/soft/0.5 | 240 | 3.32033 | 4.40099 | 0.01458 |
| weak | three/ET16/hard/0.5 | 240 | 3.31644 | 4.44364 | 0.01875 |
| weak | three/ET16/soft/0.5 | 240 | 3.32626 | 4.46522 | 0.02083 |
| weak | three/logistic1/soft/1.0 | 240 | 3.3236 | 4.37348 | 0.01875 |
| strong_circle | REI | 120 | 1.5915 | 2.57928 | 0.275 |
| strong_circle | expert/high_overlap | 120 | 0.78673 | 1.41392 | 0.72083 |
| strong_circle | expert/low_overlap | 120 | 6.19383 | 15.21741 | 0.1375 |
| strong_circle | expert/medium | 120 | 8.31524 | 20.57124 | 0.09167 |
| strong_circle | expert/strong | 120 | 0.72947 | 1.52592 | 0.72917 |
| strong_circle | expert/weak | 120 | 3.11246 | 3.96583 | 0.03333 |
| strong_circle | oracle/overlap_two | 120 | 0.78673 | 1.41392 | 0.72083 |
| strong_circle | oracle/strict_two | 120 | 0.72947 | 1.52592 | 0.72917 |
| strong_circle | oracle/three | 120 | 0.72947 | 1.52592 | 0.72917 |
| strong_circle | oracle_matched/overlap_two | 120 | 0.7539 | 1.41119 | 0.7375 |
| strong_circle | oracle_matched/strict_two | 120 | 0.74848 | 1.57855 | 0.73333 |
| strong_circle | oracle_matched/three | 120 | 0.74848 | 1.57855 | 0.73333 |
| strong_circle | overlap_two/ET16/hard/0.5 | 120 | 0.76567 | 1.46975 | 0.72917 |
| strong_circle | parent | 120 | 0.78111 | 1.61324 | 0.72083 |
| strong_circle | prior | 120 | 3.26981 | 4.12507 | 0.025 |
| strong_circle | routed_only/overlap_two | 120 | 0.79971 | 1.45565 | 0.7125 |
| strong_circle | routed_only/strict_two | 120 | 0.82638 | 1.85992 | 0.70417 |
| strong_circle | routed_only/three | 120 | 0.73994 | 1.61942 | 0.75417 |
| strong_circle | strict_two/logistic1/soft/0.5 | 120 | 0.79204 | 1.77876 | 0.72083 |
| strong_circle | three/ET16/hard/0.5 | 120 | 0.73262 | 1.51768 | 0.75833 |
| strong_circle | three/ET16/soft/0.5 | 120 | 0.77287 | 1.58906 | 0.74583 |
| strong_circle | three/logistic1/soft/1.0 | 120 | 0.72338 | 1.48895 | 0.74167 |

## 検証スコア

| model | ordinary | weak | strong |
| --- | --- | --- | --- |
| overlap_two/ET16/hard/0.5 | 2.57304 | 3.30993 | 0.7335 |
| overlap_two/logistic1/soft/0.5 | 2.57335 | 3.31277 | 0.73275 |
| overlap_two/logistic01/soft/0.5 | 2.57357 | 3.31263 | 0.73389 |
| overlap_two/logistic1/hard/0.5 | 2.57423 | 3.31461 | 0.73432 |
| overlap_two/logistic01/hard/0.5 | 2.57444 | 3.31401 | 0.7355 |
| overlap_two/ET64/hard/0.5 | 2.57462 | 3.31082 | 0.73606 |
| overlap_two/ET16/soft/0.5 | 2.57475 | 3.31312 | 0.73242 |
| overlap_two/ET16/hard/1.0 | 2.57577 | 3.32706 | 0.76736 |
| overlap_two/logistic1/soft/1.0 | 2.57604 | 3.3322 | 0.76586 |
| three/ET16/hard/0.5 | 2.57614 | 3.29536 | 0.72594 |
| overlap_two/logistic01/soft/1.0 | 2.57649 | 3.33193 | 0.7686 |
| three/logistic1/soft/0.5 | 2.57679 | 3.29222 | 0.72053 |
| three/logistic01/soft/0.5 | 2.57692 | 3.29155 | 0.72091 |
| three/logistic01/hard/0.5 | 2.57882 | 3.30281 | 0.72788 |
| overlap_two/logistic1/hard/1.0 | 2.57907 | 3.33659 | 0.76851 |
| overlap_two/ET64/hard/1.0 | 2.57926 | 3.32874 | 0.77431 |
| three/logistic1/soft/1.0 | 2.57953 | 3.29541 | 0.71274 |
| overlap_two/logistic01/hard/1.0 | 2.57955 | 3.33549 | 0.77113 |
| three/logistic01/soft/1.0 | 2.57971 | 3.2938 | 0.71502 |
| three/ET64/hard/0.5 | 2.58061 | 3.29453 | 0.73646 |
| three/logistic1/hard/0.5 | 2.58224 | 3.30694 | 0.7356 |
| three/ET16/soft/0.5 | 2.58653 | 3.28697 | 0.72987 |
| parent | 2.58821 | 3.29804 | 0.75807 |
| overlap_two/ET64/soft/0.5 | 2.59157 | 3.3137 | 0.81572 |
| overlap_two/ET16/soft/1.0 | 2.59576 | 3.33338 | 0.85065 |
| three/ET16/hard/1.0 | 2.6002 | 3.31917 | 0.75058 |
| three/ET64/soft/0.5 | 2.60188 | 3.28704 | 0.79962 |
| three/logistic01/hard/1.0 | 2.60378 | 3.33423 | 0.74294 |
| three/ET16/soft/1.0 | 2.60964 | 3.2806 | 0.80518 |
| three/logistic1/hard/1.0 | 2.61092 | 3.34275 | 0.75977 |
| three/ET64/hard/1.0 | 2.61337 | 3.31733 | 0.79252 |
| strict_two/logistic1/soft/0.5 | 2.61801 | 3.26786 | 0.77795 |
| strict_two/logistic01/soft/0.5 | 2.6182 | 3.26762 | 0.78032 |
| strict_two/logistic01/hard/0.5 | 2.61821 | 3.27087 | 0.77887 |
| strict_two/logistic1/hard/0.5 | 2.61849 | 3.27211 | 0.78121 |
| strict_two/ET16/hard/0.5 | 2.62346 | 3.26909 | 0.78954 |
| strict_two/ET64/hard/0.5 | 2.62886 | 3.26799 | 0.81128 |
| strict_two/ET16/soft/0.5 | 2.63351 | 3.2678 | 0.83608 |
| strict_two/ET64/soft/0.5 | 2.64399 | 3.26848 | 0.88885 |
| overlap_two/ET64/soft/1.0 | 2.64932 | 3.33472 | 1.11607 |
| three/ET64/soft/1.0 | 2.65474 | 3.28028 | 1.01772 |
| strict_two/logistic1/soft/1.0 | 2.66785 | 3.25493 | 0.82013 |
| strict_two/logistic01/soft/1.0 | 2.66821 | 3.25428 | 0.82559 |
| strict_two/logistic01/hard/1.0 | 2.67117 | 3.26676 | 0.82077 |
| strict_two/logistic1/hard/1.0 | 2.67232 | 3.2694 | 0.82817 |
| strict_two/ET16/hard/1.0 | 2.6843 | 3.26424 | 0.84953 |
| strict_two/ET64/hard/1.0 | 2.69602 | 3.26215 | 0.8969 |
| strict_two/ET16/soft/1.0 | 2.70559 | 3.25123 | 0.97475 |
| strict_two/ET64/soft/1.0 | 2.73436 | 3.25172 | 1.12137 |

## 注意点・監査

- 弱・強の切り替えに真の強度を使う結果はoracleの表だけ。実用版は1ショットの二値detector events [24,2047]からのみ推定。

- d5、49量子ビット、24チェック、2.048 ms、背景ノイズ・固定ハードウェア・座標領域は前回と同じ。独立の旧平常較正128ショットも固定。

- 中イベントは単純な二択ではない。境界を重ねる2専門家と3専門家を用意した。

- 専門家の学習件数は分割により少なくなる。単なる同サイズモデル比較ではなく、同じ総学習イベント予算での構成比較。

- 旧検証は再利用するが旧テストは選択に使わない。今回新規テストの結果で選び直さない。

- 95%区間は10,000回のイベント単位paired bootstrap。主評価以外は記述的副評価、多重比較・学習集合/seed不確実性は未補正。

- 主選択の通常p90は旧4.2265→4.2702 mm。今回の平均改善は大外れの改善を意味しない。単純な弱のみ/強のみの2専門家は通常平均2.6221 mmで旧2.5798 mmより悪かった。

- REIは旧設定の履歴長1024・適用版。原著条件への優越性の主張ではない。

- 全歴史データとの生成seed非重複、36層均等、専門家の学習UID、同一ハードウェア、モデル/テストhashを監査。実装テスト109件通過。

## 再実行

リポジトリ直下で `.venv/bin/python -m qp_ode_simulator.specialist_study {prepare,train,freeze,generate,evaluate} <新しい出力先>` を順に実行。trainは`--workers 3`、generateは`--workers 8`。BLAS/OMPスレッドは1推奨。既存出力を上書きしない。

- `protocol.json` / `selection.json`: 探索計画、テスト前固定。

- `metrics.csv` / `paired.json` / `predictions.csv`: 全結果、差の区間、各予測。

- `gate_test.json`: 4分類器の混同行列（行:真、列:予測、弱/中/強の順）とlog loss。

- 各専門家ディレクトリの`.joblib`と`frozen.json`: 学習済み重み、選択モデル・学習イベントID。
