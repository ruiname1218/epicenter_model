# 符号距離 d=3・5・7 の対応付き比較

## 結論

距離を増やす効果は強いイベントで確認できた。一方、弱いイベントの精密推定は未解決。全距離で検証選択されたSVR+ExtraTrees座標平均を主要な比較対象とする。

- 通常平均: d3 2.8575 → d5 2.6720 → d7 2.6811 mm。d5/d7とd3の対応付き差の95%区間は改善側。

- 強い円: 1.4164 → 0.9959 → 0.9151 mm。d3→d7で35.4%改善。1 mm以内率44.8%→63.5%→70.8%。

- 強い楕円: 1.5721 → 1.1412 → 1.1622 mm。d3→d7で26.1%改善。

- 弱いイベント: 3.3941 → 3.3777 → 3.3490 mm。d7−d3の差 −0.0452 mm、95%CI [−0.1710,+0.0823]。固定位置3.3534 mmに対する優位性も未確認。

- REIも距離増加で改善。d7の通常/強い円/強い楕円は2.7633/1.2083/1.3403 mm。

- d7のSVR+ExtraTreesはREIより通常/強い円で平均が小さく、差の95%CIも改善側。ただし強い楕円の差は −0.1781 mm [−0.3522,+0.0037]で明確ではない。

- d5→d7の追加効果は補助集計で明確ではない。平均とp90で順位も異なる。d7が必ず最良とは言えない。

- 強いイベントのinside区分（d3基準）も平均1.1970→0.8493→0.9040 mmと改善傾向。境界外イベントの包含だけで全改善を説明するものではないが、領域別の小標本・事後集計である。

- 実装・データ監査: 100 tests passed。固定ハードウェア、同一物理イベント、共通位置の役割一致、過去生成seedとの非重複を確認。

## 実験条件

- 新規物理イベント1,440件 × 3距離 × 2ショット。独立イベント数は1,440件であり、8,640ショットではない。

- 36条件（円/楕円 × ballistic/diffusive × d3基準の内/端/外 × 弱/中/強）を均等生成。

- 864学習候補・288検証・288テスト。強い楕円を学習/検証選択から除き、実際の学習720イベント、検証240イベント。テスト通常240、強い楕円48、弱96、強い円48。

- 量子ビット座標の刻み1 mm、データ/測定間の最短距離√2 mm、同種間最短距離2 mm。d=3/5/7で17/49/97量子ビット、8/24/48チェック。

- 同じ物理座標の和集合上でQP/T1/T2を一度だけ計算し、各回路に切り出す。共通座標の物理場・ハードウェア特性は同じ。回路測定の乱数は距離ごとに独立。

- 発生位置の基準領域は全距離でx,y∈[-3,3] mmに固定。outsideはここから0.25–1 mm外側。大きい符号では旧outsideも内部になり得る。

- 同一面積に量子ビットを詰める実験ではなく、間隔を固定して観測範囲を拡大した実験。

- surface_code:rotated_memory_z、2.048 ms、1 µs/round、1イベント既知の固定窓。通常背景ノイズ。

- 独立平常128ショットで較正。特徴量は7×チェック数+1（57/169/337）。切り分け時刻は学習イベントの開始時刻中央値で固定。推定時には真の位置・強度・形・時刻を渡さない。

- Ridge α3候補、SVR C3×γ3候補、ExtraTrees葉サイズ3候補を各距離の検証集合で選択。γは特徴次元数で補正。SVRとExtraTreesの座標平均も評価。

- CNNは各距離で3seed、最大30epoch・早期停止6epoch、座標アンサンブル。REIは履歴長5候補、r=1・空間倍率2。

- 全距離のモデル/設定を selected_before_test.json に凍結してからテスト採点。推定不能REIは学習位置平均にfallbackし、回答率も報告。

- 以前のd3単独実験とは新規データ・ハードウェア集合・学習規模が異なる。旧数値との直接ランキングはしない。

## 同一テストにおける平均位置誤差（mm）

### Standard test set

| model | d3 | d5 | d7 |
| --- | --- | --- | --- |
| prior | 3.3199 | 3.3199 | 3.3199 |
| REI | 3.0431 | 2.8525 | 2.7633 |
| Ridge | 2.9281 | 2.7277 | 2.7486 |
| SVR | 2.8774 | 2.6652 | 2.7010 |
| ExtraTrees | 2.8940 | 2.7551 | 2.7373 |
| SVR_ET_blend | 2.8575 | 2.6720 | 2.6811 |
| CNN | 2.9657 | 2.8412 | 2.8257 |

### Weak events

| model | d3 | d5 | d7 |
| --- | --- | --- | --- |
| prior | 3.3534 | 3.3534 | 3.3534 |
| REI | 3.3729 | 3.3333 | 3.3125 |
| Ridge | 3.3785 | 3.3662 | 3.3599 |
| SVR | 3.4452 | 3.4495 | 3.4026 |
| ExtraTrees | 3.3806 | 3.3484 | 3.3241 |
| SVR_ET_blend | 3.3941 | 3.3777 | 3.3490 |
| CNN | 3.3493 | 3.3290 | 3.3606 |

### Strong circular events

| model | d3 | d5 | d7 |
| --- | --- | --- | --- |
| prior | 3.2460 | 3.2460 | 3.2460 |
| REI | 2.2445 | 1.5993 | 1.2083 |
| Ridge | 1.7449 | 1.1953 | 1.1796 |
| SVR | 1.5063 | 1.0176 | 0.9955 |
| ExtraTrees | 1.4364 | 1.1924 | 1.0438 |
| SVR_ET_blend | 1.4164 | 0.9959 | 0.9151 |
| CNN | 1.8971 | 1.5285 | 1.3677 |

### Strong elliptical events

| model | d3 | d5 | d7 |
| --- | --- | --- | --- |
| prior | 3.2462 | 3.2462 | 3.2462 |
| REI | 2.2970 | 1.6727 | 1.3403 |
| Ridge | 1.7966 | 1.2853 | 1.1957 |
| SVR | 1.6034 | 1.1996 | 1.2432 |
| ExtraTrees | 1.6982 | 1.2982 | 1.2444 |
| SVR_ET_blend | 1.5721 | 1.1412 | 1.1622 |
| CNN | 1.9977 | 1.5193 | 1.4900 |

## d3からの差と95%信頼区間

差は大きい距離 − d3。負なら改善。イベントごとに2ショットを平均して対応付きbootstrap 10,000回。多重比較補正・学習データ再抽出による不確かさは含まない。

| comparison | group | difference | CI |
| --- | --- | --- | --- |
| Ridge: d5 minus d3 | ordinary | -0.2003 | [-0.2764, -0.1238] |
| Ridge: d7 minus d3 | ordinary | -0.1795 | [-0.2658, -0.0930] |
| SVR: d5 minus d3 | ordinary | -0.2121 | [-0.3185, -0.1040] |
| SVR: d7 minus d3 | ordinary | -0.1764 | [-0.2824, -0.0692] |
| ExtraTrees: d5 minus d3 | ordinary | -0.1388 | [-0.1946, -0.0823] |
| ExtraTrees: d7 minus d3 | ordinary | -0.1567 | [-0.2189, -0.0956] |
| SVR_ET_blend: d5 minus d3 | ordinary | -0.1854 | [-0.2617, -0.1061] |
| SVR_ET_blend: d7 minus d3 | ordinary | -0.1764 | [-0.2573, -0.0956] |
| REI: d5 minus d3 | ordinary | -0.1906 | [-0.2490, -0.1313] |
| REI: d7 minus d3 | ordinary | -0.2798 | [-0.3569, -0.2061] |
| prior: d5 minus d3 | ordinary | +0.0000 | [+0.0000, +0.0000] |
| prior: d7 minus d3 | ordinary | +0.0000 | [+0.0000, +0.0000] |
| CNN: d5 minus d3 | ordinary | -0.1244 | [-0.1915, -0.0593] |
| CNN: d7 minus d3 | ordinary | -0.1400 | [-0.2089, -0.0723] |
| Ridge: d5 minus d3 | strong_ellipse | -0.5113 | [-0.6743, -0.3505] |
| Ridge: d7 minus d3 | strong_ellipse | -0.6009 | [-0.8344, -0.3671] |
| SVR: d5 minus d3 | strong_ellipse | -0.4039 | [-0.6183, -0.1916] |
| SVR: d7 minus d3 | strong_ellipse | -0.3603 | [-0.6175, -0.0971] |
| ExtraTrees: d5 minus d3 | strong_ellipse | -0.4000 | [-0.5497, -0.2540] |
| ExtraTrees: d7 minus d3 | strong_ellipse | -0.4538 | [-0.6086, -0.2951] |
| SVR_ET_blend: d5 minus d3 | strong_ellipse | -0.4309 | [-0.6085, -0.2614] |
| SVR_ET_blend: d7 minus d3 | strong_ellipse | -0.4100 | [-0.6088, -0.2117] |
| REI: d5 minus d3 | strong_ellipse | -0.6243 | [-0.7530, -0.4896] |
| REI: d7 minus d3 | strong_ellipse | -0.9567 | [-1.1657, -0.7499] |
| prior: d5 minus d3 | strong_ellipse | +0.0000 | [+0.0000, +0.0000] |
| prior: d7 minus d3 | strong_ellipse | +0.0000 | [+0.0000, +0.0000] |
| CNN: d5 minus d3 | strong_ellipse | -0.4784 | [-0.6701, -0.2963] |
| CNN: d7 minus d3 | strong_ellipse | -0.5078 | [-0.7496, -0.2789] |
| Ridge: d5 minus d3 | weak | -0.0123 | [-0.1260, +0.1007] |
| Ridge: d7 minus d3 | weak | -0.0186 | [-0.1449, +0.1080] |
| SVR: d5 minus d3 | weak | +0.0043 | [-0.1748, +0.1828] |
| SVR: d7 minus d3 | weak | -0.0426 | [-0.2187, +0.1350] |
| ExtraTrees: d5 minus d3 | weak | -0.0321 | [-0.1191, +0.0549] |
| ExtraTrees: d7 minus d3 | weak | -0.0564 | [-0.1501, +0.0370] |
| SVR_ET_blend: d5 minus d3 | weak | -0.0164 | [-0.1408, +0.1086] |
| SVR_ET_blend: d7 minus d3 | weak | -0.0452 | [-0.1710, +0.0823] |
| REI: d5 minus d3 | weak | -0.0396 | [-0.1267, +0.0467] |
| REI: d7 minus d3 | weak | -0.0604 | [-0.1453, +0.0241] |
| prior: d5 minus d3 | weak | +0.0000 | [+0.0000, +0.0000] |
| prior: d7 minus d3 | weak | +0.0000 | [+0.0000, +0.0000] |
| CNN: d5 minus d3 | weak | -0.0203 | [-0.1197, +0.0772] |
| CNN: d7 minus d3 | weak | +0.0114 | [-0.0880, +0.1077] |
| Ridge: d5 minus d3 | strong_circle | -0.5495 | [-0.7228, -0.3755] |
| Ridge: d7 minus d3 | strong_circle | -0.5653 | [-0.7746, -0.3488] |
| SVR: d5 minus d3 | strong_circle | -0.4887 | [-0.6666, -0.3179] |
| SVR: d7 minus d3 | strong_circle | -0.5108 | [-0.6880, -0.3366] |
| ExtraTrees: d5 minus d3 | strong_circle | -0.2440 | [-0.3736, -0.1126] |
| ExtraTrees: d7 minus d3 | strong_circle | -0.3926 | [-0.5220, -0.2649] |
| SVR_ET_blend: d5 minus d3 | strong_circle | -0.4205 | [-0.5762, -0.2651] |
| SVR_ET_blend: d7 minus d3 | strong_circle | -0.5013 | [-0.6547, -0.3527] |
| REI: d5 minus d3 | strong_circle | -0.6453 | [-0.7542, -0.5330] |
| REI: d7 minus d3 | strong_circle | -1.0362 | [-1.2062, -0.8646] |
| prior: d5 minus d3 | strong_circle | +0.0000 | [+0.0000, +0.0000] |
| prior: d7 minus d3 | strong_circle | +0.0000 | [+0.0000, +0.0000] |
| CNN: d5 minus d3 | strong_circle | -0.3686 | [-0.5440, -0.1927] |
| CNN: d7 minus d3 | strong_circle | -0.5294 | [-0.7056, -0.3639] |

## 補助集計: d7からd5を引いた差

- ordinary: +0.0091 mm [-0.0597, +0.0763]

- weak: -0.0288 mm [-0.1431, +0.0853]

- strong_circle: -0.0808 mm [-0.1832, +0.0194]

- strong_ellipse: +0.0210 mm [-0.1030, +0.1436]

上のd7対d5は追加の記述的比較。これを使ったモデル再選択・再学習はしていない。

## 検証集合で選ばれた設定

- d3: 最良family（検証選択）=SVR_ET_blend, REI履歴=1024, Ridge={'alpha': 1000.0}, SVR={'C': 10.0, 'gamma': 0.001}, ExtraTrees={'leaf': 3}

- d5: 最良family（検証選択）=SVR_ET_blend, REI履歴=1024, Ridge={'alpha': 1000.0}, SVR={'C': 1.0, 'gamma': 0.0033727810650887578}, ExtraTrees={'leaf': 3}

- d7: 最良family（検証選択）=SVR_ET_blend, REI履歴=1024, Ridge={'alpha': 1000.0}, SVR={'C': 1.0, 'gamma': 0.0016913946587537093}, ExtraTrees={'leaf': 3}

## 注意点と成果物

- これはREI適用版との比較であり、原著の入力対応・オンライン検出条件の完全再現ではない。

- 形状/強度等のラベルは学習集合の層化と事後評価に使う。モデル選択には通常条件の検証平均のみを使う。

- 広い面積の観測と、境界の外側を囲む効果が含まれる。dを増やすだけで局所空間分解能が上がったとは言えない。

- 円/楕円ごとの同強度サンプルは独立であり、形だけを変えた対応付き比較ではない。

- metrics.csv: 平均、p90、1/2 mm以内率、回答率。paired.json: 距離間・REI/固定位置との差。

- region_strength_metrics.csv: d3基準の内/端/外×強度別集計。predictions.csv: 全テスト予測。

- distance_comparison.png/pdf/svg: 距離別比較図。protocol.json: 事前設定。

- モデルによる選択効果・3seed集合の差は、イベントbootstrapだけでは全て評価できない。追加の学習seed/独立データが必要。
