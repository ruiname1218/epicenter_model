# 抜本的改善4方向：新規データの横断パイロット

2026-09-11。全数値は同一の新規テスト540物理イベント（各2ショット）の位置誤差。単位mm、低い方が良い。

## 結論

- **通常の観測のまま、中イベントには物理テンプレート照合が有望**。通常SVRの2.6391→2.2759mm、約13.8%減。主比較99.1667%対応区間は差[-0.5178,-0.2071]mmで改善側、3テストrootとも改善。1mm以内率5.28→15.28%。ただしp90は4.0752→4.1154mmで改善せず、強は0.8020→1.1094mmに悪化。万能な置き換えではない。
- **同じ面積の高密度化は中に効果**。49→97qubitで中2.6391→2.4208mm、約8.3%減。主差区間[-0.3494,-0.0839]mm、3rootとも改善。中p90は3.8855mm。強0.8363mmは改善せず、平均差区間も0を含む。ハードウェア資源増加と仮定変更を含む。
- **隣接2パッチは今回の配置・学習では悪化**。中2.7489・強0.8671mm。遠い情報を足せば必ず良くなるわけではないが、パッチ配置や特徴正規化・学習量の違いを残しているので、多パッチ一般の否定ではない。
- **時間CNN・物理補助学習は通常SVRを上回らず**。中2.717–2.772、強0.996–1.057mm。補助課題そのものはある程度学べているが、位置精度には十分つながらなかった。
- **学習forward近似の逆探索は不成功**。中3.2424・強2.5368mm。強帯の検証確率RMSEは0.0551で一定確率対照0.0538より悪く、今回の近似では強イベントの応答を十分表現できていない。これは物理逆問題そのものの限界とは言えない。
- **真の伝播条件を追加するだけでは改善せず**。一方、真のdetector確率を渡すET診断は弱0.7825・中0.7356・強0.7628mm。有限のシンドロームから応答を推定する段階が重要と示唆する。ただしoracleは実用成績でも達成可能な下限の証明でもない。

弱イベントの精密な位置推定は依然未解決。全条件・尾部を同時に改善する新モデルは得ていないので、既定モデルは維持する。今回の最優先の次実験は、**中に照合、強にSVRを使い分ける推定ゲート**。真の強度を運用時に与えず、学習イベント内のOOF予測でゲートを学習し、別の新規テストで評価する必要がある。

## 実施したこと

1. 真の発生時刻・主要17伝播パラメータ・真の条件付きdetector確率を与えるoracle診断。
2. 学習したforward近似からの逆探索、および学習時の物理応答テンプレートからの逆探索。
3. 時間CNNで、発火率のみ／同時発火追加／伝播条件の補助学習／物理応答の補助学習を比較。
4. 通常d5、高密度d7、隣接する2つのd5を同じ物理イベントで比較。

回帰14学習＋時間CNN12学習（4アーム×3seed）＋forward MLP1学習、計27学習。逆探索の温度は検証のみで選択。

## 実験条件とカンニング対策

- 学習720、検証120、テスト540イベント。各2ショットを同じ分割に置く。学習は弱288・中288・強144、検証は48・48・24。テストは各強度180、3生成root。
- 強い楕円は学習・検証に含めず、テストには90イベント含める。強い円も90イベント。
- 約4.096ms、1μs/round。入力は生の測定値そのものではなく、4095内部ラウンドの二値detector events。
- 通常配置の推論はシンドロームだけ。真の座標は学習ラベル、伝播条件・物理応答は補助教師に限定。oracleは明示的な別条件。
- 位置と17条件からのforward近似は、学習由来192位置×64条件を照合。問い合わせイベントの真の条件を渡していない。
- モデル・前処理・較正・候補ライブラリ・ソースのhashを固定してからテスト生成。旧実験のgeneration seedとの非重複も確認。
- 中・強の平均を等重みで評価した検証選択は **template_inverse**。oracleと配置変更はこの選択から除外。テストを見た選び直しはしない。
- 同一の学習イベントを使うため、3ニューラルseedは初期化再現性のみ。3テストrootは新規イベント生成の再現性。独立した学習データ3集合での再学習ではない。

## 通常配置の実用モデル

CNNの各行は3seedの座標平均。SVRとETはそれぞれ1fit。入力は、SVR/ETが相対空間分布120特徴、CNNが128時間区間×24checkの発火率と同時発火。入力表現が異なるため、モデル構造だけの優劣とは解釈しない。

| model | weak | medium | strong | strong_circle | strong_ellipse |
|---|---|---|---|---|---|
| base_SVR | 3.3522 | 2.6391 | 0.8020 | 0.6622 | 0.9418 |
| base_ET | 3.2644 | 2.8777 | 1.0077 | 0.8494 | 1.1660 |
| rates_plain | 3.3675 | 2.7718 | 1.0294 | 0.8794 | 1.1794 |
| coinc_plain | 3.3648 | 2.7243 | 1.0268 | 0.8844 | 1.1691 |
| coinc_joint | 3.3461 | 2.7254 | 0.9955 | 0.8458 | 1.1453 |
| coinc_aux | 3.3544 | 2.7172 | 1.0567 | 0.9014 | 1.2119 |
| forward_inverse | 3.2935 | 3.2424 | 2.5368 | 2.4812 | 2.5924 |
| template_inverse | 3.4124 | 2.2759 | 1.1094 | 1.0755 | 1.1433 |
| Adapted_REI | 3.2769 | 3.0668 | 1.6700 | 1.5761 | 1.7639 |
| Prior | 3.2807 | 3.2427 | 3.1945 | 3.1670 | 3.2221 |

Adapted_REIは同じ通常配置の観測を使った適用版、終端history1024。推定不能時は学習座標平均に置き換える。回答率100.00%。原著条件・原著実装での優越性を示す比較ではない。

## 配置の変更：ハードウェア資源の増加を含む

通常：49qubit・24check。高密度：97qubit・48checkで外接箱[-5,5]mmを維持。2パッチ：通常＋右11mmに独立d5、98qubit・48checkで観測面積を拡大。同一の物理場を全配置の座標和集合でシミュレートし、2パッチの左側は通常と同じ記録。

| model | weak | medium | strong | strong_circle | strong_ellipse |
|---|---|---|---|---|---|
| base_SVR | 3.3522 | 2.6391 | 0.8020 | 0.6622 | 0.9418 |
| dense_SVR | 3.2338 | 2.4208 | 0.8363 | 0.6807 | 0.9918 |
| pair_SVR | 3.3301 | 2.7489 | 0.8671 | 0.7068 | 1.0274 |
| base_ET | 3.2644 | 2.8777 | 1.0077 | 0.8494 | 1.1660 |
| dense_ET | 3.2396 | 2.8011 | 0.9227 | 0.7626 | 1.0827 |
| pair_ET | 3.2599 | 2.9251 | 1.0282 | 0.8893 | 1.1672 |

配置ごとの量子故障は独立サンプル。高密度化によるクロストーク・ゲート速度・配線条件の悪化は含めない。固定デバイスは座標和集合上で定義され、過去の別デバイス配置のスコアと直接比較しない。2パッチは右側1配置のみで、最適配置探索ではない。

## Oracle診断：実用モデルの性能ではない

onsetは真の発生時刻、nuisanceは17パラメータ、nuisance_onlyはシンドロームなしの負の対照、meanは真の条件付きdetector確率。いずれもテスト真値を要求する。

| model | weak | medium | strong | strong_circle | strong_ellipse |
|---|---|---|---|---|---|
| oracle_mean_ET | 0.7825 | 0.7356 | 0.7628 | 0.5421 | 0.9834 |
| oracle_mean_SVR | 2.4549 | 0.7592 | 0.9844 | 0.8710 | 1.0979 |
| oracle_nuisance_ET | 3.2694 | 2.8955 | 1.0178 | 0.8433 | 1.1923 |
| oracle_nuisance_SVR | 3.3541 | 2.6454 | 0.8505 | 0.6857 | 1.0152 |
| oracle_nuisance_only_ET | 3.3712 | 3.3471 | 3.2096 | 3.1743 | 3.2450 |
| oracle_nuisance_only_SVR | 3.4768 | 3.3883 | 3.3301 | 3.3214 | 3.3387 |
| oracle_onset_ET | 3.2669 | 2.8743 | 1.0069 | 0.8408 | 1.1730 |
| oracle_onset_SVR | 3.3469 | 2.6400 | 0.8048 | 0.6629 | 0.9467 |

oracle_meanは確率を与える診断で、有限ショットの実測シンドロームとは異なる。これが良くても、それを単一ショットから復元できるとは限らない。oracle_nuisanceの非改善も、他の推定器や大規模学習で改善しないことの証明ではない。

## 対応付き誤差差と不確かさ

差はmodel−referenceで負が改善。イベント内2ショットを平均してから10,000回bootstrap。主比較は通常配置の検証選択、高密度SVR、2パッチSVRの各中/強（6比較）に対するBonferroni水準99.1667%。選択が基準自身なら自己比較を省略し、残りにも保守的に同じ水準を使う。それ以外は記述的95%。研究全体の反復探索に対する一括保証ではない。

| model | reference | group | difference_mm | interval | level |
|---|---|---|---|---|---|
| dense_SVR | base_SVR | medium | -0.2183 | [-0.3494, -0.0839] | 99.1667% |
| dense_SVR | base_SVR | strong | 0.0342 | [-0.0368, +0.1064] | 99.1667% |
| pair_SVR | base_SVR | medium | 0.1097 | [+0.0328, +0.1840] | 99.1667% |
| pair_SVR | base_SVR | strong | 0.0651 | [+0.0129, +0.1179] | 99.1667% |
| template_inverse | base_SVR | medium | -0.3632 | [-0.5178, -0.2071] | 99.1667% |
| template_inverse | base_SVR | strong | 0.3074 | [+0.1743, +0.4446] | 99.1667% |
| coinc_plain | rates_plain | weak | -0.0027 | [-0.0556, +0.0485] | 95% exploratory |
| coinc_plain | rates_plain | medium | -0.0475 | [-0.0909, -0.0033] | 95% exploratory |
| coinc_plain | rates_plain | strong | -0.0026 | [-0.0463, +0.0417] | 95% exploratory |
| coinc_joint | coinc_plain | weak | -0.0187 | [-0.0445, +0.0068] | 95% exploratory |
| coinc_joint | coinc_plain | medium | 0.0011 | [-0.0212, +0.0235] | 95% exploratory |
| coinc_joint | coinc_plain | strong | -0.0313 | [-0.0500, -0.0133] | 95% exploratory |
| coinc_aux | coinc_plain | weak | -0.0104 | [-0.0281, +0.0070] | 95% exploratory |
| coinc_aux | coinc_plain | medium | -0.0070 | [-0.0240, +0.0096] | 95% exploratory |
| coinc_aux | coinc_plain | strong | 0.0299 | [-0.0138, +0.0740] | 95% exploratory |
| forward_inverse | template_inverse | weak | -0.1189 | [-0.2254, -0.0103] | 95% exploratory |
| forward_inverse | template_inverse | medium | 0.9665 | [+0.7852, +1.1452] | 95% exploratory |
| forward_inverse | template_inverse | strong | 1.4274 | [+1.2551, +1.6031] | 95% exploratory |

## 1mm以内率・尾部

率は0–1表記。

| model | group | within_1mm | p90_mm | p95_mm | over_3mm |
|---|---|---|---|---|---|
| base_SVR | medium | 0.0528 | 4.0752 | 4.3540 | 0.3611 |
| base_SVR | strong | 0.7139 | 1.5291 | 1.8538 | 0.0000 |
| dense_SVR | medium | 0.0806 | 3.8855 | 4.1619 | 0.2917 |
| dense_SVR | strong | 0.6917 | 1.6288 | 1.9297 | 0.0028 |
| pair_SVR | medium | 0.0528 | 4.0139 | 4.4804 | 0.4111 |
| pair_SVR | strong | 0.6833 | 1.6685 | 2.0246 | 0.0028 |
| template_inverse | medium | 0.1528 | 4.1154 | 4.5269 | 0.2722 |
| template_inverse | strong | 0.5000 | 1.8958 | 2.2564 | 0.0083 |

## テスト生成root別

| test_seed | model | group | mean_mm |
|---|---|---|---|
| 20260911101 | base_SVR | medium | 2.7339 |
| 20260911101 | base_SVR | strong | 0.7334 |
| 20260911101 | dense_SVR | medium | 2.4716 |
| 20260911101 | dense_SVR | strong | 0.7934 |
| 20260911101 | pair_SVR | medium | 2.7949 |
| 20260911101 | pair_SVR | strong | 0.8180 |
| 20260911101 | template_inverse | medium | 2.3847 |
| 20260911101 | template_inverse | strong | 1.0639 |
| 20260911201 | base_SVR | medium | 2.6526 |
| 20260911201 | base_SVR | strong | 0.8016 |
| 20260911201 | dense_SVR | medium | 2.4161 |
| 20260911201 | dense_SVR | strong | 0.7339 |
| 20260911201 | pair_SVR | medium | 2.7553 |
| 20260911201 | pair_SVR | strong | 0.8044 |
| 20260911201 | template_inverse | medium | 2.3256 |
| 20260911201 | template_inverse | strong | 1.0379 |
| 20260911301 | base_SVR | medium | 2.5309 |
| 20260911301 | base_SVR | strong | 0.8710 |
| 20260911301 | dense_SVR | medium | 2.3748 |
| 20260911301 | dense_SVR | strong | 0.9815 |
| 20260911301 | pair_SVR | medium | 2.6965 |
| 20260911301 | pair_SVR | strong | 0.9789 |
| 20260911301 | template_inverse | medium | 2.1175 |
| 20260911301 | template_inverse | strong | 1.2263 |

## 時間モデルの学習

全アームは同じConv1d構造・ヘッド数。rates_plainは同時発火入力を0にする。coinc_plainは座標損失のみ、coinc_jointは標準化17条件のMSEを重み0.1で追加、coinc_auxはさらに標準化384物理応答のMSEを0.1で追加。近傍は幾何学的に近い3checkで、厳密な回路相互作用グラフではない。

最大80epochから検証で選択。学習は座標MSE、選択は中/強のユークリッド誤差平均。学習イベント数720の小規模条件であり、構造の性能限界ではない。

[補助教師の診断](auxiliary_diagnostic.csv)では、coinc_auxの物理応答RMSEは一定の学習平均対照から、弱0.00740→0.00282、中0.00614→0.00293、強0.05970→0.02824に減る。しかし位置誤差はSVRより大きい。「物理応答のMSEを減らす」ことと「中心の識別に必要な差を保つ」ことは同じではない。伝播条件の標準化RMSEは大部分が平均予測の水準付近であり、条件を精密に復元できたとは言えない。

| model | epoch | seconds |
|---|---|---|
| rates_plain_41 | 5 | 13.9646 |
| rates_plain_42 | 6 | 13.0132 |
| rates_plain_43 | 6 | 12.9932 |
| coinc_plain_41 | 5 | 12.9050 |
| coinc_plain_42 | 4 | 13.0487 |
| coinc_plain_43 | 4 | 12.9957 |
| coinc_joint_41 | 5 | 13.3165 |
| coinc_joint_42 | 4 | 13.0956 |
| coinc_joint_43 | 5 | 13.2205 |
| coinc_aux_41 | 5 | 13.9201 |
| coinc_aux_42 | 4 | 13.9484 |
| coinc_aux_43 | 4 | 13.9280 |

## 逆探索の限界と検証

forwardの検証確率RMSE：0.024936。選択温度：{'forward_inverse': 10.0, 'template_inverse': 1.0}。

強度別の補足診断は[forward_validation_diagnostic.json](forward_validation_diagnostic.json)。特に強帯では一定の平均確率を返す対照のRMSEも確認し、全強度RMSEだけで近似の成功とは扱わない。

forward教師はシミュレータの条件付きdetector確率だが、17パラメータは全ての物理状態を表さない。形状指数などの未入力条件・確率的空間模様が残る。16時間区間、192位置候補、近似MLPの誤差、相関を無視したbinomial複合スコアも制約である。逆探索が失敗しても物理モデルによる推定全体の不可能性とは扱わない。

## 再現と成果物

実装：`src/qp_ode_simulator/radical_data.py`、`radical_models.py`。リポジトリで環境変数`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`を設定し、以下を新しい出力先に順番に実行する。

```text
.venv/bin/python -m qp_ode_simulator.radical_data prepare NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data generate_fit NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data cache_fit NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_models train NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data generate_test NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_data cache_test NEW_ROOT
.venv/bin/python -m qp_ode_simulator.radical_models evaluate NEW_ROOT
.venv/bin/python examples/report_radical_pilot.py NEW_ROOT
```

シンドロームのみの推論入口（oracleは拒否）：

```text
.venv/bin/python -m qp_ode_simulator.radical_predict ROOT OBSERVED.npz NEW_OUTPUT.csv --model base_SVR
```

`OBSERVED.npz`の`base`は[shot,24check,4095round]の二値配列。出力は`x_mm,y_mm`。`--model`省略時は検証選択方式。高密度方式には`dense`48check、2パッチ方式には`base`と`right`各24checkを渡す。正解座標・発生時刻・伝播条件・物理応答は渡さない。既存出力CSVの上書きは拒否する。

prepareは兄弟ディレクトリの既存`feature_selection_20260910/protocol.json`を参照。新しい独立追試にはseedも新しくし、既存テスト再利用を避ける。

- [計画](PLAN_JA.md)、[固定した選択](selection.json)、[監査](audit.json)
- [全指標](metrics.csv)、[イベント別予測](predictions.csv)、[対応区間](paired.json)、[root別指標](seed_metrics.csv)
- 自動テスト131件通過：[tests.xml](tests.xml)。[真値なし推論の動作確認](inference_smoke.json)。既定モデルは自動置換していない。
