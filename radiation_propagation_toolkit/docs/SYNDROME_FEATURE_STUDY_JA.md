# d5シンドローム特徴量の比較

既存の2,880イベント学習実験を基準に、入力表現と弱イベント重視の学習を比較する。
元データ・既存モデルを上書きしない。新規テストを生成する前にモデルを固定する。

## 実験の実行

リポジトリ直下で実行する。出力名が既存なら別名を使う。
親データは出力先の隣の `distance_growth_20260909` と、その親の距離比較データを使う。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.syndrome_features_study prepare ../localization_runs/syndrome_features_20260909
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.syndrome_features_study train ../localization_runs/syndrome_features_20260909 --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python examples/extend_syndrome_feature_controls.py ../localization_runs/syndrome_features_20260909
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python examples/refine_syndrome_features.py ../localization_runs/syndrome_features_20260909
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.syndrome_features_study freeze ../localization_runs/syndrome_features_20260909
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.syndrome_features_study generate ../localization_runs/syndrome_features_20260909 --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m qp_ode_simulator.syndrome_features_study evaluate ../localization_runs/syndrome_features_20260909
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/report_syndrome_features.py ../localization_runs/syndrome_features_20260909
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` は環境のROS用pytestプラグインによる無関係な依存エラーを避ける。
生成・学習には途中再開があるが、固定済み選択を上書きして同じテストで選び直してはいけない。

## 設定

- 学習2,880、検証360、今回新規テスト720物理イベント。各イベント2ショットを同じ分割に保つ。
- distance=5、24チェック×2,047内部roundの二値detector events。1ショットから座標を推定。
- 従来169特徴量、複数時間窓649、共分散補正649、同時/遅延ペア1,009、両者1,489を比較。
- 独立平常512ショットから平均・共分散・ペア頻度を較正。テスト自身から較正を作らない。
- 学習入力に真の時刻・形状・強度・座標を渡さない。弱イベントの学習重みだけに強度ラベルを使う。
- Ridge/SVR/ExtraTrees計45学習、座標平均、観測信号に応じて学習平均座標へ寄せるゲート。
- 主選択は通常検証平均。別途、通常検証が旧モデル+0.01 mm以内に収まる弱イベント重視設定を固定。
- 元の最良ハイパーパラメータ追加は探索中の変更として `validation_amendment.json` に記録。
- 検証段階の追加探索として、PCA 8学習と弱信号分類器3学習・24混合設定を `refinement_amendment.json` に記録。合計56学習。

## 判断するときの注意

- 同じ新規テスト上の凍結済み旧モデルを主基準にする。別のテスト集合の平均との単純な差を改善率としない。
- 弱イベントの誤差が減っても、一定座標を答える基準を上回らなければ位置情報を利用できた証拠は弱い。
- 平均誤差だけでなくp90と1 mm以内率も保存する。
- 対応付きbootstrapは物理イベント単位。区間は記述的で多重比較・学習集合の不確実性は含まない。
- baseline再学習のラベルはCSV経由で最大約9e-16 mmの丸め差がある。木の分岐の同点処理に影響し得るので、旧モデルのビット単位再現とは主張しない。主基準は保存済み旧モデルそのもの。
- 追加平常較正の資源は旧モデルより多い。通常ノイズとハードウェアは同じだが、純粋にネットワーク構造だけの比較ではない。
- REIは今回のシンドローム表現に対する適用版。原著条件での優越性は検証しない。
