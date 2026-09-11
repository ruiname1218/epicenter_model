# radiation_propagation_toolkit ローカル分析

分析日: 2026-09-06。対象: https://github.com/ruiname1218/radiation_propagation_toolkit

- 取得先: `/home/rui/Downloads/epicenter_model/radiation_propagation_toolkit`
- ブランチ: `main`
- コミット: `5de1632983f3a0d5bfdc907802f98ce5fb29e889`
- Python 3.10 の `.venv` を作成し、`pip install -e '.[all]'` を実施。
- ソースコードの変更なし。実行結果はリポジトリ外の `analysis_outputs/` に保存。

## 評価

超伝導量子ビットで生じる相関バーストを、発生位置の正解ラベル付きで生成し、量子誤り訂正（QEC）の検出器データまで変換できる研究用ツール。発生位置推定の合成データ作成や、ノイズ条件を変えた QEC の感度評価に使える構成になっている。

ただし、物理的な粒子輸送やチップ内の準粒子拡散そのものを解いているわけではない。合成データ上の推定精度、実測緩和データとの整合性、実機での論理誤り率は別々に評価する必要がある。

## 処理の構成

| 段階 | 主なコード | 内容 |
|---|---|---|
| 座標・設定 | `layouts.py`, `configuration.py` | 任意の2次元座標、JSON設定の継承 |
| イベント生成 | `simulator.py` | 発生位置、円・楕円形状、到達時間、減衰範囲、強度をサンプル |
| 局所応答 | `simulator.py` | 準粒子 ODE を積分し、T1/T2・応答確率を生成 |
| QEC回路 | `stim_qec.py` | Stim回路へ Pauli ノイズを挿入し、測定値・検出器・論理観測量を生成 |
| 簡易シンドローム | `syndrome.py` | 代数的な軽量代理モデル |
| 公開API | `api.py` | 実行結果の型、設定のコピー、NPZ/CSV保存 |
| 検証 | `propagation_validation.py`, `channel_validation.py` | 伝播パラメータの回収、厳密チャネルとの比較 |

各量子ビットで `dx/dt = g - s*x - r*x²` を解く。区間内の生成率を平均して一定とみなし、その区間は解析解で進める。時間依存する生成率全体に対する厳密解という意味ではない。

伝播は距離 `d` に対して ballistic が `d/v`、diffusive が `d²/D`。後者も到達時刻の規則であり、空間結合した拡散方程式ではない。局所ODE間の準粒子移動やエネルギー保存はモデル化されていない。

出力の `true_parameters.csv` に発生位置などが入り、`simulated_events.npz` に観測・密度・T1/T2等が入る。Stim出力には検出器イベント、測定記録、論理観測量、回路テンプレート、実行情報が含まれる。

## 今回の実行確認

| 項目 | 結果 |
|---|---|
| 同梱テスト | 14 passed、スキップなし |
| 通常CLI | 2イベント × 2,500時刻 × 25量子ビット、保存成功 |
| Stim CLI | 2イベント、17量子ビット、512ラウンド、2ショット/イベント、4,096検出器 |
| PyMatching | 別途8ラウンド・4ショットで有効化し、予測と失敗フラグの生成を確認 |
| 伝播検証 | 120ケース実行、全フィット成功 |
| チャネル検証 | 112ケース実行、プロジェクト所定の基準に対する通過率17.9% |

最初のpytest実行は、シェルのROS環境から `launch_testing` が自動ロードされ、仮想環境に `yaml` がないため収集前に失敗した。これは本リポジトリのテスト失敗ではない。`PYTHONPATH` を外し外部プラグインの自動読込を停止して14件すべて通過した。

伝播検証の再実行結果:

| 指標 | Ballistic | Diffusive |
|---|---:|---:|
| 伝播パラメータ平均絶対百分率誤差 | 7.761% | 0.683% |
| 発生位置誤差の中央値 | 0.1548 mm | 0.01193 mm |
| 発生位置誤差の90パーセンタイル | 0.3616 mm | 0.02363 mm |

README掲載値を丸め精度で再現した。ただし、このベンチマークは `temporal_backend='empirical'` の共有空間・波面層を使用する。QP-ODE応答、有限ショット観測、Stim検出器を経由した発生位置推定の実証ではない。生成・推定で伝播則の種類は既知としている。

## 注意すべき点

1. **強い緩和でのPauli近似。** 今回再実行した3量子ビットの検証では、T1=100 µsで75%、50 µsで50%、30 µs以下で0%が所定基準を通過した。厳密な振幅減衰が持つ基底状態への偏りをPauli近似が失うため、強いイベント時の誤り率を定量評価する際の主要な制約となる。これはコードのテスト失敗ではなく、近似の検証結果。基準はこのプロジェクト独自で、全表面符号への誤差保証ではない。

2. **回路内の時間分解能。** `generate_circuit_syndromes` は各ラウンドの逆T1/逆T2を積分平均し、その同じ有効値を各ゲート区間に配る。ゲート長の違いは反映するが、ゲートごとに元の場を積分し直す実装ではない。既定の物理場刻みは2 µs、QECラウンドは1 µsであり、それより速い変動の精度は別途刻み幅の収束確認が必要。

3. **実機検証の再現範囲。** Google比較の集計・図・ハッシュは同梱されるが、生データ抽出・モデル選択コードと評価用プロファイルはこの配布物だけでは完結しない。今回Google生データとの再比較は実施していない。`docs/VALIDATION.md` の報告では時間応答に一定の整合性がある一方、空間分布の転移は53件中26件の95%領域カバーに留まる。FASTは実イベント3件で、未閲覧の外部検証でもない。

4. **発生位置ラベルの座標名。** 座標入力は `(x_mm, y_mm)` だが、保存ラベルは `epicenter_row`, `epicenter_col`。実装上はそれぞれ座標配列の第0成分・第1成分に対応する。通常の画像の行・列を想定して扱うと軸を取り違える。

5. **規模拡大。** `simulate` はイベント×時刻×量子ビットの配列を全件メモリ確保する。既定の診断配列8種類だけでfloat32が32バイト/要素、観測・確率・baseline確率を含め約37バイト/要素。1,000イベント×25,000時刻×100量子ビットなら、それらだけで約92.5 GB（十進）となり、作業配列とQEC出力はさらに増える。大規模学習データにはバッチ実行・分割保存が必要。

6. **保守性。** 固定ハードウェア、乱数系列の分離、設定コピー、イベント依存デコーダへの明示的ガードは良い設計。一方、主要モジュールは約1,600行規模で、同梱テスト14件は基本経路中心。今回の成功は全設定分岐や数値精度の保証ではない。`setup.py` の互換用CLI一覧には `qp-ode-validate-propagation` がなく、`pyproject.toml` と不一致があるが、今回の通常インストールではCLIが利用できた。

## 手元で再実行

```bash
cd /home/rui/Downloads/epicenter_model/radiation_propagation_toolkit
source .venv/bin/activate
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
qp-ode-simulate --n-events 2 --output qp_ode_output
qp-ode-stim --n-events 2 --stim-config examples/stim_smoke.json --output qp_ode_stim_output
qp-ode-validate-propagation --output propagation_validation_output
qp-ode-validate-channel --output channel_validation_output
```

今回の成果物は `analysis_outputs/simulation/`, `analysis_outputs/stim/`, `analysis_outputs/propagation/`, `analysis_outputs/channel/`。伝播とチャネルのフォルダにはCSV、JSON、図がある。

発生位置推定へ進めるなら、次の評価単位は「QP-ODE → 有限ショットのStim検出器 → 推定器 → 未学習イベントの位置誤差」。伝播層単独の既存ベンチマークと、この観測込みの評価を分けると改善点が明確になる。
