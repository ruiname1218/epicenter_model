# QP-ODE Simulator Toolkit 日本語ガイド

このフォルダは、元の実験リポジトリから再利用可能なQP-ODEシミュレータだけを独立させたものです。
元の実験フォルダは削除・変更せず、Googleデータ解析、試行錯誤中の図、巨大な中間配列を含めていません。

## 推奨パイプライン

```text
eventの5軸
  shape / apparent propagation / range / epicenter / strength
        ↓
各circuit qubit位置のsource proxy g(t,q)
        ↓
dx_qp/dt = g - s*x_qp - r*x_qp^2
        ↓
時刻・qubit別のT1/T2
        ↓
各gate区間のpX/pY/pZ
        ↓
Stimのreset / gates / measurement / detector
        ↓
syndrome、detection event、logical flip
```

自然なsyndrome生成には`qp-ode-stim`を使います。`qp-ode-syndrome`は高速ですが、ancilla回路を
実行しない代数proxyなので、用途を分けています。

## インストール

```bash
cd /home/rui/Downloads/qp_ode_simulator_toolkit
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[all]'
```

## 最小実行

QP場だけを生成:

```bash
qp-ode-simulate --n-events 2 --output qp_ode_output
```

小さいStim circuitまで実行:

```bash
qp-ode-stim --n-events 2 \
  --stim-config examples/stim_smoke.json \
  --output qp_ode_stim_output
```

テスト:

```bash
pytest
```

## シンドロームからの発生位置推定

`qp-ode-localize` で、1ショットのStim検出器イベントから `(x_mm, y_mm)` を推定できます。
教師あり学習、元イベント単位の分割、重心法との比較、正解不要の推論を用意しています。
設定・実行方法・制約は[LOCALIZATION_JA.md](LOCALIZATION_JA.md)を参照してください。

## Propagation自己整合性評価

公開用の軽量benchmarkとして、既知の真値から生成したballistic/diffusive場をfitし直す
`qp-ode-validate-propagation`を用意しています。

```bash
qp-ode-validate-propagation --output propagation_validation_output
```

30 us RMSの到着ジッターと0.1 log-SDのqubit応答ばらつきを加えた各60試行で、
ballistic速度の平均絶対百分率誤差は7.8%、diffusion係数は0.7%、空間減衰長は
それぞれ3.2%と2.3%でした。震源位置の中央値誤差は0.155 mmと0.012 mmです。
図と各panelの説明は[VALIDATION.md](VALIDATION.md)にあります。

これは設定した伝播則を実装が保っているかを調べるfield-level自己整合性評価です。
実機のradiation transportへの一致や、観測burstがradiation由来であることを証明する
評価ではありません。後者には外部particle detectorまたは制御線源でtagした実機eventが必要です。

## Google実機データとの比較

McEwen et al.の公開Google RReCS実測データとも比較しています。これはexcited stateを
準備し、1 us後のrelaxationを読むデータであり、surface-code syndromeではありません。
MAINは元recording単位でcalibration/validation/testを分離し、test 16 recordingから得た
53 strong eventsを評価しました。

| test指標 | Google-calibrated QP-ODE |
|---|---:|
| 絶対時間応答の95% coverage | 49/53 (92.5%) |
| 絶対nearest RMSE | 0.0365 |
| 正規化shapeの95% coverage | 48/53 (90.6%) |
| shape nearest RMSE | 0.1510 |
| FAST apparent speed / arrival shape coverage | 15/15 / 15/15 |
| FASTからMAINへのzero-shot空間map coverage | 26/53 (49.1%) |

FASTは3 us samplingなので、MAINでは分解できない初期の応答拡大も別途評価しています。
次の図は実測3 eventと256 simulationの時間・空間profile比較です。上段は時間包絡、
kNN外れ値percentile、nearest spatial RMSE、下段は各実eventと最も近いsimulationの
時間波形です。

![Google FAST実測eventとwavefront simulationの比較](validation/google_fast/fidelity_evaluation.png)

ただしこの図は、同じ3 eventを見て範囲を較正した後のin-sample posterior-predictive
checkです。独立性が比較的高い評価は、1 eventずつ外して候補を選ぶ表中の15評価ですが、
これも実質は3 event x 5 simulation seedであり、15個の独立な実測eventではありません。

時間応答は実機と中程度に整合しますが、empirical backendより改善していません。FASTは
実eventが3件だけで、15評価は3件を5 seedで繰り返したものです。またMAINのzero-shot
空間transferは弱いため、「時間方向はそこそこ、wavefrontは暫定的、空間汎化は未証明」
という結論です。図と詳しい制約は[VALIDATION.md](VALIDATION.md)を参照してください。

## 重要な注意

- Googleデータがなくても動く汎用layout版です。
- Googleデータは元の実験リポジトリで妥当性確認に使い、このtoolkitの実行時入力にはしません。
- QP-ODEの`g(t,q)`は局所source proxyで、入射粒子energyそのものではありません。
- `apparent_speed`はqubit応答のthreshold拡大速度で、基板中のphonon group velocityではありません。
- StimのPauli twirlは大規模生成に便利ですが、強いT1低下ではexact amplitude dampingとの差が増えます。
- 実機syndromeを定量予測するには、対象QPUのT1/T2、gate時間、baseline circuit noise、実detector
  データによる追加較正が必要です。

詳しい構造は[ARCHITECTURE.md](ARCHITECTURE.md)、限界は
[MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md)、channel評価は[VALIDATION.md](VALIDATION.md)を参照してください。
