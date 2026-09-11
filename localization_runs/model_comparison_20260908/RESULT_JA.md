# 同じシンドローム特徴での7モデル比較

## 結論

**7モデル群・20設定を比較しましたが、今回の未使用テストでは、Ridgeから確実に精度が上がったとは言えませんでした。**

検証ではSVRが選ばれました。新しいテストでは、強いイベントの誤差が小さくなる一方、
低強度と大きな誤差が悪化しました。Random Forestなどにも良い結果はありますが、
テストを見てそちらへモデルを選び直してはいません。既定モデル・既存artifactは変更していません。

## 比較の条件

- すべて同じ57特徴：固定区切りの前後発火率、独立平常較正との差、相対的な空間分布。
- 入力は観測シンドロームから計算。対象の真の時刻・強度・座標は使わない。
- 学習360イベント、検証120イベント、各2ショット。前回と同じ分割。
- モデルだけを変更し、特徴・区切り・較正値は再調整しない。
- 各モデル群の設定と全体の候補を検証平均位置誤差で固定してから、新しいテストを採点。
- 新規360イベント・各2ショット。通常300、保留した高強度楕円形60。全件テスト専用。
- 生成seed 20261010、測定seed 20261011、device seed 123。
- 過去4集合（1440、720、720、360イベント）とevent_uid・生成seed・測定seedの重複は0。

CNNも参考に残しましたが、CNNだけは生の時系列を入力するため、同じ57特徴での
モデル比較とは区別します。今回と前回のテストイベントは違うので、前回の2.823 mmと
今回の数値を直接比べて劣化したとは解釈しません。

## 試したモデル

| モデル群 | 設定数 | 主な設定 |
|---|---:|---|
| Ridge回帰 | 3 | 正則化alpha 10/100/1000 |
| MLP | 3 | 隠れ層32、または64→32、正則化変更 |
| Random Forest | 2 | 最小leaf数4/12 |
| ExtraTrees | 2 | 最小leaf数4/12 |
| HistGradientBoosting | 2 | 最大leaf数8/16 |
| RBF Kernel Ridge | 4 | 正則化×カーネル幅 |
| RBF SVR | 4 | C×カーネル幅 |

MLP/Random Forest/ExtraTreesは3seedの**座標平均**。その他は決定的な1モデルです。
MLPのepoch選択にも同じ外部のイベント単位検証集合を使い、内部のショット単位ランダム
分割は使いません。標準化は学習集合だけでfitしています。
モデル容量・探索数・計算量まで完全にそろえた比較ではなく、小さな固定探索です。
GNN、Transformer、新しい生時系列CNNの構造探索は今回行っていません。

## 新しいテストの結果

各モデル群の**検証で選んだ設定**です。テストで最良の設定を後から抜き出した表ではありません。

| モデル | 通常条件の平均誤差 | 未学習の強い楕円形 |
|---|---:|---:|
| Ridge | 2.995 mm | 2.028 mm |
| MLP | 3.014 mm | 2.166 mm |
| Random Forest | 2.970 mm | 2.014 mm |
| ExtraTrees | 2.988 mm | 1.993 mm |
| HistGradientBoosting | 3.022 mm | 1.940 mm |
| Kernel Ridge | 2.984 mm | 1.949 mm |
| **SVR（検証での全体選択）** | **2.996 mm** | **1.938 mm** |
| CNN単体（参考） | 3.004 mm | 2.167 mm |
| CNN 3モデル平均（参考） | 3.002 mm | 2.157 mm |

Random Forestの通常平均はRidgeより0.025 mm小さかったものの、対応付き95%区間は
[-0.066, +0.015] mmで0をまたぎます。
検証で選ばれたSVRのRidgeとの差は、通常+0.002 mm [-0.036, +0.045]、
強い楕円形-0.090 mm [-0.238, +0.057]でした。

**SVRの強い楕円形への改善傾向はありますが、今回の60イベントでは確定できません。**
その他のモデル群も、Ridgeに対する平均改善の区間は0をまたぎました。
MLPの強い楕円形では逆に誤差が増え、差+0.138 mm、区間[+0.019, +0.269]でした。
複数比較の区間は記述的なもので、多重性補正済みの確証試験ではありません。

## 弱い信号と大きな誤差

通常条件の強度別結果：

| 強度 | Ridge | SVR | Random Forest | 一定座標 |
|---|---:|---:|---:|---:|
| 低 | 3.500 mm | 3.582 mm | 3.440 mm | 3.377 mm |
| 中 | 3.045 mm | 3.049 mm | 3.041 mm | 3.195 mm |
| 高（円形） | 1.882 mm | 1.717 mm | 1.887 mm | 3.357 mm |

SVRは高強度で良くなる一方、低強度では悪化しました。低強度は引き続き一定座標を
答える基準より良くありません。単純なモデル変更で解決していません。

通常条件の90 percentile誤差：

| 方法 | 90 percentile誤差 |
|---|---:|
| Ridge | 4.557 mm |
| SVR | 4.809 mm |
| Random Forest | 4.314 mm |
| CNN単体（参考） | 4.281 mm |

SVRの2 mm以内の割合はRidgeの22.5%から26.0%に増えますが、
一部の大きな誤差も増えています。すべての指標で優れたモデルではありません。
Random Forestは外れ値側を抑える候補ですが、テスト結果に基づく新たな選択として
扱う必要があるため、ここで主モデルに差し替えてはいません。

## 保存と再現

検証の全体選択は `svr_17`：StandardScaler＋x/y別のRBF SVR、C=10、gamma=.001、
epsilon=.1 mm。出力は(x_mm, y_mm)のみ。イベントの有無や発生ラウンドは出しません。
選択済み候補だけを [selected_candidate](selected_candidate/model.json) に保存しました。

```python
from qp_ode_simulator.adaptive_localization import predict
xy_mm = predict(
    "../localization_runs/model_comparison_20260908/selected_candidate",
    "../localization_runs/model_comparison_20260908/fresh_dataset/events_00000000_00000036.npz",
)
```

自動テスト61件が通過。前回Ridgeの予測再現差は0、MLPの別プロセス再読込も確認済み。
MLPの保存参照を安定したモジュール名にする修正後、同じ条件を再実行して検証値が
すべて一致することも確認しました。初回の保存物は `experiment_initial_serialization` に
残していますが、利用するのは `experiment` または `selected_candidate` です。
選択候補の分離前後でも予測差は0です。

## 今回わかったこと

1. CNN以外でも精度は出る。MLP・木・カーネル法は同じ特徴で比較できる状態になった。
2. ただし、複雑なモデルに変えるだけで、Ridgeを一貫して上回る結果ではなかった。
3. 強いイベントの位置推定と、弱いイベントでの過剰な位置変動を抑えることは両立が課題。

次に狙うなら、弱い信号での不確かさと外れ値対策、特徴量の寄与の切り分けを優先します。
今回のテスト結果を設計に使う次のモデルには、別の未使用確認集合が必要です。
同一シミュレータ・固定配置での結果で、実機、別の符号距離、別デバイス、GNNの優劣を
判断したものではありません。

## 保存物

- [新規テスト前の固定方針](PROTOCOL_JA.md)、[探索設定](experiment/grid.json)
- [検証結果](experiment/validation.csv)、[MLPの学習履歴](experiment/mlp_histories.json)
- [選択モデルと各family代表](experiment/model.json)、[推論用候補](selected_candidate/model.json)
- [新規生成設定](fresh_dataset/dataset_manifest.json)、[確認評価のハッシュ](confirmation/evaluation.json)
- [全テスト結果](confirmation/metrics.json)、[対応付き比較](confirmation/paired.json)、[全予測](confirmation/predictions.csv)
- [仕様・再現コマンド](../../radiation_propagation_toolkit/docs/MODEL_COMPARISON_JA.md)
