# 新規テスト前に固定したモデル比較

2026-09-08 JST、新規データ生成中、テスト予測・誤差は未計算。

前回のadaptive_features実験で選んだ固定57特徴を変更しない。
平常較正値・区切り位置・学習360イベント・検証120イベント・2ショットを維持する。
入力は8チェック×2047時点のシンドロームから計算した同じ特徴。
真の発生時刻や強度を推論入力に与えない。

比較は7モデル群・20設定：Ridge 3、MLP 3、Random Forest 2、ExtraTrees 2、
HistGradientBoosting 2、RBF Kernel Ridge 4、RBF SVR 4。
すべて同じイベントで学習・検証。各familyの代表と全体の候補を検証平均位置誤差で固定。

MLPは外部のイベント単位検証集合だけでepochを選ぶ。内部のショット単位ランダム
検証分割は使わない。入力・出力の標準化は学習集合だけ。
MLP/Random Forest/ExtraTreesはseed 41/42/43の座標平均。
その他は決定的な設定1モデル（SVR・HGBはx/y別の回帰器）。
探索数やモデル容量は完全に等しくないため、各モデルの最良性能を網羅した比較ではない。

検証での全体選択は **svr_17（RBF SVR、C=10、gamma=0.001、epsilon=0.1 mm）**。
検証平均誤差2.832037 mm。Ridgeは2.851916 mm。

family代表：

| family | 固定候補 |
|---|---|
| Ridge | fixed_ridge_100.0 |
| MLP | mlp_03 |
| Random Forest | random_forest_06 |
| ExtraTrees | extra_trees_08 |
| HistGradientBoosting | hist_gradient_boosting_10 |
| Kernel Ridge | kernel_ridge_12 |
| SVR | svr_17 |

CNN単体・3モデル座標平均も参考に残す。ただしCNNだけ生の時系列が入力なので、
同じ特徴を使ったモデル間比較とは区別する。

新規360イベントを全件テスト専用に生成。通常300・保留した高強度楕円形60。
36区分各10イベント、生成seed 20261010、測定seed 20261011、device seed 123。
主比較は選択済みSVR対前回のRidge。CNN対比も参考に報告。
平均誤差だけでなく、低強度・90 percentile誤差を確認する。
対応付き区間はショットをイベント内でまとめたbootstrap。多重性補正済み確証試験ではない。
新テストを見てfamily代表や全体選択を変えない。
