"""Render all predeclared comparisons without selecting on test outcomes."""
import argparse
from pathlib import Path
from xml.etree import ElementTree
import json
import pandas as pd


def table(df):
    def fmt(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def main(root):
    read=lambda f:json.loads((root/f).read_text())
    s=read('selection.json');a=read('audit.json');pairs=read('paired.json');m=pd.read_csv(root/'metrics.csv');seeds=pd.read_csv(root/'seed_metrics.csv')
    primary=next(p for p in pairs if p['primary']);ci=primary['ci95']
    primary_table=m[(m.horizon==4)&(m.domain=='ID')&m.model.isin(['SVR','REI_full','REI_valK','CNN','Ridge','Prior'])&m.group.isin(['all','weak','medium','strong'])]
    cols=['model','group','events','mean_mm','within_1mm','p90_mm','over_3mm','answer_rate']
    mainmodels=['SVR','REI_full','REI_valK','Ridge','CNN']
    broad=m[m.model.isin(mainmodels)&(m.group=='all')][['domain','horizon','model','mean_mm','within_1mm','p90_mm']]
    shape=m[(m.horizon==4)&(m.domain=='ID')&m.model.isin(['SVR','REI_full'])&m.group.isin(['weak_circular','weak_elliptical','medium_circular','medium_elliptical','strong_circular','strong_elliptical','ballistic','diffusive'])]
    sensitivity=m[(m.horizon==4)&m.model.isin(['SVR','SVR_noquiet','REI_full','REI_1024','REI_valK'])&(m.group=='all')]
    mainpairs=pd.DataFrame([dict(domain=p['domain'],horizon=p['horizon'],group=p['group'],model=p['model'],reference=p['reference'],difference_mm=p['difference_mm'],ci95=f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]",primary=p['primary'])
        for p in pairs if p['model']=='SVR' and p['reference'] in ['REI_full','REI_valK'] and p['group'] in ['all','weak','medium','strong']])
    seedtable=seeds[(seeds.horizon==4)&seeds.model.isin(['SVR','REI_full','REI_valK'])][['domain','seed','model','mean_mm','p90_mm']]
    tests=ElementTree.parse(root/'tests.xml').getroot()[0].attrib;assert tests['errors']=='0' and tests['failures']=='0'
    verdict='SVRの平均誤差が小さい側' if ci[1]<0 else 'REIの平均誤差が小さい側' if ci[0]>0 else '0を含み、平均差は不明確'
    text=f'''# SVR対適用版REI：同一観測時間での新規比較

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
差は **{primary['difference_mm']:+.4f}mm**、対応付きイベントbootstrap95%区間は **[{ci[0]:+.4f}, {ci[1]:+.4f}]mm**。{verdict}。

{table(primary_table[cols])}

REI_fullは元記録の4095内部round全てを使用。SVR/Ridgeも同じ先頭4095roundから特徴を作り、CNNも同じ記録の発火率を使う。表現・学習の有無は異なるが、使用可能な観測記録と時間はそろえている。

REI_valKは、検証だけで履歴長を選んだ適用版。主結果は全履歴比較だが、REIの履歴長調整後でも同じ結論かを副比較で示す。テストを見てREIが不利な履歴を選ぶことはしない。

## 実験の設計

- 新規学習{a['train_events']}・検証{a['validation_events']}物理イベント、各2ショット。
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
- 固有物理イベント{a['test_physical_events']}、条件別イベント数{a['condition_events']}。観測2.048/4.096/8.192msは同じ8ms記録の先頭を使う。

{table(broad)}

## 強度・形状・伝播法別（ID・4ms）

{table(shape[cols])}

## 履歴長と較正の感度（4ms・全強度平均）

{table(sensitivity[['domain','model','mean_mm','within_1mm','p90_mm','answer_rate']])}

REIの履歴選択：{ {h:v['selected'] for h,v in s['rei_history'].items()} }（round）。主REIは全履歴、REI_1024は従来の末尾1024固定。反復数1、空間相関倍率2を全てで固定。

SVR_noquietは平常較正を用いない対照。学習済みモデルとREIで事前に利用する情報量まで同じとは言わず、較正の有無と学習コストを明示する。

## 対応付き誤差差

差はmodel−reference、負がSVR側の改善。物理イベント内2ショットを平均してから10,000回bootstrap。1つの主比較以外は全て探索的95%区間で、多重比較補正や研究全体の反復探索に対する保証はない。

{table(mainpairs)}

## 生成root別（4ms）

{table(seedtable)}

## 公平性・監査

- 同じ条件内では全モデルに同じ物理イベント・同じショット・同じ観測時間を与える。REIとSVRに別のテストを与えない。
- 主比較・比較対象・設定選択規則をテスト生成前に固定。テストで一番良いモデルへの差し替えはしない。
- モデル・較正・前処理・ソースhashを評価時に確認。旧実験との生成seed重複なし。イベント分割で2ショットを分離しない。
- REIの元実装と、保存した発火率統計からの実装の数値一致を自動テスト。推定不能例を除外せず、事前固定した学習座標平均で埋め、回答率を併記。
- 自動テスト{tests['tests']}件通過。未来ラウンドの非参照、較正なし特徴、CNNの観測のみの入力も確認。

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
'''
    (root/'RESULT_JA.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
