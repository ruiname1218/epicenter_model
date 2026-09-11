# QEC channel検証結果

同じT1/T2と1 us QEC cycleを使い、非unitalなexact generalized amplitude dampingと、
Stim経路で使用するunitalなPTGADを3-qubit反復parity-checkで比較した。全measurement
branchを密度行列で列挙しているため、結果にshot noiseはない。

| T1 (us) | 全case通過 | 通過率 | worst detector TV | worst detector p error | worst memory error |
|---:|:---:|---:|---:|---:|---:|
| 100 | no | 75.0% | 0.0347 | 0.0196 | 0.0148 |
| 50 | no | 50.0% | 0.0678 | 0.0384 | 0.0291 |
| 30 | no | 0.0% | 0.1097 | 0.0624 | 0.0476 |
| 20 | no | 0.0% | 0.1586 | 0.0906 | 0.0696 |
| 10 | no | 0.0% | 0.2850 | 0.1648 | 0.1296 |
| 5 | no | 0.0% | 0.4673 | 0.2753 | 0.2256 |
| 3 | no | 0.0% | 0.6154 | 0.3682 | 0.3161 |

判定thresholdはこのprojectのscreening基準であり、普遍的な物理許容値ではない。
強いburstで失敗する場合、PTGADを実機忠実なchannelと呼ばず、高速stress-test近似として
使用範囲を限定する。実機精度の最終確認にはmeasured detector dataが必要である。

- `qec_channel_validation.csv`: 全parameter/caseの数値
- `qec_channel_validation.png`: T1に対する近似誤差
- `qec_channel_validation_summary.json`: 判定と制約
