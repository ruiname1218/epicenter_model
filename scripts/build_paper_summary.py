"""Bilingual, source-linked research brief. All charts use saved CSV results."""
from pathlib import Path
import csv
import json
import hashlib
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results'
OUT = ROOT / 'paper'
OUT.mkdir(exist_ok=True)
FONT = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
fm.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': ['DejaVu Sans', fm.FontProperties(fname=FONT).get_name()],
                     'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#26364a',
                     'text.color': '#26364a', 'axes.edgecolor': '#b4bdc7'})
BLUE, GRAY, TEAL, ORANGE = '#2066a8', '#929ba8', '#258878', '#d88730'
SOURCES = ['rei_fair_20260911/metrics.csv', 'rei_fair_20260911/paired.json',
           'distance_main_svr_20260923/metrics.csv',
           'feature_ablation_main_aligned_20260923/metrics.csv']
def read_csv(name):
    with (RUNS / name).open() as f:
        return list(csv.DictReader(f))
MAIN = read_csv(SOURCES[0])
PAIRS = json.loads((RUNS / SOURCES[1]).read_text())
DIST = read_csv(SOURCES[2])
FEAT = read_csv(SOURCES[3])
def row(data, **keys):
    found = [r for r in data if all(str(r[k]) == str(v) for k, v in keys.items())]
    assert len(found) == 1, (keys, len(found))
    return found[0]
def metric(model, group='all', domain='ID', field='mean_mm'):
    return float(row(MAIN, horizon=4, model=model, group=group, domain=domain)[field])
def pair(group='all', domain='ID'):
    return row(PAIRS, horizon=4, model='SVR', reference='REI_full', group=group, domain=domain)

def build(lang):
    ja = lang == 'ja'
    def tr(j, e): return j if ja else e
    def txt(fig, y, j, e, size=11, color='#26364a'):
        raw = tr(j, e)
        lines = []
        for p in raw.split('\n'):
            if ja:
                # East Asian text: wrap by display width, preserving ASCII identifiers.
                import unicodedata
                buf, width = '', 0
                for ch in p:
                    w = 2 if unicodedata.east_asian_width(ch) in 'WF' else 1
                    if width + w > (104 if size <= 10 else 94):
                        lines.append(buf); buf, width = '', 0
                    buf += ch; width += w
                lines.append(buf)
            else:
                lines.extend(textwrap.wrap(p, width=102 if size <= 10 else 91) or [''])
        return fig.text(.09, y, '\n'.join(lines), va='top', fontsize=size,
                        linespacing=1.65, color=color)
    def page(n, j, e, sub_j, sub_e):
        f = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        f.text(.09, .954, 'RESEARCH BRIEF  /  2026-09-12', fontsize=9, color=TEAL)
        f.text(.09, .905, tr(j, e), fontsize=20, va='top')
        txt(f, .859, sub_j, sub_e, 10)
        f.text(.09, .032, tr('シミュレーション研究｜既存結果の整理・新規学習なし',
                             'Simulation study | Existing results; no new training'), fontsize=8, color=GRAY)
        f.text(.91, .032, f'{n} / 4', ha='right', fontsize=8, color=GRAY)
        return f
    def style(ax, label=True):
        ax.set_axisbelow(True); ax.grid(axis='y', color='#e8ecf0')
        ax.tick_params(labelsize=9)
        if label: ax.set_ylabel(tr('平均位置誤差 [mm] ↓', 'Mean localization error [mm] ↓'), fontsize=9)
    def source(f, text):
        f.text(.09, .064, text, fontsize=7, color=GRAY)
    def finish(pdf, f, n):
        pdf.savefig(f)
        f.savefig(OUT / f'{lang}_page_{n}.png', dpi=115)
        plt.close(f)

    with PdfPages(OUT / f'research_summary_{lang}.pdf', metadata={
        'Title': tr('シンドロームからの放射線発生位置推定：研究結果概要',
                    'Radiation epicenter localization from syndrome records'),
        'Subject': 'Matched adapted-REI comparison, supporting evidence, and limitations',
        'Author': 'Radiation localization research project'}) as pdf:
        f = page(1, 'シンドロームから発生位置を推定する',
                 'Localizing radiation events',
                 '主結果：同じ観測記録を使い、SVRは適用版REIより全強度平均の位置誤差を低減。',
                 'Primary result: SVR lowers overall mean error versus adapted REI using the same records.')
        txt(f, .802,
            '入力：surface codeの二値detector events（測定間のパリティ変化） → 出力：座標 (x, y)\n単一イベントが存在する固定窓を評価。イベント検出・発生時刻推定は対象外。',
            'Input: binary surface-code detector events → Output: epicenter coordinates (x, y).\nThe window contains one event. Event detection and onset-time estimation are not evaluated.', 10)
        ax = f.add_axes([.12, .445, .80, .255])
        groups = ['all', 'weak', 'medium', 'strong']; x = np.arange(4)
        for i, (m, label, c) in enumerate([('REI_full', 'Adapted REI', GRAY),
                ('SVR', 'SVR', BLUE), ('Ridge', 'Ridge', TEAL), ('CNN', 'CNN', ORANGE)]):
            ys = [metric(m, g) for g in groups]
            bars = ax.bar(x + (i-1.5)*.19, ys, .18, label=label, color=c)
            ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)
        ax.set_xticks(x, tr(['全体', '弱', '中', '強'], ['All', 'Weak', 'Medium', 'Strong']))
        ax.set_ylim(0, 3.85); style(ax)
        ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.19), frameon=False, fontsize=9)
        txt(f, .404,
            '全体平均 2.571 → 2.214 mm（13.9%減）。SVR−REIの95%区間：−0.414〜−0.301 mm。\n強の1 mm以内率 23.9% → 77.8%。ただし中は7.5%、弱は改善しなかった。',
            'Overall mean: 2.571 → 2.214 mm (13.9% lower). SVR−REI, 95% CI: −0.414 to −0.301 mm.\nStrong events within 1 mm: 23.9% → 77.8%. Medium: only 7.5%; weak events did not improve.', 10)
        txt(f, .31,
            '比較の設計', 'Matched evaluation', 13)
        txt(f, .276,
            'd5 rotated surface-code memory-Z：49 qubit・24 check、1 µs/round。\n学習864・検証180・通常テスト540物理イベント、各2ショット。テストは各強度180件。\n約4 msの同じ4095内部roundを使用。前処理と設定はテスト生成前に固定。\n学習教師は座標のみ。CNNは3初期化の座標平均。モデル構造・探索予算は同一ではない。\n誤差は座標のユークリッド距離。全体平均は3強度を均等に重み付けした値。\n全体平均を事前主比較とし、イベント単位の対応付きbootstrap（10,000回）で評価。',
            'd5 rotated surface-code memory-Z: 49 qubits, 24 checks, 1 µs per round.\nTrain / validation / nominal test: 864 / 180 / 540 physical events; two shots each, 180 tests per band.\nAll methods use the same 4,095 internal rounds (~4 ms); settings are frozen before test generation.\nOnly coordinates supervise learning. CNN averages three initializations; search budgets differ.\nError is Euclidean coordinate distance. The overall mean gives equal weight to the three bands.\nOverall mean is the prespecified primary contrast; paired event bootstrap uses 10,000 draws.', 10)
        source(f, '[1] rei_fair_20260911  |  Figure: group means; CI refers to the primary paired difference.')
        finish(pdf, f, 1)

        f = page(2, '改善する条件と、改善しない条件', 'Where the improvement holds',
                 '最新比較の追加分析。全体平均の主比較以外は探索的結果として扱う。',
                 'Additional analyses from the matched study. All contrasts except the primary are exploratory.')
        ax = f.add_axes([.14, .57, .76, .22])
        labels = tr(['通常', '回路雑音2倍', '未知の遅い伝播'], ['Nominal', 'Circuit noise ×2', 'Unseen slow propagation'])
        for i, (m, c, label) in enumerate([('REI_full', GRAY, 'Adapted REI'), ('SVR', BLUE, 'SVR')]):
            bars = ax.bar(np.arange(3)+(i-.5)*.27,
                          [metric(m, domain=d) for d in ['ID', 'noise2', 'slow']], .25, color=c, label=label)
            ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=4)
        ax.set_xticks(np.arange(3), labels); ax.set_ylim(0,3.1); style(ax)
        ax.legend(frameon=False, ncol=2, fontsize=9)
        txt(f, .52,
            '雑音2倍でも平均改善を維持。未知の遅い伝播では差（SVR−REI）−0.053 mm、\n95%区間 −0.147〜+0.042 mmで優位性は不明確。CNNは2.452 mm（点推定の参考値）。',
            'The mean benefit persists with doubled circuit noise. Under unseen slow propagation,\nSVR−REI is −0.053 mm (95% CI −0.147 to +0.042): no clear advantage. CNN: 2.452 mm (point estimate).', 10)
        txt(f, .431, '強いイベント：円形と楕円形の両方で改善',
            'Strong events: improvement for both shapes', 13)
        txt(f, .39,
            '円形：REI 1.556 → SVR 0.653 mm。楕円形：1.568 → 0.832 mm。\n両形状を学習に含むため、「未知形状に汎化した」という結果ではない。',
            'Circular: REI 1.556 → SVR 0.653 mm. Elliptical: 1.568 → 0.832 mm.\nBoth shapes are in training; this is not evidence of generalization to an unseen shape.', 10)
        txt(f, .314, '平均改善は、大外れ全般の改善ではない',
            'Lower mean error does not imply a better tail', 13)
        txt(f, .276,
            '通常・約4 msの全体p90：REI 4.071 → SVR 4.210 mmと悪化（p90：誤差の90%点）。\n弱の平均も3.198 → 3.266 mmで改善なし。失敗条件を除外せず報告する。',
            'Nominal ~4 ms overall p90 (90th error percentile): REI 4.071 → SVR 4.210 mm, worse.\nWeak-event mean: 3.198 → 3.266 mm, with no improvement. These cases remain in the report.', 10)
        txt(f, .193,
            '条件の範囲：通常540イベントとその回路雑音2倍版、独立の遅い伝播216イベント。\n固有テストイベントは計756。雑音2倍はゲート・測定・リセット雑音のみ（T1/T2は不変）。\n遅い伝播：速度2–6 m/s対学習12–40、拡散係数0.5–2 mm²/ms対学習5–20。\nモデル・通常較正は固定。固定窓内に波が届かない影響も、この条件差に含まれる。',
            'Tests: 540 nominal events and their noise-shifted versions, plus 216 independent slow events;\n756 unique physical events. Only gate, measurement and reset noise are doubled, not T1/T2.\nSlow speed: 2–6 vs training 12–40 m/s; diffusion: 0.5–2 vs training 5–20 mm²/ms.\nModels and nominal calibration stay fixed. Later arrival within the fixed window also affects results.', 9)
        source(f, '[1] rei_fair_20260911  |  All results on this page use ~4 ms.')
        finish(pdf, f, 2)

        f = page(3, '符号距離と特徴量の影響', 'Code distance and feature choice',
                 '主比較と同じ観測プロトコル・SVR設定を用いた補助実験。',
                 'Supporting studies using the main observation protocol and SVR settings.')
        ax = f.add_axes([.14, .56, .76, .22])
        for group, color, label, marker in [('all', BLUE, tr('全体', 'All'), 'o'),
                                             ('weak', GRAY, tr('弱', 'Weak'), 's'),
                                             ('medium', GRAY, tr('中', 'Medium'), '^'),
                                             ('strong', GRAY, tr('強', 'Strong'), 'D')]:
            vals = [float(row(DIST, distance=d, group=group)['mean_mm']) for d in [3,5,7]]
            ax.plot([3,5,7], vals, marker=marker, linewidth=2.0, color=color, label=label)
            for d, v in zip([3,5,7], vals): ax.annotate(f'{v:.2f}', (d,v), xytext=(0,8), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks([3,5,7]); ax.set_ylim(0,3.65); style(ax)
        ax.set_xlabel(tr('surface-code distance', 'Surface-code distance'), fontsize=9)
        ax.legend(frameon=False, ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(.5,1.18))
        txt(f, .50,
            '符号距離を3→7に増やすと、全体誤差は2.419→2.079 mm、強イベントは1.115→0.632 mmに低下した。\n中イベントはd=5で最良、弱イベントの改善は小さい。これは観測範囲とcheck数も同時に増える比較である。',
            'Increasing code distance from 3 to 7 lowers overall error from 2.419 to 2.079 mm and strong-event error from 1.115 to 0.632 mm.\nMedium events are best at d=5; weak events improve little. The comparison also increases the observed area and check count.', 9)
        ax = f.add_axes([.14, .245, .76, .16])
        feature_order = [('F0_relative_spatiotemporal', tr('空間+時間', 'Spatial + temporal'), BLUE),
                         ('F1_spatial_only', tr('空間のみ', 'Spatial only'), BLUE),
                         ('F2_temporal_only', tr('時間のみ', 'Temporal only'), GRAY),
                         ('N1_shuffled_checks', tr('位置シャッフル', 'Shuffled positions'), GRAY),
                         ('N2_shuffled_time', tr('時間シャッフル', 'Shuffled time'), GRAY)]
        vals = [float(row(FEAT, model=name, group='all')['mean_mm']) for name, _, _ in feature_order]
        bars = ax.bar(np.arange(len(vals)), vals, color=[c for _,_,c in feature_order], width=.62)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
        ax.set_xticks(np.arange(len(vals)), [label for _,label,_ in feature_order], rotation=18, ha='right', fontsize=8)
        ax.set_ylim(0,3.65); style(ax)
        txt(f, .182,
            '空間情報を除くと全体誤差は約3.25–3.31 mmに悪化し、空間情報が重要であることを示す。\nこの特徴量比較は主比較と同じテストを使うが、特徴量アブレーションとして探索的に報告する。',
            'Removing spatial information increases overall error to about 3.25–3.31 mm, indicating that spatial information is important.\nThis ablation uses the same test set as the primary comparison but remains exploratory.', 8)
        source(f, '[2] distance_main_svr_20260923  |  [3] feature_ablation_main_aligned_20260923')
        finish(pdf, f, 3)

        f = page(4, '論文で主張すること・しないこと', 'A focused research story',
                 '主題：シンドロームによる位置推定の有効性、必要な観測条件、適用限界。',
                 'Focus: localization from syndrome records, its observation requirements, and its limits.')
        txt(f, .80, '研究の結論案', 'Proposed central claim', 14)
        txt(f, .758,
            '今回のシミュレーション分布では、同じ観測記録を用いたSVRが適用版REIより\n全強度平均の位置誤差を低減した。探索的な強度別分析では、中・強で改善した。\n弱では改善せず、未知の遅い伝播では差が不明確で、通常条件の全体p90は悪化した。',
            'On this simulation distribution, SVR lowers overall mean localization error\nversus adapted REI using the same records. Exploratory analyses show medium/strong gains.\nWeak events do not improve; the slow-shift advantage is unclear and nominal p90 worsens.', 11)
        txt(f, .65, '公平性と解釈の境界', 'Fairness and interpretation', 14)
        txt(f, .607,
            '1. テスト真値は推論に渡さない。強度別の結果は真値ラベルで集計し、強度判定器は未評価。\n2. 平常差分較正はSVR/Ridgeのみ。CNNは学習データで標準化。REIと事前情報は異なる。\n3. 約4/8 msではREIの検証選択も全履歴。較正なしSVRも通常4 msで2.257 mm。\n4. REIは適用版で、原著の完全再現ではない。計算コストや原著に対する優越性は未実証。\n5. 主比較以外は探索的。区間は固定学習集合でのテスト変動で、研究全体の反復探索は未補正。\n6. 単一デバイスの現象論的伝播＋局所QP-ODE＋Pauli近似。実機の物理妥当性は未保証。',
            '1. Inference uses no test truth. Strength groups use true labels, not a tested strength classifier.\n2. SVR/Ridge subtract quiet rates; CNN uses training-set scaling. Prior information differs from REI.\n3. REI validation selects full history at ~4/8 ms. Uncalibrated SVR: 2.257 mm at nominal ~4 ms.\n4. Adapted REI is not a full reproduction. Original-paper and computational superiority are unproven.\n5. Intervals condition on training data; secondary and research-wide repeated searches are unadjusted.\n6. One simulated device: phenomenological propagation + local QP-ODE + Pauli approximation.', 10)
        txt(f, .399, '主比較に混ぜない結果', 'Keep separate from the primary comparison', 13)
        txt(f, .361,
            '物理照合は潜在確率を学習に使う別方式。oracleは真値を使う診断で、実用精度ではない。\n固定面積density比較・符号距離・特徴量結果は補助分析であり、主比較と別に解釈する。\ndensity-pitch縮小パイロットは標本数と物理範囲が異なるため、実験ブランチに分離する。\n異なるテストの最良値を集めて優劣をつけない。弱・中・強は生成器の強度帯であり、実測エネルギーによる普遍的な分類ではない。',
            'Physical-template methods use latent probabilities during training; oracle results are diagnostics.\nThe fixed-area density, code-distance, and feature results are supporting analyses, separate from the primary contrast.\nThe compact density-pitch pilot remains on the experiment branch because its sample size and physical extent differ.\nDo not form a best-score leaderboard across different tests. Weak / medium / strong are simulator generation bands, not universal measured-energy classes.', 10)
        txt(f, .273, '投稿に向けて必要な検証', 'Before submission', 13)
        txt(f, .235,
            '独立した学習集合で追試／物理モデルの妥当性・先行研究との差を確認／待ち時間込みの速度評価。\n実機・BB code・連続監視・複数イベント・復号への効果は未実証。論文としての新規性は別途検証する。',
            'Repeat with independent training sets; validate physics and literature novelty; measure end-to-end latency.\nReal hardware, BB codes, continuous monitoring, multiple events and decoding benefits are not established.', 10)
        txt(f, .147,
            '出典（実験フォルダ内のRESULT_JA.mdと保存済み評価データ）\n[1] rei_fair_20260911  [2] distance_main_svr_20260923  [3] feature_ablation_main_aligned_20260923\n同梱のsources.jsonに入力ファイルのSHA-256を保存。図はCSVから再生成し、棒の軸はゼロ始まり。',
            'Sources: saved evaluation data and RESULT_JA.md in each experiment directory.\n[1] rei_fair_20260911  [2] distance_main_svr_20260923  [3] feature_ablation_main_aligned_20260923\nInput SHA-256 hashes are in sources.json. Charts are generated from CSV; bars start at zero.', 8)
        finish(pdf, f, 4)

if __name__ == '__main__':
    assert abs(metric('SVR')-2.214011)<1e-5
    assert pair()['primary'] is True
    manifest = {}
    for name in SOURCES + [f'{r}/RESULT_JA.md' for r in ['rei_fair_20260911','distance_main_svr_20260923','feature_ablation_main_aligned_20260923']]:
        p = RUNS / name
        manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for lang in ['ja','en']: build(lang)
    (OUT / 'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(OUT)
