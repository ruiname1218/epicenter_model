# Radiation epicenter localization from syndrome records

A curated **simulation-study results release**. The main comparison is SVR versus an
adapted REI localizer using the same observation records; CNN and Ridge are secondary
comparators. This is a research brief, not a peer-reviewed paper or a complete
reproduction of the original REI publication.

## Read the results

- [English research brief — 4 pages](paper/research_summary_en.pdf)
- [日本語の研究概要 — 4ページ](paper/research_summary_ja.pdf)
- [日本語ガイド](docs/README_JA.md)
- [Content review and corrections](paper/REVIEW_JA.md)

### Main result

At approximately 4 ms on the balanced nominal test set (540 physical events), mean
Euclidean localization error was **2.214 mm for SVR versus 2.571 mm for adapted REI**
(13.9% lower). The paired-event 95% interval for SVR minus REI was
**−0.414 to −0.301 mm**.

This is conditional on the selected simulation distribution, training set and device.
Weak events did not improve; overall p90 worsened. The advantage under unseen slow
propagation was unclear. No claim is made about real hardware, event detection,
decoding performance, or superiority over the original REI implementation.

## Layout

```text
paper/       Bilingual briefs, source hashes and review record
results/     Three selected studies: metrics, predictions, protocols and audit records
scripts/     Rebuild the briefs and verify published metrics
docs/        Scope, provenance and reproducibility instructions
```

| Study | Role | Full report |
|---|---|---|
| `rei_fair_20260911` | Primary matched-window comparison and exploratory stress tests | [Report](results/rei_fair_20260911/RESULT_JA.md) |
| `long_observation_20260910` | Supporting paired observation-window experiment | [Report](results/long_observation_20260910/RESULT_JA.md) |
| `feature_selection_20260910` | Supporting feature / kernel-parameter analysis | [Report](results/feature_selection_20260910/RESULT_JA.md) |

Scores from different studies must not be combined into a common leaderboard.
Historical reports retain their original context; the briefs use only the selected
comparisons. `main` intentionally excludes model checkpoints, raw syndrome arrays,
feature caches and unrelated exploratory results.

## Reproduce the published figures

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-figures.txt
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/build_paper_summary.py
```

The figure script uses DejaVu Sans and
`/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf` for Japanese. On Debian/Ubuntu,
install `fonts-droid-fallback` if needed. This workflow rebuilds figures from saved
evaluation data; **it does not regenerate syndromes or retrain models**.
See [reproducibility and provenance](docs/REPRODUCIBILITY.md).

## Branch policy

- `main`: curated, lightweight paper-facing evidence and figure reproduction.
- `experiments/snapshot-20260912`: experimental working snapshot, including the
  simulation/training implementation and historical results; large artifacts require
  a separately documented storage policy. Do not merge this entire branch into `main`.

The experiment branch is not a separate repository: both branches have the same
public visibility. No software/data license has yet been selected. Public availability
does not itself grant a reuse license; see [provenance](docs/REPRODUCIBILITY.md).
