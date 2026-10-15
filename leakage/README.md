# Leakage source localization

A curated **detector-level simulation study** using a local ReserveQ-derived
leakage injector and rotated surface-code detection records. The input is one
shot of check-by-time observations; the output is a source coordinate.

## Read the results

- [Complete English research report (3 pages, monochrome)](results/leakage_research_report_complete_20261009/leakage_research_report_en.pdf)
- [English poster](results/leakage_poster_summary/leakage_results_poster_en.pdf)
- [Japanese detailed experimental report](results/poster_matched_v3/REPORT_JA.md)
- [Poster-to-report coverage inventory (44 result entries)](results/leakage_research_report_complete_20261009/poster_result_coverage.csv)

The primary nominal comparison uses 540 held-out events, two separately predicted
shots per event and three training replicas. Each replica uses 864 training and
180 validation events. The main window is a 16-round prefix with 15 internal layers.

| Method | Mean error (nominal mm) |
|---|---:|
| Adapted REI | 2.695 |
| SVR | 1.843 |
| Ridge | 1.892 |
| CNN | 1.692 |

The separately evaluated earlier weighted centroid achieved **2.082**. Thus SVR's
reduction is 31.6% relative to adapted REI but 11.5% relative to that centroid.
Do not interpret the former as an improvement over all analytic baselines.
Adapted REI's 66.7% answer rate and specified fallback are included in the report;
the same answered-shot subset is also compared across methods.

## Included evidence

- `results/poster_matched_v3/`: metrics, per-shot predictions, validation choices,
  protocol, audits, figures and the original environment record.
- `results/legacy_centroid_reference_v3/`: **supplementary**, untuned old-centroid
  comparison on the same test records; not part of the primary poster figure.
- `results/leakage_research_report_complete_20261009/`: complete report and its figure.
- `results/leakage_poster_summary/`: English poster and explanatory notes.

Tests were split by event; true position, onset and strength are not model inputs.
Training and test source locations nevertheless come from the same discrete set.
The spatial scale is assumed, not calibrated hardware mm. Weak/medium/strong
denote corruption probabilities, not measured leakage strengths. Density changes
also change physical propagation length at fixed graph transport. Read these as
conditional simulation results, not general physical density effects.

No measured time/space-response comparison, qutrit-circuit validation, BB-code
localization, unseen-location generalization or decoding evaluation is included.

## Verify and rebuild

From this directory, in a separate virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-reports.txt
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/build_research_report.py
.venv/bin/python scripts/build_leakage_poster.py
```

Builders place derived files in `generated/`, leaving the published evidence
unchanged. Verification checks the publication file hashes, coordinate errors,
112 aggregate rows, event counts and the 44-entry poster coverage inventory.
It does not refit models or rerun the raw-data audit.

Raw arrays and checkpoints remain local and are intentionally omitted. The local
simulator derives from `droman9/ReserveQ-leakage` commit `441718d`; upstream simulator
source is not vendored here. Adapted REI uses the local radiation toolkit's
`rei_rates` implementation, not a claim of full original-paper reproduction.
The simulation/training environment is recorded separately from report dependencies.
No new license is assigned to upstream code or these results.
