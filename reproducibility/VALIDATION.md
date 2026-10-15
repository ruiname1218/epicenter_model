# Reproduction checks — 2026-10-15

These checks used a newly created Python 3.10.12 CPU virtual environment, with
the dependencies recorded in `requirements-linux-py310.lock`. No original raw
arrays, trained models or feature caches were supplied to the runners.

| Run actually completed | Prediction rows | Metric rows | Runtime* | Result |
|---|---:|---:|---:|---|
| Radiation smoke | 1,944 | 378 | 111 s | Split, count, coordinate-error and aggregate checks passed |
| Leakage smoke | 672 | 112 | 8 s | Same checks passed |
| Leakage paper profile, all six studies | 90,720 | 112 | 188 s | Checks passed; all compared numeric metrics and per-shot predictions matched the archived results exactly |

*One local CPU environment; excludes installation and fetching. These are not
runtime benchmarks or promises for other machines. Smoke scores are not paper results.

The leakage comparison checked all 112 metric rows and 90,720 prediction rows;
the maximum absolute difference was zero for both metrics and predictions.
This verifies computational reproduction of this archived simulation experiment,
not physical accuracy of the leakage model or validation on hardware.

A separate temporary source-only export also fetched both upstream revisions
afresh and completed both smoke pipelines (radiation: 114 s). This was a clean
source export, not a clone of a newly published commit. It used the same newly
installed virtual environment and did not access the original experiment folder.
A final fresh-fetch leakage smoke test also passed after the manifest and
runtime-copy changes; the fetched leakage Git checkout remained clean.

## Limits

- The radiation **full paper profile has not been rerun in this check**. Its
  configuration and archived simulator dependency hashes are checked, and its
  end-to-end smoke pipeline passes; this is not full numerical reproduction.
- Radiation's separate feature-ablation, density, coverage and physical-response
  panels still have saved-result reproduction only, not this training launcher.
- Cross-platform, GPU and independent-machine reproduction have not been tested.
- Upstream licensing remains unresolved. Pinned fetching is not a license grant.

## Maintenance record

The runner exposes existing CNN epoch budgets, uses the original event/split
design, and instruments a runtime copy of the leakage source for shared event
coordinates/onsets. No new physical assumptions or tuning criteria were added.
During validation, the completion metadata was found to change after the leakage
manifest was written. The runner now finalizes that hash after completion and
the verifier checks the manifest. This changes bookkeeping, not scientific results.

Lightweight machine-readable checks are in `validation/`. Full newly generated
data and models remain local under the ignored `.repro_runs/` directory.
