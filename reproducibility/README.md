# Rebuilding experiments from generated records

This directory adds a source-to-results path, separate from the saved-result
verifiers. It does not modify any published result. A smoke run checks the pipeline;
its small-data accuracy must **not** be compared with the paper tables.

## Environment (Linux x86-64, CPU)

Use Python **3.10.12**, Git and network access for the initial source/package fetch.
The direct dependency list records the original numerical-library versions. The
lockfile also fixes the transitive dependencies resolved in the clean test environment.

```bash
python3.10 -m venv .venv-repro
env -u PYTHONPATH .venv-repro/bin/python -m pip install -r reproducibility/requirements-linux-py310.lock
env -u PYTHONPATH .venv-repro/bin/python -m pip check
```

`env -u PYTHONPATH` avoids importing unrelated system packages (for example ROS).
CPU PyTorch is fetched from its official wheel index. Other platforms, Python
versions and GPUs are not covered by this environment specification.

## Small end-to-end tests

From the repository root:

```bash
env -u PYTHONPATH .venv-repro/bin/python reproducibility/run.py radiation --profile smoke
env -u PYTHONPATH .venv-repro/bin/python reproducibility/verify_run.py radiation-smoke
env -u PYTHONPATH .venv-repro/bin/python reproducibility/run.py leakage --profile smoke
env -u PYTHONPATH .venv-repro/bin/python reproducibility/verify_run.py leakage-smoke
```

The first run fetches the simulator and original learning/evaluation modules at
immutable commits from `sources.json`. It writes only to ignored `.repro_sources/`
and `.repro_runs/<name>/run/`. Existing run directories are refused; use
`--run-name another-name` for a repeat. No prior local experiment, raw data, trained
model or feature cache is required.

## Paper-condition runs

```bash
env -u PYTHONPATH .venv-repro/bin/python reproducibility/run.py radiation --profile paper
env -u PYTHONPATH .venv-repro/bin/python reproducibility/verify_run.py radiation-paper
env -u PYTHONPATH .venv-repro/bin/python reproducibility/run.py leakage --profile paper
env -u PYTHONPATH .venv-repro/bin/python reproducibility/verify_run.py leakage-paper
env -u PYTHONPATH .venv-repro/bin/python reproducibility/compare_paper.py radiation-paper
env -u PYTHONPATH .venv-repro/bin/python reproducibility/compare_paper.py leakage-paper
```

| Study | Smoke | Paper profile |
|---|---|---|
| Radiation | 36 train, 36 validation, 36 nominal + 36 slow test events; 4-ms evaluation; one CNN seed, 2 epochs | 864 train, 180 validation, 540 nominal + 216 slow test events; 2/4/8-ms evaluation; three CNN seeds, 80 epochs |
| Leakage | 36 train, 12 validation, 12 test events; one replica, 2 CNN epochs | 864 train and 180 validation per replica; 540 shared test events; three replicas, 60 CNN epochs |

All windows retain the actual simulator timing: the radiation smoke run still
generates the 8,192-round records, rather than substituting an easier simulator.
Radiation includes paired doubled-noise records and the frozen-model slow shift.
Leakage includes all six poster studies, including d=15 fixed-area density.
Paper runs are substantially larger; plan CPU time and several GB of free storage.
The paper profile is a runnable specification, not by itself evidence that a full
rerun has completed. Consult `VALIDATION.md` for what was actually tested.

### Exact scope

- Radiation: the matched REI/SVR/Ridge/CNN main benchmark and its 2/4/8-ms,
  noise and slow-shift analyses. The separate radiation feature-ablation, density,
  coverage and physical-validation studies remain **saved-result releases**;
  this launcher does not yet rerun every radiation poster panel.
- Leakage: the matched six-study suite. The supplementary old-centroid recheck
  remains a saved-result comparison, outside this training launcher.

## Settings, model selection and isolation

`configs/*-paper.json` fixes event counts, seeds/replicas and training budgets;
the archived `protocol.json` and (for radiation) `specs.json` fix geometry,
physics, event seeds and splits. SVR searches C={0.1,1,10} and gamma numerator
{0.057,0.57}; ridge searches alpha={1,10,100,1000}. Selection uses validation
mean coordinate error, not test labels. CNN checkpoint selection also uses
validation error. REI follows each study's original history-selection policy;
radiation's primary comparator is full-history REI, unlike leakage's selected history.

The code records effective settings, Python/package versions, source hashes,
validation selections and elapsed time in each run. Smoke counts override the
archived design description; `run_metadata.json` identifies the actual profile.
Bootstrap rejects modified cached source. Test events remain event-disjoint from
training/validation; two shots never cross a split.

The radiation adapter exposes the epoch count but preserves the paper's 80 epochs.
The leakage adapter replaces a legacy helper import with the same sampling/container
functions and exposes the epoch count. Its source/onset patch reproduces the
instrumentation used to share event truth across shots. Neither adapter changes
the intended paper-condition physics or feature definitions.

## Numerical expectations and rights

Fixed seeds do not guarantee bit-identical results across numerical libraries,
platforms or thread implementations. Internal recomputed errors are checked with
floating-point tolerance. Do not silently accept changed scientific conclusions:
compare paper-run metrics and event splits with the published tables and report any
discrepancy. Smoke scores are functional checks only.

Upstream projects currently have no selected license in the inspected snapshots.
This repository does not assign them a license or vendor their simulator source.
Bootstrap fetches the original repositories at pinned commits and applies only the
local instrumentation patch. Fetchability does not establish redistribution or
reuse permission; those terms still need resolution before a formal software release.
