# Source localization from QEC detection records

Curated results from two **synthetic localization studies**, organized separately
because their event models, source distributions and observation windows differ.
This is a research-results release, not a hardware validation or a complete
reproduction of the original REI publication.

| Study | Start here | Report | Poster |
|---|---|---|---|
| Radiation | [Overview and reproduction](radiation/README.md) | [English](radiation/paper/research_summary_en.pdf) / [日本語](radiation/paper/research_summary_ja.pdf) | [PDF](radiation/paper/Poster_Rui_Suzuki_v2.pdf) |
| Leakage | [Overview and reproduction](leakage/README.md) | [Complete English report, monochrome](leakage/results/leakage_research_report_complete_20261009/leakage_research_report_en.pdf) | [PDF](leakage/results/leakage_poster_summary/leakage_results_poster_en.pdf) |

## Repository layout

```text
radiation/   Existing radiation results, bilingual briefs, poster and scripts
leakage/     Leakage results, complete English report, poster and scripts
```

The leakage report includes every result series shown on the leakage poster:
model comparisons by strength proxy, feature ablations, observation length,
noise/transport shifts, fixed-area density and fixed-spacing coverage by strength.
Its earlier weighted-centroid check is stored separately as a supplementary
comparison, not mislabeled as REI.

Do not combine radiation and leakage errors into one leaderboard. Leakage uses a
chosen **nominal mm** scale, discrete data-qubit source locations and short
round-based windows. Its strength proxies are not radiation energies.

## Verify saved results

Each study has its own dependencies and reproduction instructions:

```bash
python radiation/scripts/verify_results.py
python leakage/scripts/verify_results.py
```

These checks and report builders use saved evaluation files. They do **not**
regenerate syndromes or retrain the models. Raw arrays, feature caches, trained
models, virtual environments and files over 25 MiB are intentionally excluded.
Original experiment data remain in the local research workspace.

## Branches and scope

- `main`: curated, lightweight research evidence and figure/report reproduction.
- `experiments/snapshot-20260912`: a separate historical experiment snapshot.
  This publication update does not synchronize that branch or make it a complete
  current training kit.

The previous top-level `paper/`, `results/`, `scripts/` and `docs/` directories now
live under `radiation/`. Their contents remain available in Git history; run their
documented reproduction commands from the `radiation/` directory.

No claim is made about hardware generalization, event detection, decoding benefit
or superiority over original REI. No repository-wide reuse license has been
selected; public access is not itself a license. See the study-specific provenance.
