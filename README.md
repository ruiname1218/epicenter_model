# Epicenter localization — experimental snapshot

This branch preserves the current research code, configurations, reports, figures and
lightweight evaluation artifacts from the experimental working directory.
It is an **exploratory archive**, not the curated paper-facing release.

**For the selected research results, start with [main](https://github.com/ruiname1218/epicenter_model/tree/main).**

## Start here

- [Full experiment history (Japanese)](localization_runs/ALL_RESULTS_JA.md)
- [Latest matched-window SVR / adapted-REI comparison](localization_runs/rei_fair_20260911/RESULT_JA.md)
- [Japanese research brief](localization_runs/paper_summary_20260912/research_summary_ja.pdf)
- [English research brief](localization_runs/paper_summary_20260912/research_summary_en.pdf)
- [Simulator and experiment implementation](radiation_propagation_toolkit/)
- [Initial simulator analysis](ANALYSIS_JA.md) — a historical record, not the current state

Different experiments have different training sets, test sets, signal distributions,
observation windows and sometimes privileged training information. Do not combine
their best scores into a single leaderboard. Older reports retain their original
context; the latest brief distinguishes primary evidence, exploratory results and limits.

## What is intentionally omitted

At the owner's request, this branch uses **ordinary Git, not Git LFS**.

- Raw/feature arrays: `.npz`, `.npy`.
- Model/checkpoint formats: `.joblib`, `.pkl`, `.pickle`, `.pt`, `.pth`, `.ckpt`,
  `.safetensors`, `.h5`, `.hdf5`, `.onnx` (including small files of these types).
- Any remaining individual file larger than **25 MiB**, including large text tables.
- Virtual environments, nested `.git` internals, build products and cache directories.
- Secret-like filenames and symlinks are excluded; retained files are scanned for
  common credential signatures before copying. This is not an exhaustive security audit.

The original local files were not deleted or modified. Some saved manifests refer to
omitted files. **Cloning this branch does not provide all raw training data or pretrained
models, and does not by itself allow every historical pipeline to be rerun.**
Do not load untrusted pickle/joblib checkpoints from other sources.

Publication inventories:

- [Snapshot counts and byte totals](publication/snapshot_summary.json)
- [Retained original files and SHA-256 hashes](publication/included_files.csv)
- [Omitted files, sizes and reasons](publication/excluded_files.csv)
- [Omitted environment/VCS/cache directories](publication/excluded_directories.csv)
- [Snapshot-copy script](publication/create_snapshot.py)

The inventory describes copied source files, not newly added publication documentation.
The original nested `.gitignore` is retained; listed included files are explicitly
staged so that existing ignore rules cannot silently drop small research artifacts.

## Branches and provenance

- `main`: selected results, bilingual briefs, auditing and figure-reproduction tools.
- `experiments/snapshot-20260912`: this lightweight snapshot of all experimental work.

The branches have separate root histories so unrelated exploratory content is not
introduced into `main`. Both are public branches of the same repository. Do not merge
the entire experiment snapshot into the paper-facing branch.

Simulator upstream: <https://github.com/ruiname1218/radiation_propagation_toolkit>,
base commit `5de1632983f3a0d5bfdc907802f98ce5fb29e889`, plus local research changes.
This snapshot does not replace or modify the original local simulator repository.
No new software or data license is granted here; the upstream README says that a
license has not yet been selected.

## 日本語

このブランチは、全探索のコード・設定・報告・図・軽量な評価ファイルを保存したものです。
生シンドローム、特徴配列、学習済みモデル、および単体25 MiB超のファイルはpushしていません。
除外したファイルは上記一覧に記録し、元のローカルデータはすべて保持しています。
論文向けに選別・整理した結果は `main` を参照してください。
生データやモデルを除いているため、このブランチ単独で全実験の再学習・推論ができるわけではありません。
