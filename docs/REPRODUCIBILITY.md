# Reproducibility, scope and provenance

## Two reproducibility levels

1. **This main branch:** rebuild the bilingual report from saved CSV files, verify
   the nominal 4 ms metrics against saved predictions, inspect configuration/selection
   records and published confidence intervals.
2. **Experiment snapshot:** simulation, training, inference code and historical
   artifacts in their original directory layout. Raw arrays/checkpoints have a
   separate large-file storage policy. Original pipelines also depend on earlier
   runs and frozen file hashes; this main branch is not a standalone retraining kit.

The PDFs and metrics are retained as experiment evidence, not a claim that every
original pipeline is independently reproducible with these lightweight files alone.
Saved timing measurements are not comparable end-to-end inference latency.

## Data interpretation

- Input: detector-event records from rotated surface-code memory-Z simulations.
- Target: continuous epicenter coordinates; error is Euclidean distance in mm.
- Event presence is assumed within the observation window.
- Two shots from one physical event remain in the same train/validation/test split.
- The main nominal test contains 180 events per weak/medium/strong band. This is a
  balanced simulation distribution, not measured natural event frequencies.
- Noise-shifted observations reuse the nominal radiation events; they do not add
  540 independent physical events. There are 756 unique test events including slow OOD.
- The primary mean-error interval uses paired event bootstrap with fixed trained models.
  Secondary analyses and repeated research-wide exploration are not globally corrected.
- Strong ellipses are included in the main benchmark's training but excluded from
  training in the two older supporting studies. Do not rank scores across studies.

## Original code provenance

The simulator working tree was based on
<https://github.com/ruiname1218/radiation_propagation_toolkit>, upstream commit
`5de1632983f3a0d5bfdc907802f98ce5fb29e889`, with local research additions.
The existing simulator repository and its local Git history are not modified by
publication. The original README states that no software license has been selected;
this release does not introduce or assume a new license.

`paper/sources.json` gives hashes for the input CSV/JSON files and original result
reports used by the figure script. Protocols and manifests may refer to raw files,
original paths or older runs absent from main. Those references document provenance;
they are not promises that those artifacts are present here.

Rebuilding a PDF may change metadata timestamps. Compare input hashes, plotted
values and report content rather than demanding byte-identical PDF files.

## Before a formal paper submission

Confirm the original REI input/algorithm mapping, complete the literature review and
novelty argument, specify all generator/model/statistical settings, and repeat with
independent training sets. Test physical-model mismatch and measure acquisition plus
inference delay before making hardware or operational claims.
