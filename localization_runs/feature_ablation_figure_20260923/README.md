# Feature-ablation figure — existing result, exploratory

Files: `feature_ablation_existing.png`, `.pdf`, `.svg`.

## What is plotted

Each panel shows one strength band. The x-axis is a feature representation; the y-axis
is its paired change in mean Euclidean localization error relative to
`relative__g120`. Positive values mean higher error (worse). The SVR learner is fixed
at C=1, epsilon=0.1 and the same numerical gamma 0.57/120 for every representation.
The training-event corpus and test records are shared across feature conditions.

Diamonds and whiskers show the paired event-bootstrap mean difference and its 95%
interval from 10,000 resamples. These are exploratory unadjusted intervals over many
feature contrasts. The three smaller points are paired estimates for three independent
test-generation roots (60 physical events per strength and root), not independent
training replicates. There is only one fixed training corpus in this study.

This study used d5, 4.096 ms and 1,080 training / 180 validation / 540 fresh test
physical events, two syndrome shots per event. The validation set was reused from an
earlier study. The 540-event test is common to all features. The report and full
protocol are in `../feature_selection_20260910/RESULT_JA.md` and `protocol.json`.

Feature labels: Full = the original 265-dimensional representation; Relative+level =
120 relative-pattern values plus five response-level summaries; Early/Late = the first/
latter temporal portions; Log/Signed/Sqrt = alternative transforms of the 120 relative
values; Top60 = 60 train-selected values; PCA32 = 32 train-fitted principal components.
Definitions follow the source report; they are not the plan's F0–N2 ablations.

In the plotted historical comparison, weak-event changes are small relative to this
baseline. For strong events, the 265-feature full representation is worse than the
120-feature relative-pattern baseline by about 0.455 mm. The log transform has a
small medium-event improvement (about 0.025 mm); its strong-event difference is
uncertain. These are exploratory, unadjusted contrasts, not universal feature rankings.

## Relation to the requested re-experiment plan

This is a useful existing-data figure, **not the fully planned new-corpus experiment**.
It has no five independent train/validation replicates; it does not contain the planned
120-cell F0 matrix, fixed spatial-only / temporal-only / global-level conditions, or
check/time shuffling controls. The first study is 4 ms and closely matches the planned
window, but its feature recipes are not interchangeable with F0–N2. Do not label the
three test roots as training replicates, infer hardware performance, or claim a causal
information-theory result from these ablations.

The historical comparison was produced with fixed learner settings across feature
recipes, which is a controlled representation comparison. Feature dimensions differ
intentionally; equal numerical gamma does not make effective model capacity identical.
For the full five-training-replicate poster specified in the plan, the new experiment
still needs to be run.

## Rebuild

From `radiation_propagation_toolkit/` run:

```bash
.venv/bin/python examples/plot_feature_ablation_existing.py
```

The script reads saved CSV/JSON summaries only; it does not train models or modify
the historical experiment.
