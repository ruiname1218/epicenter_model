# Main-aligned feature ablation

This is a re-analysis on `rei_fair_20260911`, using the same train/validation split, nominal ID test events, d5 device, 4.096 ms record, labels and fixed SVR recipe as the primary SVR-vs-adapted-REI comparison. It transforms the frozen `fit_x4.npy` and `test_x4.npy` features; no new syndrome generation occurs.

F0 is the primary SVR input (relative spatial distribution over five windows). F1 removes temporal variation, F2 removes check-to-check variation, F3 retains only the global normalized response level, F4 removes each-check temporal mean, F5/F6 retain early/later windows, and N1/N2 destroy check/time order by deterministic per-shot permutations. All variants have 120 numeric inputs and use C=1, epsilon=.1, gamma=.57/120.

The plot should be described as a controlled re-analysis under the primary benchmark protocol. It is not the fully preplanned new-corpus study with five independent training/validation replicates, absolute baseline-adjusted F0, or the full multiplicity correction plan.
