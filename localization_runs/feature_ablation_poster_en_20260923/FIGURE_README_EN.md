# English poster feature-ablation figure

This figure keeps only the five requested conditions: F0 (spatial + temporal), F1 (spatial only), F2 (no spatial / temporal only), N1 (shuffled check positions), and N2 (shuffled time order). It uses the same primary corpus, train/test split, 120-dimensional input size, and fixed SVR hyperparameters as the main SVR--REI comparison. The REI reference line is intentionally omitted because this figure is about feature importance, not model ranking.

The key result is that removing spatial check information (F2) and destroying check-position correspondence (N1) substantially increase the localization error, while shuffling temporal order (N2) has little effect in this representation.
