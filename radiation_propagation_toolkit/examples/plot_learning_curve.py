"""Plot predeclared learning curves, with event-clustered uncertainty."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    results = json.loads((args.experiment / "metrics.json").read_text())
    frame = pd.DataFrame(results["learning_curve"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, split, title in zip(axes, ["test_id", "test_ood"],
                                ["Unseen events, trained condition types", "Held-out strong elliptical events"]):
        subset = frame[frame.split == split]
        for model, color in [("cnn", "#0072B2"), ("mlp", "#D55E00")]:
            rows = subset[subset.model == model].sort_values("train_events")
            x = rows.train_events.to_numpy()
            bounds = rows.mean_95pct_event_bootstrap_mm.tolist()
            ax.plot(x, rows.mean_error_mm, "o-", color=color, label=model.upper())
            ax.fill_between(x, [b[0] for b in bounds], [b[1] for b in bounds], color=color, alpha=.12)
        baseline = subset[subset.model == "mlp"].sort_values("train_events")
        ax.plot(baseline.train_events, baseline.centroid_mean_error_mm, "--", color="#009E73", label="Syndrome centroid")
        ax.plot(baseline.train_events, baseline.constant_mean_error_mm, ":", color="#666666", label="Training mean position")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Independent training events")
        ax.set_xticks(baseline.train_events)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean single-shot epicenter error (mm)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed device / surface code d=3; seed-averaged error, 95% event-bootstrap bands", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
