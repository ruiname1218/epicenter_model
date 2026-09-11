"""Plot held-out localization errors; this is evaluation, not model inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    frame = pd.read_csv(args.model_dir / "test_predictions.csv")
    report = json.loads((args.model_dir / "metrics.json").read_text())
    labels = pd.read_csv(args.dataset / "true_parameters.csv")
    splits = json.loads((args.model_dir / "splits.json").read_text())
    training_center = labels.iloc[splits["train"]][["epicenter_row", "epicenter_col"]].mean().to_numpy()
    with np.load(args.dataset / "stim_syndrome_events.npz") as archive:
        qubits = archive["circuit_physical_coords_mm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    first = frame[frame["shot"] == 0]
    axis = axes[0]
    axis.scatter(qubits[:, 0], qubits[:, 1], c="0.75", marker="s", s=25, label="Qubits")
    axis.scatter(first.true_x_mm, first.true_y_mm, c="#007f73", s=23, label="True source")
    axis.scatter(first.predicted_x_mm, first.predicted_y_mm, c="#b74329", marker="x", s=26, label="Prediction")
    for row in first.itertuples():
        axis.plot([row.true_x_mm, row.predicted_x_mm], [row.true_y_mm, row.predicted_y_mm], c="0.65", lw=0.6, alpha=0.6)
    axis.set(xlabel="x (mm)", ylabel="y (mm)", title="Unseen events: first shot only", aspect="equal")
    axis.legend(fontsize=8, loc="upper left")
    truth = frame[["true_x_mm", "true_y_mm"]].to_numpy()
    centroid = frame[["centroid_x_mm", "centroid_y_mm"]].to_numpy()
    for name, errors, color in (
        ("ExtraTrees", frame.error_mm.to_numpy(), "#007f73"),
        ("Syndrome centroid", np.linalg.norm(truth - centroid, axis=1), "#b74329"),
        ("Training mean", np.linalg.norm(truth - training_center, axis=1), "0.4"),
    ):
        ordered = np.sort(errors)
        axes[1].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", color=color, label=name)
    axes[1].set(xlabel="Position error (mm)", ylabel="Fraction of predictions", ylim=(0, 1), title="Single-shot error distribution")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    count = report["independent_events"]["test"]
    shots = report["shots_per_event"]
    fig.suptitle(f"Synthetic syndrome localization | {count} held-out events x {shots} shots\nIndependent unit: event; fixed device and circuit", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
