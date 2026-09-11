"""Bounded-feature-memory training from committed syndrome shards."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import dataset_manifest, iter_shards
from .localization import (
    SCHEMA_VERSION, _hash_file, _learning_dependencies, _new_directory,
    extract_features, feature_specification, location_metrics, split_events,
    syndrome_centroid,
)


def train_sharded_localizer(
    dataset: str | Path, output: str | Path, *, bins: int = 16,
    seed: int = 42, epochs: int = 20, batch_size: int = 256,
) -> dict:
    """Train a 128/64-unit MLP incrementally; select epoch using validation only.

    Raw syndrome arrays are read one shard at a time. Feature shards are cached on
    disk for repeated epochs. Labels and per-sample evaluation results stay in RAM.
    """
    joblib, sklearn, _ = _learning_dependencies()
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    source, destination = Path(dataset), Path(output)
    manifest = dataset_manifest(source)
    _new_directory(destination)
    with tempfile.TemporaryDirectory(prefix="qp_localization_features_") as temporary:
        cache = Path(temporary)
        cached, label_frames = [], []
        spec = None
        for index, (data, labels) in enumerate(iter_shards(source)):
            if spec is None:
                spec = feature_specification(data, bins)
            features, rates = extract_features(data, spec)
            events, shots, _ = data["detector_events"].shape
            controls = labels["is_control"].to_numpy(dtype=bool)
            if not np.all(labels.loc[~controls, "source_count"].to_numpy() == 1):
                raise ValueError("training supports only single-source events")
            targets = labels[["epicenter_row", "epicenter_col"]].to_numpy(dtype=float)
            if not np.isfinite(targets[~controls]).all():
                raise ValueError("non-control coordinates must be finite")
            feature_path = cache / f"features_{index}.npy"
            np.save(feature_path, features)
            cached.append({
                "path": feature_path,
                "event_ids": np.repeat(labels.event.to_numpy(), shots),
                "shots": np.tile(np.arange(shots), events),
                "targets": np.repeat(targets, shots, axis=0),
                "centroid": syndrome_centroid(rates, spec),
            })
            label_frames.append(labels)
        labels = pd.concat(label_frames, ignore_index=True).set_index("event", drop=False)
        usable = labels.loc[~labels.is_control.astype(bool), "event"].to_numpy(dtype=int)
        splits = split_events(usable, seed)
        train_targets = labels.loc[splits["train"], ["epicenter_row", "epicenter_col"]].to_numpy()
        center = train_targets.mean(axis=0)
        extent = np.ptp(spec["geometry"]["circuit_physical_coords_mm"], axis=0)
        scale = float(max(extent.max() / 2.0, 1e-3))
        scaler = StandardScaler()
        for entry in cached:
            mask = np.isin(entry["event_ids"], splits["train"])
            if mask.any():
                features = np.load(entry["path"], mmap_mode="r")
                scaler.partial_fit(features[mask])

        def evaluate(estimator, split):
            rows, truths, predictions, centroids = [], [], [], []
            for entry in cached:
                mask = np.isin(entry["event_ids"], splits[split])
                if not mask.any():
                    continue
                features = np.load(entry["path"], mmap_mode="r")
                prediction = estimator.predict(scaler.transform(features[mask])) * scale + center
                truth = entry["targets"][mask]
                rows.append(pd.DataFrame({
                    "event": entry["event_ids"][mask], "shot": entry["shots"][mask],
                    "true_x_mm": truth[:, 0], "true_y_mm": truth[:, 1],
                    "predicted_x_mm": prediction[:, 0], "predicted_y_mm": prediction[:, 1],
                    "error_mm": np.linalg.norm(truth - prediction, axis=1),
                    "centroid_x_mm": entry["centroid"][mask, 0],
                    "centroid_y_mm": entry["centroid"][mask, 1],
                }))
                truths.append(truth)
                predictions.append(prediction)
                centroids.append(entry["centroid"][mask])
            truth, prediction = np.concatenate(truths), np.concatenate(predictions)
            metrics = {
                "model": location_metrics(truth, prediction),
                "training_mean_baseline": location_metrics(truth, np.broadcast_to(center, truth.shape)),
                "syndrome_centroid_baseline": location_metrics(truth, np.concatenate(centroids)),
            }
            frame = pd.concat(rows, ignore_index=True)
            metrics["by_condition"] = {}
            for column in ("geometry", "propagation_law", "epicenter_region", "strength_band"):
                if column not in labels:
                    continue
                frame[column] = labels.loc[frame.event, column].to_numpy()
                groups = {}
                for value, group in frame.groupby(column):
                    indices = group.index.to_numpy()
                    group_metrics = location_metrics(truth[indices], prediction[indices])
                    group_metrics["independent_events"] = int(group.event.nunique())
                    groups[str(value)] = group_metrics
                metrics["by_condition"][column] = groups
            frame["event_uid"] = labels.loc[frame.event, "event_uid"].to_numpy()
            return metrics, frame

        estimator = MLPRegressor(
            hidden_layer_sizes=(128, 64), solver="adam", alpha=1e-3,
            learning_rate_init=1e-3, batch_size=batch_size, random_state=seed,
        )
        rng = np.random.default_rng(seed)
        best, best_error, best_epoch = None, np.inf, 0
        history = []
        for epoch in range(epochs):
            for shard_index in rng.permutation(len(cached)):
                entry = cached[shard_index]
                indices = np.flatnonzero(np.isin(entry["event_ids"], splits["train"]))
                rng.shuffle(indices)
                features = np.load(entry["path"], mmap_mode="r")
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    estimator.set_params(batch_size=len(batch))
                    estimator.partial_fit(
                        scaler.transform(features[batch]), (entry["targets"][batch] - center) / scale,
                    )
            validation, _ = evaluate(estimator, "validation")
            error = validation["model"]["mean_error_mm"]
            history.append({"epoch": epoch + 1, "validation_mean_error_mm": error})
            if error < best_error:
                best, best_error, best_epoch = copy.deepcopy(estimator), error, epoch + 1
        if best is None:
            raise ValueError("training did not produce finite validation errors")
        report = {
            "schema_version": SCHEMA_VERSION, "model": "streaming MLPRegressor(128,64)",
            "scope": "single-source event windows; same device/code/distribution; no event detector",
            "seed": seed, "features": int(scaler.n_features_in_), "time_bins": bins,
            "epochs": epochs, "selected_epoch": best_epoch, "history": history,
            "shots_per_event": manifest["stim_config"].get("shots_per_event", 1),
            "excluded_controls": int(labels.is_control.astype(bool).sum()),
            "independent_events": {name: len(ids) for name, ids in splits.items()},
            "split_unit": "source event ID; all shots remain together",
            "code": manifest["code"], "circuit_id": manifest["circuit_id"],
            "dataset_manifest_sha256": _hash_file(source / "dataset_manifest.json"),
            "localization_source_sha256": _hash_file(Path(__file__)),
            "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
            "evaluation": {},
        }
        for split in ("validation", "test"):
            report["evaluation"][split], frame = evaluate(best, split)
            frame.to_csv(destination / f"{split}_predictions.csv", index=False)
        joblib.dump({
            "schema_version": SCHEMA_VERSION, "features": spec,
            "estimator": Pipeline([("scaler", scaler), ("regressor", best)]),
            "target_scale_mm": scale, "target_center_mm": center,
            "sklearn_version": sklearn.__version__, "scope": report["scope"],
        }, destination / "model.joblib", compress=3)
        (destination / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (destination / "splits.json").write_text(json.dumps({name: ids.tolist() for name, ids in splits.items()}, indent=2) + "\n")
    return report
