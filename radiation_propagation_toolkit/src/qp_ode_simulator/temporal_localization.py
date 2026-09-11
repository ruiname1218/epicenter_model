"""Fixed-layout temporal CNN and controlled MLP learning curves.

The CNN consumes individual binary detector trajectories, not simulator fields.
All model/epoch choices use validation; test predictions are made only after all
predeclared runs have finished. Raw input/features are disk-backed caches.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import _generation_lock, _write_json, dataset_manifest, iter_shards
from .learning_plan import validate_plan
from .localization import (
    GEOMETRY_KEYS, _hash_file, _new_directory, extract_features, feature_specification,
    load_syndromes, location_metrics, syndrome_centroid,
)


class TemporalCNN(nn.Module):
    """Shared time filters per check, then an ordered fixed-layout spatial head."""

    def __init__(self, sites: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 8, 9, stride=4, padding=4), nn.ReLU(),
            nn.Conv1d(8, 16, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=4, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Linear(sites * 129, 64), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(64, 2))

    def forward(self, values):
        batch, sites, times = values.shape
        # Fixed numerical scaling, not event-specific standardization: absolute
        # detector rates remain available. No onset alignment or truth inputs.
        encoded = self.temporal(values.reshape(batch * sites, 1, times) * 10)
        encoded = encoded.reshape(batch, sites, -1)
        encoded = torch.cat((encoded, values.mean(dim=2, keepdim=True) * 10), dim=2)
        return self.head(encoded.flatten(1))


def detector_series(data, spec):
    """Return (event*shot, ordered site, interior round); reject missing checks."""
    extract_features(data, spec)  # Reuse strict circuit/layout/schedule/bit checks.
    coords = np.asarray(data["detector_coords"])
    times = len(data["round_time_ms"]) - 1
    indices = np.flatnonzero((coords[:, 2] >= 1) & (coords[:, 2] <= times))
    lookup = {tuple(site): i for i, site in enumerate(spec["sites_grid"])}
    site = np.array([lookup[tuple(coords[i, :2])] for i in indices])
    tick = coords[indices, 2].astype(int) - 1
    if len(np.unique(site * times + tick)) != len(lookup) * times or len(indices) != len(lookup) * times:
        raise ValueError("CNN requires exactly one interior detector per site and round")
    bits = data["detector_events"].reshape(-1, len(coords))
    result = np.empty((len(bits), len(lookup), times), dtype=np.uint8)
    result[:, site, tick] = bits[:, indices]
    return result


def _json_spec(spec):
    return {k: {a: b.tolist() for a, b in v.items()} if k == "geometry"
            else v.tolist() if isinstance(v, np.ndarray) else v for k, v in spec.items()}


def _prepare(dataset, destination, bins):
    manifest = dataset_manifest(dataset)
    samples = manifest["completed_events"] * manifest["stim_config"]["shots_per_event"]
    destination.mkdir(parents=True, exist_ok=False)
    cursor, frames, spec = 0, [], None
    for data, labels in iter_shards(dataset):
        if spec is None:
            spec = feature_specification(data, bins)
            features, rates = extract_features(data, spec)
            series = detector_series(data, spec)
            raw = np.lib.format.open_memmap(destination / "series.npy", mode="w+", dtype=np.uint8,
                                           shape=(samples, *series.shape[1:]))
            flat = np.lib.format.open_memmap(destination / "features.npy", mode="w+", dtype=np.float32,
                                            shape=(samples, features.shape[1]))
        else:
            features, rates = extract_features(data, spec)
            series = detector_series(data, spec)
        end = cursor + len(features)
        raw[cursor:end], flat[cursor:end] = series, features
        centroid = syndrome_centroid(rates, spec)
        shots = data["detector_events"].shape[1]
        rows = labels.iloc[np.repeat(np.arange(len(labels)), shots)].reset_index(drop=True).copy()
        rows["shot"] = np.tile(np.arange(shots), len(labels))
        rows["centroid_x_mm"], rows["centroid_y_mm"] = centroid[:, 0], centroid[:, 1]
        frames.append(rows)
        cursor = end
    if cursor != samples:
        raise ValueError("cache sample count disagrees with manifest")
    raw.flush()
    flat.flush()
    return raw, flat, pd.concat(frames, ignore_index=True), spec, manifest


def _batch_indices(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def _cnn_predict(model, raw, indices, center, scale, batch_size):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(torch.from_numpy(np.array(raw[idx], dtype=np.float32))).numpy() * scale + center
            for idx in _batch_indices(indices, batch_size)
        ])


def predict_cnn(model_directory, input_path, *, batch_size=128):
    """Predict without labels; validate geometry/order/timing against the artifact."""
    root = Path(model_directory)
    metadata = json.loads((root / "model.json").read_text())
    if metadata.get("model") != "temporal_cnn_v1" or batch_size < 1:
        raise ValueError("invalid CNN artifact or batch size")
    if _hash_file(root / "weights.pt") != metadata["weights_sha256"]:
        raise ValueError("CNN weights checksum mismatch")
    spec = metadata["specification"]
    raw = detector_series(load_syndromes(input_path), spec)
    model = TemporalCNN(len(spec["sites_grid"]))
    model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
    return _cnn_predict(model, raw, np.arange(len(raw)), np.asarray(metadata["target_center_mm"]),
                        metadata["target_scale_mm"], batch_size)


def _fit(kind, raw, flat, truth, train, val, spec, output, *, seed, epochs, patience, batch_size):
    import joblib
    import sklearn
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    output.mkdir()
    center = truth[train].mean(axis=0)
    scale = max(float(np.ptp(spec["geometry"]["circuit_physical_coords_mm"], axis=0).max() / 2), 1e-3)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if kind == "cnn":
        model = TemporalCNN(raw.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    else:
        scaler = StandardScaler()
        for idx in _batch_indices(train, batch_size):
            scaler.partial_fit(flat[idx])
        model = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                             batch_size=batch_size, random_state=seed)
    best, best_error, selected, history = None, np.inf, 0, []
    begin = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if kind == "cnn":
            model.train()
        for idx in _batch_indices(rng.permutation(train), batch_size):
            target = (truth[idx] - center) / scale
            if kind == "cnn":
                optimizer.zero_grad(set_to_none=True)
                prediction = model(torch.from_numpy(np.array(raw[idx], dtype=np.float32)))
                loss = nn.functional.mse_loss(prediction, torch.tensor(target, dtype=torch.float32))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                model.set_params(batch_size=len(idx))
                model.partial_fit(scaler.transform(flat[idx]), target)
        if kind == "cnn":
            prediction = _cnn_predict(model, raw, val, center, scale, batch_size)
        else:
            prediction = np.concatenate([model.predict(scaler.transform(flat[idx])) * scale + center
                                         for idx in _batch_indices(val, batch_size)])
        error = float(np.linalg.norm(prediction - truth[val], axis=1).mean())
        history.append({"epoch": epoch, "validation_mean_error_mm": error})
        if error < best_error:
            best_error, selected = error, epoch
            best = copy.deepcopy(model.state_dict() if kind == "cnn" else model)
        if epoch - selected >= patience:
            break
    if best is None:
        raise ValueError("no finite validation error during training")
    info = {"model": "temporal_cnn_v1" if kind == "cnn" else "MLPRegressor(128,64)",
            "seed": seed, "selected_epoch": selected, "history": history,
            "validation_mean_error_mm": best_error, "training_seconds": time.perf_counter() - begin,
            "target_center_mm": center.tolist(), "target_scale_mm": scale,
            "specification": _json_spec(spec)}
    if kind == "cnn":
        model.load_state_dict(best)
        torch.save(best, output / "weights.pt")
        info["weights_sha256"] = _hash_file(output / "weights.pt")
        info["parameter_count"] = sum(p.numel() for p in model.parameters())
    else:
        joblib.dump({"schema_version": 1, "features": spec,
                     "estimator": Pipeline([("scaler", scaler), ("regressor", best)]),
                     "target_scale_mm": scale, "target_center_mm": center,
                     "sklearn_version": sklearn.__version__, "scope": "fixed device/code; conditional localization"},
                    output / "model.joblib", compress=3)
    _write_json(output / "model.json", info)
    return info


def _mean_interval(frame, column="error_mm"):
    values = frame.groupby("event")[column].mean().to_numpy()
    rng = np.random.default_rng(2718)
    # Bound bootstrap working memory even on large event collections.
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(1000)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(frame):
    result = {"independent_events": int(frame.event.nunique()), "predictions": len(frame),
              "mean_error_mm": float(frame.error_mm.mean()),
              "median_error_mm": float(frame.error_mm.median()),
              "p90_error_mm": float(frame.error_mm.quantile(.9)),
              "within_1mm_fraction": float((frame.error_mm <= 1).mean()),
              "within_2mm_fraction": float((frame.error_mm <= 2).mean()),
              "mean_95pct_event_bootstrap_mm": _mean_interval(frame)}
    return result


def run_experiment(dataset, plan_path, output, *, seeds=(41, 42, 43), epochs=40,
                   patience=8, batch_size=64, bins=16, threads=2):
    import importlib.metadata
    import joblib
    root, plan_path, destination = Path(dataset), Path(plan_path), Path(output)
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)
            or min(epochs, patience, batch_size, threads) < 1):
        raise ValueError("invalid experiment settings")
    plan = json.loads(plan_path.read_text())
    manifest = dataset_manifest(root)
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    validate_plan(plan, labels, manifest, root)
    _new_directory(destination)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    from . import learning_plan, localization, dataset as dataset_module
    settings = {"plan_sha256": _hash_file(plan_path), "dataset_manifest_sha256": _hash_file(root / "dataset_manifest.json"),
                "seeds": list(seeds), "epochs": epochs, "patience": patience, "batch_size": batch_size,
                "bins": bins, "threads": threads, "models": ["mlp", "cnn"],
                "source_sha256": _hash_file(Path(__file__)),
                "dependency_source_sha256": {Path(m.__file__).name: _hash_file(Path(m.__file__))
                                              for m in (learning_plan, localization, dataset_module)},
                "versions": {n: importlib.metadata.version(n) for n in ("numpy", "torch", "scikit-learn", "stim")},
                "selection_rule": "lowest mean validation distance over seeds; representative artifact uses first seed",
                "test_policy": "evaluate all predeclared runs only after completing training; no test-based tuning"}
    _write_json(destination / "experiment.json", settings)
    _write_json(destination / "split_plan.json", plan)
    with _generation_lock(destination):
        raw, flat, rows, spec, _ = _prepare(root, destination / "cache", bins)
        truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
        val = np.flatnonzero(rows.event.isin(plan["validation"]))
        runs = []
        for size, ids in plan["training_sets"].items():
            train = np.flatnonzero(rows.event.isin(ids))
            for kind in ("mlp", "cnn"):
                for seed in seeds:
                    name = f"{kind}_n{size}_seed{seed}"
                    info = _fit(kind, raw, flat, truth, train, val, spec, destination / name,
                                seed=seed, epochs=epochs, patience=patience, batch_size=batch_size)
                    record = {"model": kind, "train_events": int(size), "seed": seed, "directory": name,
                              "validation_mean_error_mm": info["validation_mean_error_mm"],
                              "selected_epoch": info["selected_epoch"], "training_seconds": info["training_seconds"]}
                    runs.append(record)
                    print(f"trained {name}: validation={record['validation_mean_error_mm']:.4f} mm", flush=True)
        run_frame = pd.DataFrame(runs)
        run_frame.to_csv(destination / "validation_runs.csv", index=False)
        chosen = run_frame.groupby(["model", "train_events"]).validation_mean_error_mm.mean().idxmin()
        _write_json(destination / "selected_model.json", {
            "model": chosen[0], "train_events": int(chosen[1]), "representative_seed": int(seeds[0]),
            "directory": f"{chosen[0]}_n{chosen[1]}_seed{seeds[0]}",
            "criterion": settings["selection_rule"],
        })
        # Test access starts here, after every training/validation choice is frozen.
        frames = []
        for run in runs:
            directory = destination / run["directory"]
            info = json.loads((directory / "model.json").read_text())
            if run["model"] == "cnn":
                model = TemporalCNN(raw.shape[1])
                model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
            else:
                artifact = joblib.load(directory / "model.joblib")
            for split in ("test_id", "test_ood"):
                indices = np.flatnonzero(rows.event.isin(plan[split]))
                center, scale = np.asarray(info["target_center_mm"]), info["target_scale_mm"]
                if run["model"] == "cnn":
                    prediction = _cnn_predict(model, raw, indices, center, scale, batch_size)
                else:
                    prediction = np.concatenate([artifact["estimator"].predict(flat[idx]) * scale + center
                                                 for idx in _batch_indices(indices, batch_size)])
                frame = rows.iloc[indices][["event", "event_uid", "shot", "geometry", "propagation_law",
                                           "epicenter_region", "strength_band", "event_onset_ms"]].copy()
                frame["true_x_mm"], frame["true_y_mm"] = truth[indices, 0], truth[indices, 1]
                frame["predicted_x_mm"], frame["predicted_y_mm"] = prediction[:, 0], prediction[:, 1]
                frame["error_mm"] = np.linalg.norm(prediction - truth[indices], axis=1)
                frame["centroid_error_mm"] = np.linalg.norm(rows.iloc[indices][["centroid_x_mm", "centroid_y_mm"]].to_numpy() - truth[indices], axis=1)
                frame["constant_error_mm"] = np.linalg.norm(center - truth[indices], axis=1)
                for key in ("model", "train_events", "seed"):
                    frame[key] = run[key]
                frame["split"] = split
                frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(destination / "test_predictions.csv", index=False)
        reports, conditions = [], []
        for keys, group in predictions.groupby(["model", "train_events", "split"]):
            record = {"model": keys[0], "train_events": int(keys[1]), "split": keys[2], **summarize(group),
                      "seed_mean_error_sd_mm": float(group.groupby("seed").error_mm.mean().std(ddof=0)),
                      "centroid_mean_error_mm": float(group.centroid_error_mm.mean()),
                      "constant_mean_error_mm": float(group.constant_error_mm.mean())}
            group = group.copy()
            group["difference_from_centroid_mm"] = group.error_mm - group.centroid_error_mm
            record["paired_difference_from_centroid_95pct_mm"] = _mean_interval(group, "difference_from_centroid_mm")
            reports.append(record)
            for column in ("geometry", "propagation_law", "epicenter_region", "strength_band"):
                for value, subgroup in group.groupby(column):
                    conditions.append({"model": keys[0], "train_events": int(keys[1]), "split": keys[2],
                                       "condition": column, "value": str(value), **summarize(subgroup)})
        _write_json(destination / "metrics.json", {"scope": "synthetic fixed-device conditional single-source localization",
                    "uncertainty": "event bootstrap across test events, averaging their shots/seeds; conditional on fitted models",
                    "learning_curve": reports, "by_condition": conditions})
    return reports


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--bins", type=int, default=16)
    train.add_argument("--threads", type=int, default=2)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        run_experiment(args.dataset, args.plan, args.output, seeds=args.seeds, epochs=args.epochs,
                       patience=args.patience, batch_size=args.batch_size, bins=args.bins, threads=args.threads)
    else:
        if args.output.exists():
            parser.error("prediction output already exists")
        prediction = predict_cnn(args.model, args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data = load_syndromes(args.input)
        events, shots, _ = data["detector_events"].shape
        ids = data.get("event_ids", np.arange(events))
        frame = pd.DataFrame(prediction, columns=["predicted_x_mm", "predicted_y_mm"])
        frame.insert(0, "shot", np.tile(np.arange(shots), events))
        frame.insert(0, "event", np.repeat(ids, shots))
        frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
