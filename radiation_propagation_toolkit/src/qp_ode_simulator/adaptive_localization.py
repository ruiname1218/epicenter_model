"""Validation-selected segmentation, empirical templates, and CNN shrinkage.

Pattern matching here is a training-syndrome nearest-template baseline, NOT a
physical forward-model likelihood. Oracle segmentation is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import _write_json, dataset_manifest, iter_shards
from .learning_plan import validate_plan
from .localization import _new_directory, _hash_file, load_syndromes
from .temporal_localization import _prepare, detector_series, _json_spec, summarize, _mean_interval
from .temporal_diagnosis import DiagnosisCNN, predict_arrays
from .window_calibration import validate_calibration


def estimated_cuts(raw):
    """Observed aggregate upward change point, not an oracle physical onset."""
    ticks = raw.shape[-1]
    if ticks < 12:
        raise ValueError("at least twelve interior rounds required")
    points = np.arange(max(2, ticks//64), ticks//2+1, max(1, ticks//128))
    sums = raw.mean(1).cumsum(1)
    before = sums[:, points-1] / points
    after = (sums[:, -1, None]-sums[:, points-1]) / (ticks-points)
    score = (after-before) * np.sqrt(points*(ticks-points)/ticks)
    return points[np.argmax(score, axis=1)]


def segmented_features(raw, cuts, quiet):
    """Identical representation for fixed, estimated and true-time cut points."""
    raw = np.asarray(raw)
    n, sites, ticks = raw.shape
    cuts = np.asarray(cuts, dtype=int)
    quiet = np.asarray(quiet)
    if cuts.shape != (n,) or np.any(cuts < 1) or np.any(cuts > ticks-3):
        raise ValueError("cuts must leave a nonempty prefix and three suffix bins")
    if quiet.shape != (sites,) or not np.isfinite(quiet).all() or np.any((quiet <= 0) | (quiet >= 1)):
        raise ValueError("invalid quiet rates")
    cumulative = np.concatenate((np.zeros((n, sites, 1)), raw.cumsum(2)), axis=2)
    points = np.stack((np.zeros(n, dtype=int), cuts, cuts+(ticks-cuts)//3,
                       cuts+2*(ticks-cuts)//3, np.full(n, ticks)), axis=1)
    rates = []
    for column in range(4):
        lo, hi = points[:, column], points[:, column+1]
        total = cumulative[np.arange(n), :, hi]-cumulative[np.arange(n), :, lo]
        rates.append(total / (hi-lo)[:, None])
    rates = np.stack(rates, axis=2)
    excess = rates-quiet[None, :, None]
    # Regularized positive spatial profile retains amplitude separately.
    positive = np.maximum(excess[:, :, 1:], 0)
    relative = positive / (positive.mean(1, keepdims=True)+.003)
    return np.concatenate((excess.reshape(n, -1), relative.reshape(n, -1),
                           cuts[:, None]/ticks), axis=1).astype(np.float32)


def shrink(prediction, center, signal, threshold):
    """Heuristic shrinkage toward training prior; weight is NOT a probability."""
    if threshold < 0:
        raise ValueError("negative shrinkage threshold")
    if threshold == 0:
        return np.array(prediction, copy=True)
    signal = np.maximum(np.asarray(signal), 0)
    weight = signal**2 / (signal**2+threshold**2)
    return np.asarray(center)+weight[:, None]*(prediction-np.asarray(center))


def representations(raw, info, onset=None):
    fixed = np.full(len(raw), info["fixed_cut"], dtype=int)
    result = {"fixed": segmented_features(raw, fixed, info["quiet"]),
              "estimated": segmented_features(raw, estimated_cuts(raw), info["quiet"])}
    if onset is not None:
        times = np.asarray(info["specification"]["geometry"]["round_time_ms"])[1:]
        cuts = np.searchsorted(times, onset)
        result["oracle"] = segmented_features(raw, cuts, info["quiet"])
    return result


def cnn_predictions(raw, info, states):
    result = []
    for state, metadata in states:
        model = DiagnosisCNN(raw.shape[1], "original")
        model.load_state_dict(state)
        result.append(predict_arrays(model, raw, np.arange(len(raw)), np.asarray(metadata["target_center_mm"]),
                                     metadata["target_scale_mm"], None, metadata["time_center_ms"], metadata["time_scale_ms"]))
    return result[0], np.mean(result, axis=0)


def candidate_prediction(candidate, features, single, ensemble, signal, center):
    kind = candidate["kind"]
    if kind == "single":
        return single
    if kind == "ensemble":
        return ensemble
    if kind == "shrink":
        return shrink(ensemble, center, signal, candidate["threshold"])
    if kind == "constant":
        return np.broadcast_to(center, ensemble.shape).copy()
    if candidate["representation"] not in features:
        raise ValueError("oracle candidate requires true onset and is diagnostic only")
    values = features[candidate["representation"]]
    return np.mean([model.predict(values) for model in candidate["models"]], axis=0)


def fit(dataset, plan_path, cnn_experiment, calibration, output):
    root, destination, experiment = map(Path, (dataset, output, cnn_experiment))
    manifest = dataset_manifest(root)
    plan = json.loads(Path(plan_path).read_text())
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    validate_plan(plan, labels, manifest, root)
    prior = json.loads((experiment / "experiment.json").read_text())
    if prior["dataset_sha256"] != _hash_file(root / "dataset_manifest.json") or prior["plan_sha256"] != _hash_file(Path(plan_path)):
        raise ValueError("CNN training dataset or split mismatch")
    _new_directory(destination)
    torch.set_num_threads(2)
    raw, _, rows, spec, _ = _prepare(root, destination / "cache", 16)
    quiet = validate_calibration(calibration, manifest, spec)
    train = np.flatnonzero(rows.event.isin(plan["training_sets"][str(max(map(int, plan["training_sets"])))]))
    val = np.flatnonzero(rows.event.isin(plan["validation"]))
    truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
    times = np.asarray(spec["geometry"]["round_time_ms"])[1:]
    info = {"format": "adaptive_localization_v1", "specification": _json_spec(spec),
            "quiet": quiet["probabilities"], "fixed_cut": int(np.searchsorted(times, rows.iloc[train].event_onset_ms.median())),
            "center": truth[train].mean(0).tolist(), "dataset_sha256": _hash_file(root / "dataset_manifest.json"),
            "plan": plan, "source_sha256": _hash_file(Path(__file__)),
            "selection": "minimum validation mean distance, ordinary candidates only; freeze before fresh test",
            "reference_scope": "empirical training-syndrome nearest templates, not physical likelihood"}
    states = []
    for seed in (41, 42, 43):
        folder = experiment / f"original_seed{seed}"
        metadata = json.loads((folder / "model.json").read_text())
        if _hash_file(folder / "weights.pt") != metadata["weights_sha256"]:
            raise ValueError("CNN weight checksum mismatch")
        detector_series(next(iter_shards(root))[0], metadata["specification"])
        states.append((torch.load(folder / "weights.pt", map_location="cpu", weights_only=True), metadata))
    features = representations(raw, info, rows.event_onset_ms.to_numpy())
    single, ensemble = cnn_predictions(raw, info, states)
    signal = raw.mean((1, 2))-np.mean(info["quiet"])
    candidates = [{"name": "cnn_single", "kind": "single"}, {"name": "cnn_ensemble", "kind": "ensemble"},
                  {"name": "constant", "kind": "constant"}]
    for threshold in (.0005, .001, .002, .004, .008, .016):
        candidates.append({"name": f"shrink_{threshold}", "kind": "shrink", "threshold": threshold})
    for representation, values in features.items():
        for alpha in (10., 100., 1000.):
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(values[train], truth[train])
            candidates.append({"name": f"{representation}_ridge_{alpha}", "kind": "regressor", "representation": representation, "models": [estimator]})
        for leaf in (4, 12):
            estimators = [ExtraTreesRegressor(n_estimators=128, min_samples_leaf=leaf, random_state=seed, n_jobs=2).fit(values[train], truth[train])
                          for seed in (41, 42, 43)]
            candidates.append({"name": f"{representation}_trees_{leaf}", "kind": "regressor", "representation": representation, "models": estimators})
        # Aggregate only TRAINING shots into independent-event templates.
        event_ids = rows.iloc[train].event.to_numpy()
        unique = np.unique(event_ids)
        template_x = np.stack([values[train][event_ids == event].mean(0) for event in unique])
        template_y = np.stack([truth[train][event_ids == event].mean(0) for event in unique])
        for neighbors in (8, 24, 64):
            estimator = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=min(neighbors, len(unique)), weights="distance"))
            estimator.fit(template_x, template_y)
            candidates.append({"name": f"{representation}_templates_{neighbors}", "kind": "regressor", "representation": representation, "models": [estimator]})
    validation = []
    for candidate in candidates:
        prediction = candidate_prediction(candidate, {k:v[val] for k,v in features.items()}, single[val], ensemble[val], signal[val], info["center"])
        validation.append({"name": candidate["name"], "diagnostic_only": candidate.get("representation") == "oracle",
                           "validation_error_mm": float(np.linalg.norm(prediction-truth[val], axis=1).mean())})
    frame = pd.DataFrame(validation)
    selected = frame[~frame.diagnostic_only].sort_values("validation_error_mm", kind="stable").iloc[0]["name"]
    info["selected"] = selected
    frame.to_csv(destination / "validation.csv", index=False)
    joblib.dump({"candidates": candidates, "states": states}, destination / "models.joblib", compress=3)
    info["artifact_sha256"] = _hash_file(destination / "models.joblib")
    _write_json(destination / "model.json", info)
    print(frame.sort_values("validation_error_mm").to_string(index=False), flush=True)
    print(f"FROZEN SELECTION: {selected}", flush=True)


def load_model(directory):
    root = Path(directory)
    info = json.loads((root / "model.json").read_text())
    if info.get("format") != "adaptive_localization_v1" or _hash_file(root / "models.joblib") != info["artifact_sha256"]:
        raise ValueError("invalid adaptive model artifact")
    return info, joblib.load(root / "models.joblib")


def export_selected(directory, output):
    """Export the frozen candidate only; do not reselect using test results."""
    info, artifact = load_model(directory)
    candidate = next(c for c in artifact["candidates"] if c["name"] == info["selected"])
    if candidate.get("representation") == "oracle":
        raise ValueError("cannot deploy a privileged oracle")
    destination = Path(output)
    _new_directory(destination)
    states = [] if candidate["kind"] == "regressor" else artifact["states"]
    joblib.dump({"candidates": [candidate], "states": states}, destination / "models.joblib", compress=3)
    info = {**info, "inference_only": True, "parent_artifact_sha256": info["artifact_sha256"],
            "artifact_sha256": _hash_file(destination / "models.joblib")}
    _write_json(destination / "model.json", info)


def predict(directory, input_path, candidate_name=None):
    """Observed bits only, including for data-dependent segmentation/shrinkage."""
    info, artifact = load_model(directory)
    name = candidate_name or info["selected"]
    candidate = next(c for c in artifact["candidates"] if c["name"] == name)
    if candidate.get("representation") == "oracle":
        raise ValueError("oracle candidate is diagnostic only")
    raw = detector_series(load_syndromes(input_path), info["specification"])
    features = representations(raw, info)
    if candidate["kind"] == "regressor":
        # A feature regressor does not need to execute any CNN at inference.
        single = ensemble = None
    else:
        single, ensemble = cnn_predictions(raw, info, artifact["states"])
    return candidate_prediction(candidate, features, single, ensemble,
                                raw.mean((1, 2))-np.mean(info["quiet"]), info["center"])


def evaluate(directory, dataset, training_dataset, output):
    root, old, destination = map(Path, (dataset, training_dataset, output))
    info, artifact = load_model(directory)
    if info.get("inference_only"):
        raise ValueError("comparison evaluation requires the full experiment artifact")
    if _hash_file(old / "dataset_manifest.json") != info["dataset_sha256"]:
        raise ValueError("training manifest mismatch")
    old_manifest, new_manifest = dataset_manifest(old), dataset_manifest(root)
    if (old_manifest["generation"]["device_seed"] != new_manifest["generation"]["device_seed"]
            or old_manifest["circuit_id"] != new_manifest["circuit_id"]):
        raise ValueError("test device or circuit mismatch")
    previous = pd.concat([p for _, p in iter_shards(old)], ignore_index=True)
    fresh = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    for field in ("event_uid", "generation_seed", "syndrome_seed"):
        if set(previous[field]) & set(fresh[field]):
            raise ValueError("fresh test overlaps training data")
    if fresh.is_control.astype(bool).any() or (fresh.source_count != 1).any():
        raise ValueError("single-event conditional evaluation only")
    detector_series(next(iter_shards(root))[0], info["specification"])
    _new_directory(destination)
    torch.set_num_threads(2)
    _write_json(destination / "evaluation.json", {"model_sha256": info["artifact_sha256"], "selected_before_test": info["selected"],
                "dataset_sha256": _hash_file(root / "dataset_manifest.json"), "source_sha256": _hash_file(Path(__file__)),
                "policy": "no fitting or model reselection on these events"})
    raw, _, rows, spec, _ = _prepare(root, destination / "cache", 16)
    features = representations(raw, info, rows.event_onset_ms.to_numpy())
    single, ensemble = cnn_predictions(raw, info, artifact["states"])
    signal = raw.mean((1, 2))-np.mean(info["quiet"])
    truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
    held = np.ones(len(rows), dtype=bool)
    for key, value in info["plan"]["held_out_conjunction"].items():
        held &= rows[key].to_numpy() == value
    frames = []
    for candidate in artifact["candidates"]:
        prediction = candidate_prediction(candidate, features, single, ensemble, signal, info["center"])
        frame = rows[["event", "shot", "strength_band", "geometry", "propagation_law", "epicenter_region", "event_onset_ms"]].copy()
        frame["candidate"] = candidate["name"]
        frame["split"] = np.where(held, "test_ood", "test_id")
        frame["true_x_mm"], frame["true_y_mm"] = truth[:,0], truth[:,1]
        frame["predicted_x_mm"], frame["predicted_y_mm"] = prediction[:,0], prediction[:,1]
        frame["error_mm"] = np.linalg.norm(prediction-truth, axis=1)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(destination / "predictions.csv", index=False)
    metrics, paired = [], []
    for (name, split), group in predictions.groupby(["candidate", "split"]):
        for band in ("all", 0, 1, 2):
            sub = group if band == "all" else group[group.strength_band == band]
            if not len(sub):
                continue
            metrics.append({"candidate": name, "split": split, "band": band, **summarize(sub)})
            for reference in ("cnn_single", "cnn_ensemble", "fixed_ridge_100.0"):
                baseline = predictions[(predictions.candidate == reference) & (predictions.split == split)]
                if band != "all":
                    baseline = baseline[baseline.strength_band == band]
                delta = (sub.set_index(["event", "shot"]).error_mm-baseline.set_index(["event", "shot"]).error_mm).rename("error_mm").reset_index()
                paired.append({"candidate": name, "reference": reference, "split": split, "band": band,
                               "difference_mm": float(delta.error_mm.mean()), "paired_event_95pct_mm": _mean_interval(delta)})
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "paired.json", paired)
    print(pd.DataFrame(metrics).query("band == 'all'")[["candidate", "split", "mean_error_mm"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--cnn-experiment")
    parser.add_argument("--calibration")
    parser.add_argument("--model")
    parser.add_argument("--training-dataset")
    args = parser.parse_args()
    if args.model:
        if not args.training_dataset:
            parser.error("--training-dataset required")
        evaluate(args.model, args.dataset, args.training_dataset, args.output)
    else:
        if not all((args.plan, args.cnn_experiment, args.calibration)):
            parser.error("--plan, --cnn-experiment, --calibration required")
        fit(args.dataset, args.plan, args.cnn_experiment, args.calibration, args.output)


if __name__ == "__main__":
    main()
