"""Paired 2x2 ablation: observation duration and independent quiet calibration.

Calibration uses observed detector bits, never event onset or true relaxation
fields. Test metrics are computed only after every validation choice is frozen.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .api import run_stim_pipeline
from .dataset import _generation_lock, _write_json, dataset_manifest, iter_shards
from .learning_plan import validate_plan
from .localization import GEOMETRY_KEYS, _hash_file, _new_directory, feature_specification, load_syndromes
from .temporal_localization import (
    TemporalCNN, _cnn_predict, _fit, _json_spec, _mean_interval, _prepare, detector_series, summarize,
)


class WindowView:
    """Lazy prefix and optional quiet-rate normalization; no whole-cache copy."""

    def __init__(self, raw, rounds, probabilities=None):
        if type(rounds) is not int or rounds < 3 or rounds - 1 > raw.shape[2]:
            raise ValueError("window must contain at least three rounds within source")
        self.raw, self.times = raw, rounds - 1
        self.shape = (raw.shape[0], raw.shape[1], self.times)
        self.probabilities = None if probabilities is None else np.asarray(probabilities, dtype=np.float32)
        if self.probabilities is not None:
            if (self.probabilities.shape != (raw.shape[1],) or not np.isfinite(self.probabilities).all()
                    or np.any(self.probabilities <= 0) or np.any(self.probabilities >= 1)):
                raise ValueError("quiet probabilities must be finite, per-site and strictly between zero and one")

    def __getitem__(self, indices):
        values = np.array(self.raw[indices, :, :self.times], dtype=np.float32)
        if self.probabilities is not None:
            p = self.probabilities[:, None]
            # TemporalCNN multiplies inputs by ten. Compensate so its first
            # convolution sees (bit-p)/sqrt(p*(1-p)), without eventwise scaling.
            values = (values - p) / (10 * np.sqrt(p * (1 - p)))
        return values


def generate_calibration(dataset, output, *, shots=256, source_seed=20260922, syndrome_seed=20260923):
    if type(shots) is not int or shots < 2:
        raise ValueError("quiet calibration needs at least two shots")
    root, destination = Path(dataset), Path(output)
    # A complete dataset is not needed to measure independent quiet calibration.
    manifest = dataset_manifest(root, require_complete=False)
    _new_directory(destination)
    base, stim = copy.deepcopy(manifest["simulator_config"]), copy.deepcopy(manifest["stim_config"])
    base.update(n_events=1, seed=source_seed, device_seed=manifest["generation"]["device_seed"])
    base["event_timing"]["control_fraction"] = 1.
    stim.update(shots_per_event=shots, seed=syndrome_seed)
    result = run_stim_pipeline(base, stim)
    if not bool(result.simulation.parameters.iloc[0].is_control):
        raise ValueError("quiet acquisition unexpectedly contains radiation")
    arrays = result.syndrome_arrays
    path = destination / "quiet_syndromes.npz"
    np.savez_compressed(path, detector_events=arrays["detector_events"],
                        circuit_id=np.asarray(manifest["circuit_id"]),
                        **{key: arrays[key] for key in GEOMETRY_KEYS})
    data = load_syndromes(path)
    spec = feature_specification(data, 1)
    series = detector_series(data, spec)
    probabilities = series.mean(axis=(0, 2))
    WindowView(series, len(data["round_time_ms"]), probabilities)  # Validate finite usable rates.
    record = {"format": "quiet_detector_calibration_v1", "probabilities": probabilities.tolist(),
              "shots": shots, "rounds": len(data["round_time_ms"]),
              "observations_per_site": shots * series.shape[2],
              "raw_sha256": _hash_file(path), "source_seed": source_seed, "syndrome_seed": syndrome_seed,
              "device_seed": manifest["generation"]["device_seed"],
              "simulator_config": base, "stim_config": stim,
              "method": "per-site empirical interior detector rate from independent no-radiation shots",
              "circuit_id": manifest["circuit_id"], "generation_source_sha256": _hash_file(Path(__file__))}
    _write_json(destination / "calibration.json", record)
    return record


def validate_calibration(root, manifest, specification):
    root = Path(root)
    record = json.loads((root / "calibration.json").read_text())
    if record.get("format") != "quiet_detector_calibration_v1":
        raise ValueError("unsupported calibration")
    if record["device_seed"] != manifest["generation"]["device_seed"] or record["circuit_id"] != manifest["circuit_id"]:
        raise ValueError("calibration device or circuit differs")
    expected = copy.deepcopy(manifest["stim_config"])
    actual = copy.deepcopy(record["stim_config"])
    for config in (expected, actual):
        config.pop("seed", None)
        config.pop("shots_per_event", None)
    if actual != expected:
        raise ValueError("calibration acquisition settings differ")
    # Compare every simulator setting except event count/seeds and control flag.
    expected = copy.deepcopy(manifest["simulator_config"])
    actual = copy.deepcopy(record["simulator_config"])
    for config in (expected, actual):
        for key in ("seed", "n_events", "device_seed"):
            config.pop(key, None)
        config["event_timing"].pop("control_fraction", None)
    if actual != expected:
        raise ValueError("calibration device simulation settings differ")
    path = root / "quiet_syndromes.npz"
    if record["raw_sha256"] != _hash_file(path):
        raise ValueError("calibration raw checksum mismatch")
    observed = detector_series(load_syndromes(path), specification)
    rates = observed.mean(axis=(0, 2))
    if not np.allclose(rates, record["probabilities"], atol=1e-12, rtol=0):
        raise ValueError("saved calibration does not match observed bits")
    return record


def predict_window_model(model_directory, input_path, *, batch_size=128):
    root = Path(model_directory)
    metadata = json.loads((root / "model.json").read_text())
    if metadata.get("model") != "window_calibrated_cnn_v1" or batch_size < 1:
        raise ValueError("invalid window CNN artifact or batch size")
    if _hash_file(root / "weights.pt") != metadata["weights_sha256"]:
        raise ValueError("weights checksum mismatch")
    data = observation_prefix(load_syndromes(input_path), metadata["window_rounds"])
    specification = copy.deepcopy(metadata["specification"])
    expected = observation_prefix({**specification["geometry"],
                                   "detector_events": np.zeros((1, 1, len(specification["geometry"]["detector_coords"])), dtype=np.uint8)},
                                  metadata["window_rounds"])
    specification["geometry"] = {k: expected[k] for k in GEOMETRY_KEYS}
    specification["bins"] = min(specification["bins"], metadata["window_rounds"]-1)
    raw = detector_series(data, specification)
    view = WindowView(raw, metadata["window_rounds"], metadata["quiet_probabilities"])
    model = TemporalCNN(raw.shape[1])
    model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
    return _cnn_predict(model, view, np.arange(len(raw)), np.asarray(metadata["target_center_mm"]),
                        metadata["target_scale_mm"], batch_size)


def observation_prefix(data, rounds):
    """Use only available rounds; never require future bits for short inference."""
    if rounds < 3 or len(data["round_time_ms"]) < rounds:
        raise ValueError("insufficient observed rounds")
    result = {key: np.asarray(value) for key, value in data.items()}
    selected = result["detector_coords"][:, 2] < rounds
    result["detector_coords"] = result["detector_coords"][selected]
    result["detector_events"] = result["detector_events"][:, :, selected]
    for key in ("round_start_time_ms", "round_time_ms", "gate_slice_duration_ms"):
        result[key] = result[key][:rounds]
    return result


def _centroid(raw, indices, rounds, spec, round_duration_ms, batch_size=128):
    before = max(1, int(round(.064 / round_duration_ms)))
    predictions = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        values = np.array(raw[selected, :, :rounds-1], dtype=np.float32)
        weights = np.maximum(values.mean(axis=2) - values[:, :, :before].mean(axis=2), 0)
        total = weights.sum(axis=1, keepdims=True)
        result = np.broadcast_to(spec["layout_center_mm"], (len(selected), 2)).copy()
        np.divide(weights @ spec["sites_mm"], total, out=result, where=total > 0)
        predictions.append(result)
    return np.concatenate(predictions)


def run_ablation(dataset, plan_path, calibration, output, *, seeds=(41, 42, 43),
                 windows=(1024, 2048), epochs=40, patience=8, batch_size=64):
    import importlib.metadata
    from . import temporal_localization, learning_plan, dataset as dataset_module, localization
    root, plan_path, destination = Path(dataset), Path(plan_path), Path(output)
    manifest = dataset_manifest(root)
    plan = json.loads(plan_path.read_text())
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    validate_plan(plan, labels, manifest, root)
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)
            or len(windows) != 2 or len(set(windows)) != 2
            or any(type(w) is not int or w < 3 or w > manifest["stim_config"]["rounds"] for w in windows)
            or min(epochs, patience, batch_size) < 1):
        raise ValueError("invalid ablation settings")
    _new_directory(destination)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    settings = {"seeds": list(seeds), "windows_rounds": list(windows), "normalization": [False, True],
                "epochs": epochs, "patience": patience, "batch_size": batch_size,
                "dataset_manifest_sha256": _hash_file(root / "dataset_manifest.json"),
                "plan_sha256": _hash_file(plan_path),
                "calibration_sha256": _hash_file(Path(calibration) / "calibration.json"),
                "source_hashes": {Path(p).name: _hash_file(Path(p)) for p in
                                  (__file__, temporal_localization.__file__, learning_plan.__file__,
                                   dataset_module.__file__, localization.__file__)},
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "stim", "scikit-learn")},
                "selection": "smallest seed-mean validation distance; representative seed is first seed",
                "test_policy": "all models frozen before any test errors; same new test events for all four arms"}
    _write_json(destination / "experiment.json", settings)
    _write_json(destination / "split_plan.json", plan)
    with _generation_lock(destination):
        raw, flat, rows, spec, _ = _prepare(root, destination / "cache", min(16, manifest["stim_config"]["rounds"]-1))
        quiet = validate_calibration(calibration, manifest, spec)
        _write_json(destination / "calibration.json", quiet)
        train_ids = plan["training_sets"][str(max(map(int, plan["training_sets"])))]
        train = np.flatnonzero(rows.event.isin(train_ids))
        val = np.flatnonzero(rows.event.isin(plan["validation"]))
        truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
        runs = []
        for rounds in windows:
            for calibrated in (False, True):
                rates = quiet["probabilities"] if calibrated else None
                view = WindowView(raw, rounds, rates)
                for seed in seeds:
                    name = f"cnn_r{rounds}_{'calibrated' if calibrated else 'raw'}_seed{seed}"
                    info = _fit("cnn", view, flat, truth, train, val, spec, destination / name,
                                seed=seed, epochs=epochs, patience=patience, batch_size=batch_size)
                    info.update(model="window_calibrated_cnn_v1", window_rounds=rounds,
                                quiet_probabilities=rates, calibration_sha256=settings["calibration_sha256"] if calibrated else None,
                                normalization="(bit-p)/[10*sqrt(p*(1-p))] before CNN's factor ten" if calibrated else "original binary input")
                    _write_json(destination / name / "model.json", info)
                    runs.append({"rounds": rounds, "calibrated": calibrated, "seed": seed, "directory": name,
                                 "validation_mean_error_mm": info["validation_mean_error_mm"],
                                 "selected_epoch": info["selected_epoch"], "training_seconds": info["training_seconds"]})
                    print(f"trained {name}: validation={info['validation_mean_error_mm']:.4f} mm", flush=True)
        frame = pd.DataFrame(runs)
        frame.to_csv(destination / "validation_runs.csv", index=False)
        selected = frame.groupby(["rounds", "calibrated"]).validation_mean_error_mm.mean().idxmin()
        _write_json(destination / "selected_model.json", {
            "rounds": int(selected[0]), "calibrated": bool(selected[1]), "seed": seeds[0],
            "directory": f"cnn_r{selected[0]}_{'calibrated' if selected[1] else 'raw'}_seed{seeds[0]}"})
        # Fresh test observations are used for scoring only after this point.
        frames = []
        for run in runs:
            folder = destination / run["directory"]
            info = json.loads((folder / "model.json").read_text())
            view = WindowView(raw, run["rounds"], info["quiet_probabilities"])
            model = TemporalCNN(raw.shape[1])
            model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True))
            center = np.asarray(info["target_center_mm"])
            for split in ("test_id", "test_ood"):
                indices = np.flatnonzero(rows.event.isin(plan[split]))
                pred = _cnn_predict(model, view, indices, center, info["target_scale_mm"], batch_size)
                centroid = _centroid(raw, indices, run["rounds"], spec, manifest["stim_config"]["round_duration_ms"])
                out = rows.iloc[indices][["event", "event_uid", "shot", "strength_band", "geometry",
                                          "propagation_law", "epicenter_region", "event_onset_ms"]].copy()
                out["predicted_x_mm"], out["predicted_y_mm"] = pred[:, 0], pred[:, 1]
                out["true_x_mm"], out["true_y_mm"] = truth[indices, 0], truth[indices, 1]
                out["error_mm"] = np.linalg.norm(pred - truth[indices], axis=1)
                out["centroid_error_mm"] = np.linalg.norm(centroid - truth[indices], axis=1)
                out["constant_error_mm"] = np.linalg.norm(center - truth[indices], axis=1)
                out["split"] = split
                for key in ("rounds", "calibrated", "seed"):
                    out[key] = run[key]
                frames.append(out)
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(destination / "test_predictions.csv", index=False)
        summary = summarize_ablation(predictions, windows)
        _write_json(destination / "metrics.json", summary)
    return summary


def summarize_ablation(predictions, windows):
    summaries, conditions, contrasts = [], [], []
    for keys, group in predictions.groupby(["rounds", "calibrated", "split"]):
        settings = {"rounds": int(keys[0]), "calibrated": bool(keys[1]), "split": keys[2]}
        summaries.append({**settings, **summarize(group),
                          "centroid_mean_error_mm": float(group.centroid_error_mm.mean()),
                          "constant_mean_error_mm": float(group.constant_error_mm.mean())})
        for column in ("strength_band", "epicenter_region", "propagation_law"):
            for value, subgroup in group.groupby(column):
                conditions.append({**settings, "condition": column, "value": str(value), **summarize(subgroup)})
    short, long = sorted(windows)
    for split, group in predictions.groupby("split"):
        pivot = group.pivot(index=["event", "shot", "seed"], columns=["rounds", "calibrated"], values="error_mm")
        for name, target, reference in (
                ("long_minus_short_raw", (long, False), (short, False)),
                ("calibration_effect_short", (short, True), (short, False)),
                ("calibration_effect_long", (long, True), (long, False)),
                ("combined_minus_original", (long, True), (short, False))):
            difference = (pivot[target] - pivot[reference]).rename("difference").reset_index()
            contrasts.append({"split": split, "comparison": name,
                              "mean_difference_mm": float(difference.difference.mean()),
                              "paired_95pct_event_bootstrap_mm": _mean_interval(difference, "difference")})
    return {"scope": "synthetic fixed-device conditional localization; paired 2x2 pilot",
            "uncertainty": "event-cluster bootstrap over fixed fits; no hardware or predictive-interval claim",
            "groups": summaries, "by_condition": conditions, "paired_contrasts": contrasts}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--dataset", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--shots", type=int, default=256)
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--calibration", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--windows-rounds", type=int, nargs=2, default=[1024, 2048])
    train.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=8)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        generate_calibration(args.dataset, args.output, shots=args.shots)
    elif args.command == "train":
        run_ablation(args.dataset, args.plan, args.calibration, args.output,
                     windows=args.windows_rounds, seeds=args.seeds, epochs=args.epochs, patience=args.patience)
    else:
        if args.output.exists():
            parser.error("output already exists")
        prediction = predict_window_model(args.model, args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction, columns=["predicted_x_mm", "predicted_y_mm"]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
