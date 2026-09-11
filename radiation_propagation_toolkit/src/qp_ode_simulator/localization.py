"""Single-shot, fixed-layout epicenter regression from Stim detector bits.

Inference reads only detector bits and circuit geometry/timing. Simulator truth is
used for supervised targets and evaluation, never for feature construction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse


SCHEMA_VERSION = 1
GEOMETRY_KEYS = (
    "detector_coords",
    "circuit_qubit_ids",
    "circuit_grid_coords",
    "circuit_physical_coords_mm",
    "circuit_qubit_roles",
    "round_start_time_ms",
    "round_time_ms",
    "gate_slice_duration_ms",
)


def _learning_dependencies():
    try:
        import joblib
        import sklearn
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as exc:
        raise ImportError("Install localization support: pip install -e '.[localization]'") from exc
    return joblib, sklearn, ExtraTreesRegressor


def load_syndromes(path: str | Path) -> dict[str, np.ndarray]:
    """Load only the observable data and circuit metadata needed by the model."""
    source = Path(path)
    if source.is_dir():
        source = source / "stim_syndrome_events.npz"
    with np.load(source, allow_pickle=False) as archive:
        required = ("detector_events", *GEOMETRY_KEYS)
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"missing syndrome input arrays: {sorted(missing)}")
        result = {key: archive[key] for key in required}
        if "event_is_control" in archive:
            result["event_is_control"] = archive["event_is_control"]
        for key in ("event_ids", "event_uids", "circuit_id"):
            if key in archive:
                result[key] = archive[key]
    bits = result["detector_events"]
    if bits.ndim == 1:
        bits = bits[None, None, :]
    elif bits.ndim == 2:
        bits = bits[:, None, :]
    if bits.ndim != 3 or any(size == 0 for size in bits.shape):
        raise ValueError("detector_events must be non-empty (event, shot, detector) bits")
    if not np.isin(bits, [0, 1]).all():
        raise ValueError("detector_events must contain binary observations, not probabilities")
    result["detector_events"] = bits.astype(np.uint8, copy=False)
    for key in GEOMETRY_KEYS:
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} must be finite")
    return result


def feature_specification(data: Mapping[str, np.ndarray], bins: int = 16) -> dict:
    """Freeze the detector ordering, layout and acquisition schedule at training."""
    coords = np.asarray(data["detector_coords"])
    rounds = len(data["round_time_ms"])
    if coords.shape != (data["detector_events"].shape[-1], 3):
        raise ValueError("detector_coords must have shape (detector, 3)")
    if not isinstance(bins, int) or bins < 1 or bins > rounds - 1:
        raise ValueError("bins must be an integer between 1 and rounds - 1")
    if np.any(coords[:, 2] != np.floor(coords[:, 2])):
        raise ValueError("this extractor requires integer Stim detector round coordinates")
    if np.any(coords[:, 2] < 0) or np.any(coords[:, 2] > rounds):
        raise ValueError("detector round coordinates fall outside the circuit schedule")
    # Initial/final boundaries have different check coverage; use interior rounds.
    interior = (coords[:, 2] >= 1) & (coords[:, 2] < rounds)
    sites = np.unique(coords[interior, :2], axis=0)
    if len(sites) == 0:
        raise ValueError("no interior syndrome detectors")
    grid = np.asarray(data["circuit_grid_coords"])
    physical = np.asarray(data["circuit_physical_coords_mm"])
    if grid.ndim != 2 or grid.shape[1] != 2 or physical.shape != grid.shape:
        raise ValueError("circuit coordinate arrays must have matching (qubit, 2) shapes")
    design = np.column_stack((grid, np.ones(len(grid))))
    transform, _, rank, _ = np.linalg.lstsq(design, physical, rcond=None)
    if rank != 3 or not np.allclose(design @ transform, physical, atol=1e-5):
        raise ValueError("circuit grid must map affinely to physical coordinates")
    return {
        "schema_version": SCHEMA_VERSION,
        "circuit_id": str(data["circuit_id"].item()) if "circuit_id" in data else None,
        "bins": bins,
        "geometry": {key: np.array(data[key], copy=True) for key in GEOMETRY_KEYS},
        "sites_grid": sites,
        "sites_mm": np.column_stack((sites, np.ones(len(sites)))) @ transform,
        "layout_center_mm": 0.5 * (physical.min(axis=0) + physical.max(axis=0)),
    }


def extract_features(data: Mapping[str, np.ndarray], spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return one feature row per shot and its (site, time-bin) detector rates."""
    if spec.get("circuit_id") is not None and (
        "circuit_id" not in data or str(data["circuit_id"].item()) != spec["circuit_id"]
    ):
        raise ValueError("input circuit_id differs from training")
    for key in GEOMETRY_KEYS:
        actual, expected = np.asarray(data[key]), np.asarray(spec["geometry"][key])
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-7):
            raise ValueError(f"input {key} differs from training; retrain for this circuit/layout")
    bits = np.asarray(data["detector_events"])
    coords = np.asarray(data["detector_coords"])
    if bits.ndim != 3 or bits.shape[-1] != len(coords) or not np.isin(bits, [0, 1]).all():
        raise ValueError("detector_events must be binary (event, shot, detector) observations")
    rounds, bins = len(data["round_time_ms"]), int(spec["bins"])
    sites = spec["sites_grid"]
    indices = np.flatnonzero((coords[:, 2] >= 1) & (coords[:, 2] < rounds))
    site_lookup = {tuple(site): i for i, site in enumerate(sites)}
    site_index = np.asarray([site_lookup[tuple(coords[i, :2])] for i in indices])
    time_bin = ((coords[indices, 2] - 1) * bins // (rounds - 1)).astype(int)
    groups = site_index * bins + time_bin
    counts = np.bincount(groups, minlength=len(sites) * bins)
    if np.any(counts == 0):
        raise ValueError("some site/time bins have no detectors; reduce --bins")
    pooling = sparse.csr_matrix(
        (1.0 / counts[groups], (groups, indices)),
        shape=(len(sites) * bins, len(coords)), dtype=np.float32,
    )
    flat = bits.reshape(-1, bits.shape[-1])
    rates = np.asarray(pooling @ flat.T).T.reshape(len(flat), len(sites), bins)
    total = np.sum(rates * counts.reshape(len(sites), bins), axis=2)
    total /= counts.reshape(len(sites), bins).sum(axis=1)
    relative = rates / (rates.mean(axis=1, keepdims=True) + 1e-3)
    features = np.concatenate(
        [rates.reshape(len(flat), -1), relative.reshape(len(flat), -1),
         np.diff(rates, axis=2).reshape(len(flat), -1), total], axis=1,
    ).astype(np.float32)
    return features, rates


def split_events(event_ids: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    """Assign whole events to 60/20/20 splits before expanding their shots."""
    ids = np.asarray(event_ids, dtype=int)
    if len(ids) < 15 or len(np.unique(ids)) != len(ids):
        raise ValueError("training requires at least 15 distinct non-control events")
    shuffled = np.random.default_rng(seed).permutation(ids)
    holdout = max(1, int(np.floor(0.2 * len(ids))))
    return {
        "train": np.sort(shuffled[2 * holdout:]),
        "validation": np.sort(shuffled[holdout:2 * holdout]),
        "test": np.sort(shuffled[:holdout]),
    }


def location_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    errors = np.linalg.norm(np.asarray(truth) - np.asarray(prediction), axis=1)
    return {
        "samples": len(errors),
        "mean_error_mm": float(errors.mean()),
        "median_error_mm": float(np.median(errors)),
        "p90_error_mm": float(np.quantile(errors, 0.9)),
        "p95_error_mm": float(np.quantile(errors, 0.95)),
        "within_1mm_fraction": float(np.mean(errors <= 1.0)),
        "within_2mm_fraction": float(np.mean(errors <= 2.0)),
    }


def syndrome_centroid(rates: np.ndarray, spec: dict) -> np.ndarray:
    """Heuristic baseline; subtract the observed first bin, without true onset."""
    weights = np.maximum(rates.mean(axis=2) - rates[:, :, 0], 0)
    total = weights.sum(axis=1, keepdims=True)
    result = np.broadcast_to(spec["layout_center_mm"], (len(rates), 2)).copy()
    np.divide(weights @ spec["sites_mm"], total, out=result, where=total > 0)
    return result


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def train_localizer(
    dataset: str | Path,
    output: str | Path,
    *,
    bins: int = 16,
    seed: int = 42,
    trees: int = 256,
    jobs: int = 2,
) -> dict:
    """Train a conditional single-source regressor and evaluate untouched events."""
    joblib, sklearn, regressor_type = _learning_dependencies()
    source, destination = Path(dataset), Path(output)
    if trees < 1 or jobs == 0:
        raise ValueError("trees must be positive and jobs must be nonzero")
    if source.resolve() == destination.resolve():
        raise ValueError("model output must differ from the dataset directory")
    data = load_syndromes(source)
    labels = pd.read_csv(source / "true_parameters.csv")
    n_events, shots, _ = data["detector_events"].shape
    required = {"event", "epicenter_row", "epicenter_col", "is_control", "source_count"}
    if not required.issubset(labels.columns):
        raise ValueError(f"true_parameters.csv requires columns: {sorted(required)}")
    if len(labels) != n_events or not np.array_equal(labels["event"], np.arange(n_events)):
        raise ValueError("label rows must match archive events in order, numbered 0..N-1")
    if not labels["is_control"].isin([True, False, 0, 1]).all():
        raise ValueError("is_control labels must be booleans")
    controls = labels["is_control"].to_numpy(dtype=bool)
    if "event_is_control" in data and not np.array_equal(controls, data["event_is_control"]):
        raise ValueError("archive and CSV control labels disagree")
    usable = np.flatnonzero(~controls)
    if not np.all(labels.iloc[usable]["source_count"].to_numpy() == 1):
        raise ValueError("localization currently supports single-source events only")
    targets = labels[["epicenter_row", "epicenter_col"]].to_numpy(dtype=float)
    if not np.isfinite(targets[usable]).all():
        raise ValueError("non-control epicenter coordinates must be finite")
    splits = split_events(usable, seed)
    spec = feature_specification(data, bins)
    features, rates = extract_features(data, spec)
    truth = np.repeat(targets, shots, axis=0)
    sample_events = np.repeat(np.arange(n_events), shots)
    sample_shots = np.tile(np.arange(shots), n_events)
    masks = {name: np.isin(sample_events, ids) for name, ids in splits.items()}
    _new_directory(destination)
    candidates = []
    best = None
    best_error = np.inf
    # Two prespecified settings; the test set is never used to select a model.
    for leaf in (1, 4):
        estimator = regressor_type(
            n_estimators=trees, min_samples_leaf=leaf, max_features=0.8,
            random_state=seed, n_jobs=jobs,
        )
        estimator.fit(features[masks["train"]], truth[masks["train"]])
        metrics = location_metrics(
            truth[masks["validation"]], estimator.predict(features[masks["validation"]]),
        )
        candidates.append({"min_samples_leaf": leaf, "validation": metrics})
        if metrics["mean_error_mm"] < best_error:
            best_error = metrics["mean_error_mm"]
            best = estimator
    training_center = targets[splits["train"]].mean(axis=0)
    model = {
        "schema_version": SCHEMA_VERSION,
        "estimator": best,
        "features": spec,
        "sklearn_version": sklearn.__version__,
        "training_center_mm": training_center,
        "scope": "single-source event windows, single shot, fixed circuit/layout; no event detector",
        "coordinate_convention": "x_mm=epicenter_row; y_mm=epicenter_col",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": "ExtraTreesRegressor",
        "scope": model["scope"],
        "seed": seed,
        "features": int(features.shape[1]),
        "time_bins": bins,
        "excluded_controls": int(controls.sum()),
        "shots_per_event": shots,
        "independent_events": {name: len(ids) for name, ids in splits.items()},
        "split_unit": "original event; all shots of an event stay together",
        "model_selection": candidates,
        "selected_min_samples_leaf": best.min_samples_leaf,
        "trees": trees,
        "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "dataset_sha256": _hash_file(source / "stim_syndrome_events.npz"),
        "labels_sha256": _hash_file(source / "true_parameters.csv"),
        "localization_source_sha256": _hash_file(Path(__file__)),
        "evaluation": {},
    }
    for name in ("validation", "test"):
        mask = masks[name]
        prediction = best.predict(features[mask])
        centroid = syndrome_centroid(rates[mask], spec)
        center = np.broadcast_to(training_center, prediction.shape)
        report["evaluation"][name] = {
            "model": location_metrics(truth[mask], prediction),
            "training_mean_baseline": location_metrics(truth[mask], center),
            "syndrome_centroid_baseline": location_metrics(truth[mask], centroid),
        }
        frame = pd.DataFrame({
            "event": sample_events[mask], "shot": sample_shots[mask],
            "true_x_mm": truth[mask, 0], "true_y_mm": truth[mask, 1],
            "predicted_x_mm": prediction[:, 0], "predicted_y_mm": prediction[:, 1],
            "error_mm": np.linalg.norm(truth[mask] - prediction, axis=1),
            "centroid_x_mm": centroid[:, 0], "centroid_y_mm": centroid[:, 1],
        })
        # Diagnostics are attached after prediction, exclusively for error analysis.
        for column in (
            "axis_ratio", "angle_degrees", "apparent_speed_m_per_s",
            "diffusion_coefficient_mm2_per_ms", "initial_lambda_mm",
            "maximum_lambda_mm", "maximum_distance_mm",
            "qp_generation_scale_per_us", "qp_minimum_t1_us", "event_onset_ms",
        ):
            if column in labels:
                frame[column] = labels.iloc[sample_events[mask]][column].to_numpy()
        grouped = {}
        for column in ("geometry", "propagation_law", "epicenter_region"):
            if column not in labels:
                continue
            frame[column] = labels.iloc[sample_events[mask]][column].to_numpy()
            grouped[column] = {}
            for value in sorted(frame[column].unique()):
                selection = frame[column].to_numpy() == value
                metrics = location_metrics(truth[mask][selection], prediction[selection])
                metrics["independent_events"] = int(frame.loc[selection, "event"].nunique())
                grouped[column][str(value)] = metrics
        report["evaluation"][name]["by_condition"] = grouped
        frame.to_csv(destination / f"{name}_predictions.csv", index=False)
    joblib.dump(model, destination / "model.joblib", compress=3)
    (destination / "splits.json").write_text(
        json.dumps({name: ids.tolist() for name, ids in splits.items()}, indent=2) + "\n"
    )
    (destination / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def predict_epicenters(model_path: str | Path, input_path: str | Path) -> pd.DataFrame:
    """Return millimetre coordinates from syndrome bits; no truth CSV is read.

    Only load model.joblib files from trusted sources (joblib uses pickle).
    """
    joblib, sklearn, _ = _learning_dependencies()
    model = joblib.load(model_path)
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported localization model schema")
    if model["sklearn_version"] != sklearn.__version__:
        raise ValueError("use the scikit-learn version recorded in the model, or retrain")
    data = load_syndromes(input_path)
    features, _ = extract_features(data, model["features"])
    prediction = model["estimator"].predict(features)
    if "target_scale_mm" in model:
        prediction = prediction * model["target_scale_mm"] + model["target_center_mm"]
    events, shots, _ = data["detector_events"].shape
    return pd.DataFrame({
        "event": np.repeat(data.get("event_ids", np.arange(events)), shots),
        "shot": np.tile(np.arange(shots), events),
        "x_mm": prediction[:, 0], "y_mm": prediction[:, 1],
    })
