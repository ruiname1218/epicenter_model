"""Paired window/strength/range experiment using real Stim detector samples.

One family shares position, shape, propagation and nuisance random streams across
all strength/range variants. Families, not shots or variants, define the split.
"""

from __future__ import annotations

import copy
import itertools
import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .api import run_stim_pipeline
from .dataset import _generation_lock, _identity, _seed, _write_json
from .localization import (
    GEOMETRY_KEYS, _hash_file, extract_features, feature_specification,
    load_syndromes, location_metrics, syndrome_centroid,
)


def validate_study(config: dict) -> None:
    n = config["families"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 36 or n % 12:
        raise ValueError("families must be a multiple of 12 and at least 36")
    for key in ("seed", "device_seed", "syndrome_seed", "shots", "trees"):
        value = config[key]
        minimum = 1 if key in {"shots", "trees"} else 0
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"invalid {key}")
    for key in ("generation_per_us", "range_scales", "windows_ms", "bin_widths_us"):
        values = np.asarray(config[key], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{key} must contain finite positive values")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"{key} must contain distinct values")
    held = config["held_out_condition"]
    if (not isinstance(held, list) or len(held) != 2
            or any(not isinstance(x, int) for x in held)
            or not 0 <= held[0] < len(config["generation_per_us"])
            or not 0 <= held[1] < len(config["range_scales"])):
        raise ValueError("invalid held_out_condition indices")
    if len(config["generation_per_us"]) * len(config["range_scales"]) < 2:
        raise ValueError("at least two conditions are required")


def family_splits(config: dict) -> dict[str, list[int]]:
    """Last two balanced cycles are validation/test; every cycle has 12 strata."""
    validate_study(config)
    n = config["families"]
    return {"train": list(range(n - 24)), "validation": list(range(n - 24, n - 12)),
            "test": list(range(n - 12, n))}


def paired_configuration(base: dict, config: dict, family: int, strength: int, reach: int) -> dict:
    cells = list(itertools.product(["circular", "elliptical"], ["ballistic", "diffusive"],
                                   ["inside", "edge", "outside"]))
    order = np.random.default_rng(_seed(config["seed"], family // 12, 0)).permutation(12)
    shape, law, region = cells[order[family % 12]]
    result = copy.deepcopy(base)
    result.update(n_events=1, seed=_seed(config["seed"], family, 1), device_seed=config["device_seed"])
    result["event_timing"]["control_fraction"] = 0.0
    result["source_model"]["source_count_choices"] = [1]
    result["shape"]["geometry_choices"] = [shape]
    result["propagation"]["law_choices"] = [law]
    result["epicenter"]["region_probabilities"] = {region: 1.0}
    result["temporal_model"]["qp_ode"]["generation_scale_per_us"] = config["generation_per_us"][strength]
    scale = config["range_scales"][reach]
    for key in ("initial_lambda_mm", "maximum_lambda_mm", "maximum_distance_mm"):
        value = result["range"][key]
        result["range"][key] = [v * scale for v in value] if isinstance(value, list) else value * scale
    return result


def prefix_syndromes(data: dict, window_ms: float) -> dict:
    """Extract an open observation prefix; never synthesize final-boundary checks."""
    starts = np.asarray(data["round_start_time_ms"])
    duration = float(2 * (data["round_time_ms"][0] - starts[0]))
    count = int(round(window_ms / duration))
    if count < 3 or count > len(starts) or not np.isclose(count * duration, window_ms):
        raise ValueError("window must span at least 3 complete rounds within the source")
    result = {key: np.asarray(data[key]) for key in GEOMETRY_KEYS}
    # Interior detector coordinate t is available at the end of round t.
    selected = data["detector_coords"][:, 2] < count
    result["detector_coords"] = data["detector_coords"][selected]
    result["detector_events"] = data["detector_events"][:, :, selected]
    for key in ("round_start_time_ms", "round_time_ms", "gate_slice_duration_ms"):
        result[key] = data[key][:count]
    return result


def _generate_variant(arguments: tuple) -> dict:
    base, stim, config, family, strength, reach, output = arguments
    root = Path(output)
    name = f"f{family:04d}_s{strength}_r{reach}"
    config_event = paired_configuration(base, config, family, strength, reach)
    circuit = copy.deepcopy(stim)
    circuit["seed"] = _seed(config["syndrome_seed"], family, 2) % (2**63 - 1)
    start, cpu = time.perf_counter(), time.process_time()
    result = run_stim_pipeline(config_event, circuit)
    elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
    arrays = result.syndrome_arrays
    path = root / f"{name}.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, detector_events=arrays["detector_events"],
                            **{key: arrays[key] for key in GEOMETRY_KEYS})
    temporary.replace(path)
    row = result.simulation.parameters.iloc[0]
    label = {key: row[key] for key in (
        "epicenter_row", "epicenter_col", "axis_ratio", "angle_degrees",
        "geometry", "propagation_law", "epicenter_region", "event_onset_ms",
        "front_width_ms", "qp_minimum_t1_us",
    )}
    record = {
        "family": family, "strength": strength, "reach": reach,
        "generation_per_us": config["generation_per_us"][strength],
        "range_scale": config["range_scales"][reach], **label,
        "fraction_round_qubits_below_30us": float(np.mean(arrays["round_t1_us"] < 30)),
        "hardware_t1_map": row["qp_baseline_t1_by_qubit_us"],
        "wall_seconds": elapsed, "cpu_seconds": cpu_elapsed,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "file": path.name, "sha256": _hash_file(path), "bytes": path.stat().st_size,
    }
    _write_json(root / f"{name}.json", record)
    return record


def generate_study(base: dict, stim: dict, config: dict, output: str | Path,
                   *, workers: int = 2, resume: bool = False) -> dict:
    """Generate paired variants in worker processes, committing each separately."""
    validate_study(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    base, stim = copy.deepcopy(base), copy.deepcopy(stim)
    if base.get("event_timing", {}).get("control_fraction", 0) != 0:
        raise ValueError("study requires control_fraction=0; it evaluates conditional localization")
    if stim.get("radiation_channel", {}).get("targets", "all") != "all":
        raise ValueError("paired study uses targets=all")
    maximum = max(config["windows_ms"])
    duration = stim["round_duration_ms"]
    stim["rounds"] = int(round(maximum / duration))
    stim["shots_per_event"] = config["shots"]
    stim["decoder"]["enabled"] = False
    base["time"]["end_ms"] = stim["start_time_ms"] + maximum + 2 * base["time"]["dt_ms"]
    from . import api, simulator, stim_qec, localization
    manifest = {"study": config, "simulator": base, "stim": stim,
                "split_families": family_splits(config),
                "hashes": {Path(p).name: _hash_file(Path(p)) for p in
                           (__file__, api.__file__, simulator.__file__, stim_qec.__file__, localization.__file__)}}
    root = Path(output)
    manifest_path = root / "study_manifest.json"
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError("study output is not empty; choose another directory or resume")
    root.mkdir(parents=True, exist_ok=True)
    with _generation_lock(root):
        if resume:
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("study configuration or code changed; cannot resume")
        else:
            _write_json(manifest_path, manifest)
        pending, records = [], []
        for family, strength, reach in itertools.product(range(config["families"]),
                range(len(config["generation_per_us"])), range(len(config["range_scales"]))):
            path = root / f"f{family:04d}_s{strength}_r{reach}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if _hash_file(root / record["file"]) != record["sha256"]:
                    raise ValueError(f"corrupt study variant: {path}")
                records.append(record)
            else:
                pending.append((base, stim, config, family, strength, reach, str(root)))
        count = len(records) + len(pending)
        begin = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_generate_variant, args) for args in pending]
            for future in as_completed(futures):
                records.append(future.result())
                if len(records) % 12 == 0 or len(records) == count:
                    print(f"generated {len(records)}/{count} variants ({time.perf_counter()-begin:.1f}s this run)", flush=True)
        records.sort(key=lambda row: (row["family"], row["strength"], row["reach"]))
        frame = pd.DataFrame(records)
        paired = ["epicenter_row", "epicenter_col", "axis_ratio", "angle_degrees",
                  "geometry", "propagation_law", "epicenter_region", "event_onset_ms", "front_width_ms"]
        if (frame.groupby("family")[paired].nunique() != 1).any().any():
            raise ValueError("paired event conditions changed unexpectedly")
        if frame.hardware_t1_map.nunique() != 1:
            raise ValueError("hardware calibration changed between variants")
        frame.to_csv(root / "variants.csv", index=False)
        _write_json(root / "generation_summary.json", {
            "variants": len(frame), "families": config["families"],
            "wall_seconds_this_run": time.perf_counter() - begin, "workers": workers,
            "mean_cpu_seconds_per_variant": float(frame.cpu_seconds.mean()),
            "mean_bytes_per_variant": float(frame.bytes.mean()),
            "max_worker_peak_rss_mib": float(frame.worker_peak_rss_mib.max()),
        })
    return manifest


def clustered_mean_interval(frame: pd.DataFrame, column: str = "error_mm") -> list[float]:
    """Descriptive bootstrap interval; paired variants/shots share one family."""
    values = frame.groupby("family")[column].mean().to_numpy()
    rng = np.random.default_rng(2718)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate_study(output: str | Path) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor
    root = Path(output)
    manifest = json.loads((root / "study_manifest.json").read_text())
    config = manifest["study"]
    variants = pd.read_csv(root / "variants.csv")
    expected = config["families"] * len(config["generation_per_us"]) * len(config["range_scales"])
    if len(variants) != expected or variants.duplicated(["family", "strength", "reach"]).any():
        raise ValueError("study variants are incomplete or duplicated")
    frames, selection = [], []
    for window, width in itertools.product(config["windows_ms"], config["bin_widths_us"]):
        features, labels = [], []
        for row in variants.itertuples(index=False):
            path = root / row.file
            if _hash_file(path) != row.sha256:
                raise ValueError(f"corrupt variant {path}")
            data = prefix_syndromes(load_syndromes(path), window)
            bins = int(round(window * 1000 / width))
            spec = feature_specification(data, bins)
            x, rates = extract_features(data, spec)
            centroid = syndrome_centroid(rates, spec)
            features.append(x)
            for shot in range(len(x)):
                labels.append({"family": row.family, "strength": row.strength, "reach": row.reach,
                               "shot": shot, "x_mm": row.epicenter_row, "y_mm": row.epicenter_col,
                               "centroid_x": centroid[shot, 0], "centroid_y": centroid[shot, 1]})
        x = np.concatenate(features)
        labels = pd.DataFrame(labels)
        y = labels[["x_mm", "y_mm"]].to_numpy()
        heldout = (labels.strength == config["held_out_condition"][0]) & (labels.reach == config["held_out_condition"][1])
        for protocol in ("all_conditions", "held_out_combination"):
            train = labels.family.isin(manifest["split_families"]["train"])
            val = labels.family.isin(manifest["split_families"]["validation"])
            test = labels.family.isin(manifest["split_families"]["test"])
            if protocol == "held_out_combination":
                train &= ~heldout
                val &= ~heldout
            best, best_error = None, np.inf
            for leaf in (1, 4):
                model = ExtraTreesRegressor(n_estimators=config["trees"], min_samples_leaf=leaf,
                                            max_features=0.8, random_state=config["seed"], n_jobs=1)
                model.fit(x[train], y[train])
                error = float(np.linalg.norm(model.predict(x[val]) - y[val], axis=1).mean())
                if error < best_error:
                    best, best_error = model, error
            selection.append({"window_ms": window, "bin_width_us": width, "protocol": protocol,
                              "validation_mean_error_mm": best_error, "leaf": best.min_samples_leaf})
            frame = labels.loc[test].copy()
            pred = best.predict(x[test])
            frame["predicted_x_mm"], frame["predicted_y_mm"] = pred[:, 0], pred[:, 1]
            frame["error_mm"] = np.linalg.norm(pred - y[test], axis=1)
            frame["center_error_mm"] = np.linalg.norm(y[test] - y[train].mean(axis=0), axis=1)
            frame["centroid_error_mm"] = np.linalg.norm(frame[["centroid_x", "centroid_y"]].to_numpy() - y[test], axis=1)
            frame["unseen_combination"] = heldout[test].to_numpy() if protocol == "held_out_combination" else False
            frame["window_ms"], frame["bin_width_us"], frame["protocol"] = window, width, protocol
            frames.append(frame)
        print(f"evaluated window={window} ms, bin={width} us", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(root / "study_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(root / "validation_selection.csv", index=False)
    summary = summarize_predictions(predictions, len(manifest["split_families"]["test"]))
    _write_json(root / "study_summary.json", summary)
    return summary


def summarize_predictions(predictions: pd.DataFrame, test_families: int) -> dict:
    summary = {"scope": "paired synthetic pilot; conditional single-event localization, one device",
               "independent_test_families": test_families,
               "evaluation_source_sha256": _hash_file(Path(__file__)),
               "uncertainty": "95% family-bootstrap descriptive intervals; not hardware validation",
               "groups": []}
    columns = ["window_ms", "bin_width_us", "protocol", "strength", "reach"]
    for keys, group in predictions.groupby(columns):
        result = {name: value.item() if isinstance(value, np.generic) else value
                  for name, value in zip(columns, keys)}
        result.update(location_metrics(group[["x_mm", "y_mm"]].to_numpy(),
                                       group[["predicted_x_mm", "predicted_y_mm"]].to_numpy()))
        result.update(mean_error_interval_mm=clustered_mean_interval(group),
                      independent_families=int(group.family.nunique()),
                      center_mean_error_mm=float(group.center_error_mm.mean()),
                      centroid_mean_error_mm=float(group.centroid_error_mm.mean()))
        summary["groups"].append(result)
    return summary


def main():
    import argparse
    from .configuration import config_path, load_json
    from .simulator import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--stim-config", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not args.evaluate_only:
        generate_study(load_config(config_path("simulator_base"), args.profile), load_json(args.stim_config),
                       load_json(args.study_config), args.output, workers=args.workers, resume=args.resume)
    evaluate_study(args.output)


if __name__ == "__main__":
    main()
