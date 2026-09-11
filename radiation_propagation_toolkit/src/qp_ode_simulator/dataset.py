"""Reproducible, resumable shards of labelled circuit syndrome events."""

from __future__ import annotations

import copy
import itertools
import importlib.metadata
import json
import os
import platform
from contextlib import contextmanager, ExitStack
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .api import run_stim_pipeline
from .localization import GEOMETRY_KEYS, _hash_file, load_syndromes
from .stim_qec import build_stim_layout, uses_fixed_hardware, uses_peak_matched_qp


def _identity(value: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _generation_lock(root: Path):
    path = root / ".generation.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            pid = int(path.read_text())
            if pid <= 0:
                raise ValueError("invalid lock pid")
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink()  # Only a confirmed dead process's lock is reclaimed.
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        else:
            raise ValueError(f"dataset is locked by process {pid}: {root}")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _seed(root: int, event: int, stream: int) -> int:
    return int(np.random.SeedSequence([root, event, stream]).generate_state(1, dtype=np.uint64)[0])


def validate_sampling(plan: dict) -> list[tuple]:
    allowed = {"geometry", "propagation_law", "epicenter_region", "generation_bands_per_us"}
    if set(plan) != allowed:
        raise ValueError(f"sampling plan requires exactly {sorted(allowed)}")
    for key, choices in (
        ("geometry", {"circular", "elliptical"}),
        ("propagation_law", {"ballistic", "diffusive"}),
        ("epicenter_region", {"inside", "edge", "outside"}),
    ):
        values = plan[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or not set(values) <= choices:
            raise ValueError(f"invalid sampling choices for {key}")
    bands = np.asarray(plan["generation_bands_per_us"], dtype=float)
    if bands.ndim != 2 or bands.shape[1] != 2 or len(bands) == 0:
        raise ValueError("generation_bands_per_us must contain [low, high] pairs")
    if not np.isfinite(bands).all() or np.any(bands <= 0) or np.any(bands[:, 1] < bands[:, 0]):
        raise ValueError("generation bands must be finite, positive and ordered")
    return list(itertools.product(plan["geometry"], plan["propagation_law"], plan["epicenter_region"], range(len(bands))))


def event_configuration(base: dict, plan: dict, event_id: int, seed: int, device_seed: int) -> tuple[dict, dict]:
    """Assign balanced cells and stable per-event seeds, independent of shard size."""
    cells = validate_sampling(plan)
    cycle, offset = divmod(event_id, len(cells))
    order = np.random.default_rng(_seed(seed, cycle, 0)).permutation(len(cells))
    geometry, law, region, band = cells[order[offset]]
    rng = np.random.default_rng(_seed(seed, event_id, 1))
    lower, upper = plan["generation_bands_per_us"][band]
    generation = float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
    config = copy.deepcopy(base)
    config["n_events"] = 1
    config["seed"] = _seed(seed, event_id, 2)
    config["device_seed"] = device_seed
    config["shape"]["geometry_choices"] = [geometry]
    config["propagation"]["law_choices"] = [law]
    config["epicenter"]["region_probabilities"] = {region: 1.0}
    config["temporal_model"]["qp_ode"]["generation_scale_per_us"] = generation
    fraction = float(config.get("event_timing", {}).get("control_fraction", 0.0))
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("control_fraction must be in [0, 1]")
    # One-event calls must not round a 10% control fraction to zero every time.
    config.setdefault("event_timing", {})["control_fraction"] = float(rng.random() < fraction)
    return config, {"strength_band": int(band), "generation_seed": config["seed"]}


def dataset_manifest(root: str | Path, *, require_complete: bool = True) -> dict:
    path = Path(root) / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "qp_syndrome_shards_v1":
        raise ValueError("unsupported dataset manifest")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError("dataset is incomplete; resume generation before training")
    if manifest.get("completed_events", 0) != sum(s["events"] for s in manifest["shards"]):
        raise ValueError("manifest event counts disagree")
    if require_complete and manifest["completed_events"] != manifest["generation"]["n_events"]:
        raise ValueError("complete dataset has an incorrect event count")
    return manifest


def _shard_path(root: Path, name: str) -> Path:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("shard names must be local filenames")
    return root / name


def iter_shards(root: str | Path, *, verify: bool = True, require_complete: bool = True):
    """Read one raw shard at a time and validate IDs, labels and circuit identity."""
    root = Path(root)
    manifest = dataset_manifest(root, require_complete=require_complete)
    expected_start = 0
    for shard in manifest["shards"]:
        paths = {key: _shard_path(root, shard[key]) for key in ("npz", "csv")}
        if verify:
            for key, path in paths.items():
                if _hash_file(path) != shard[f"{key}_sha256"]:
                    raise ValueError(f"shard checksum mismatch: {path}")
        data = load_syndromes(paths["npz"])
        labels = pd.read_csv(paths["csv"])
        ids = np.arange(expected_start, expected_start + shard["events"])
        if (not np.array_equal(data.get("event_ids"), ids)
                or not np.array_equal(labels["event"], ids)
                or not np.array_equal(data.get("event_uids"), labels["event_uid"].to_numpy())
                or data["detector_events"].shape[:2] != (len(ids), manifest["stim_config"].get("shots_per_event", 1))):
            raise ValueError("shard event IDs, labels or sample counts disagree")
        expected_uids = np.asarray([f"{manifest['source_id']}:{i}" for i in ids])
        if not np.array_equal(data["event_uids"], expected_uids):
            raise ValueError("event UID does not match source identity")
        if str(data.get("circuit_id", "")) != manifest["circuit_id"]:
            raise ValueError("shard belongs to a different circuit/code")
        if not labels["is_control"].isin([True, False, 0, 1]).all() or not np.array_equal(
            labels["is_control"].to_numpy(dtype=bool), data["event_is_control"]
        ):
            raise ValueError("control flags disagree")
        expected_start += len(ids)
        yield data, labels


def _generate_event(arguments):
    base, plan, stim_config, event_id, seed, device_seed, syndrome_seed, source_id, circuit_id = arguments
    config, sampled = event_configuration(base, plan, event_id, seed, device_seed)
    circuit_config = copy.deepcopy(stim_config)
    circuit_config["seed"] = _seed(syndrome_seed, event_id, 3) % (2**63 - 1)
    result = run_stim_pipeline(config, circuit_config)
    arrays = result.syndrome_arrays
    row = result.simulation.parameters.iloc[0].to_dict()
    row.update(sampled)
    row.update(event=event_id, event_uid=f"{source_id}:{event_id}", device_seed=device_seed,
               syndrome_seed=circuit_config["seed"], circuit_id=circuit_id)
    row["fraction_round_qubits_below_30us"] = float(np.mean(arrays["round_t1_us"][0] < 30))
    return arrays["detector_events"][0].copy(), row, {key: arrays[key] for key in GEOMETRY_KEYS}


def generate_dataset(
    simulator_config: dict, stim_config: dict, plan: dict, output: str | Path, *,
    n_events: int, batch_size: int = 16, seed: int = 42, device_seed: int = 123,
    syndrome_seed: int = 456, resume: bool = False, progress=print, workers: int = 1,
) -> dict:
    """Generate bounded-memory shards; committed shards survive interruption."""
    validate_sampling(plan)
    for name, value in (("n_events", n_events), ("batch_size", batch_size), ("workers", workers)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (seed, device_seed, syndrome_seed)):
        raise ValueError("seeds must be non-negative integers")
    if not uses_fixed_hardware(simulator_config) or uses_peak_matched_qp(simulator_config):
        raise ValueError("dataset generation requires fixed hardware and no peak matching")
    if simulator_config.get("source_model", {}).get("source_count_choices", [1]) != [1]:
        raise ValueError("localization datasets currently require a single source")
    layout = build_stim_layout(stim_config)
    import hashlib
    circuit_id = hashlib.sha256(str(layout.circuit).encode()).hexdigest()
    base = copy.deepcopy(simulator_config)
    for key in ("n_events", "seed", "device_seed"):
        base.pop(key, None)
    source_id = _identity({"simulator": base, "sampling": plan, "seed": seed, "device_seed": device_seed,
                           "physical_coords_mm": layout.physical_coords_mm.tolist()})
    from . import api, layouts, simulator, stim_qec, syndrome
    generation = {
        "n_events": n_events, "seed": seed, "device_seed": device_seed,
        "syndrome_seed": syndrome_seed,
        "source_hashes": {Path(p).name: _hash_file(Path(p)) for p in (
            __file__, api.__file__, layouts.__file__, simulator.__file__, stim_qec.__file__, syndrome.__file__,
        )},
        "runtime_versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "stim")
        }},
    }
    manifest = {
        "format": "qp_syndrome_shards_v1", "status": "generating",
        "source_id": source_id, "circuit_id": circuit_id,
        "code": {"task": stim_config.get("task", "surface_code:rotated_memory_z"), "distance": stim_config["distance"]},
        "simulator_config": base, "stim_config": copy.deepcopy(stim_config),
        "sampling": copy.deepcopy(plan), "generation": generation,
        "completed_events": 0, "shards": [],
        "coordinate_convention": "x_mm=epicenter_row, y_mm=epicenter_col",
    }
    root = Path(output)
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError("output is not empty; use a new directory or --resume")
    if resume and not (root / "dataset_manifest.json").is_file():
        raise ValueError("no manifest to resume")
    root.mkdir(parents=True, exist_ok=True)
    with _generation_lock(root), ExitStack() as stack:
        if resume:
            previous = dataset_manifest(root, require_complete=False)
            for key in ("source_id", "circuit_id", "generation", "stim_config"):
                if previous[key] != manifest[key]:
                    raise ValueError(f"cannot resume: {key} changed")
            for _ in iter_shards(root, require_complete=False):
                pass
            manifest = previous
        else:
            _write_json(root / "dataset_manifest.json", manifest)
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers)) if workers > 1 else None
        for start in range(manifest["completed_events"], n_events, batch_size):
            stop = min(start + batch_size, n_events)
            bits, parameters, controls = [], [], []
            arguments = [(base, plan, stim_config, event_id, seed, device_seed, syndrome_seed, source_id, circuit_id)
                         for event_id in range(start, stop)]
            results = pool.map(_generate_event, arguments) if pool else map(_generate_event, arguments)
            for event_bits, row, geometry in results:
                bits.append(event_bits)
                parameters.append(row)
                controls.append(int(row["is_control"]))
            prefix = f"events_{start:08d}_{stop:08d}"
            npz_path, csv_path = root / f"{prefix}.npz", root / f"{prefix}.csv"
            temporary = root / f"{prefix}.npz.tmp"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, detector_events=np.stack(bits), **geometry,
                                    event_ids=np.arange(start, stop), event_uids=np.asarray([r["event_uid"] for r in parameters]),
                                    event_is_control=np.asarray(controls, dtype=np.uint8), circuit_id=np.asarray(circuit_id))
            temporary.replace(npz_path)
            csv_temporary = root / f"{prefix}.csv.tmp"
            pd.DataFrame(parameters).to_csv(csv_temporary, index=False)
            csv_temporary.replace(csv_path)
            manifest["shards"].append({"npz": npz_path.name, "csv": csv_path.name, "events": stop - start,
                                       "npz_sha256": _hash_file(npz_path), "csv_sha256": _hash_file(csv_path)})
            manifest["completed_events"] = stop
            _write_json(root / "dataset_manifest.json", manifest)
            if progress is not None:
                progress(f"saved {stop}/{n_events} events ({len(manifest['shards'])} shards)")
        manifest["status"] = "complete"
        _write_json(root / "dataset_manifest.json", manifest)
    return manifest


def audit_dataset(dataset: str | Path, output: str | Path, windows_ms: list[float]) -> dict:
    """Compare prefixes of the same trajectories; diagnostics are not model inputs."""
    root = Path(dataset)
    manifest = dataset_manifest(root)
    circuit = manifest["stim_config"]
    duration = circuit["round_duration_ms"] * circuit["rounds"]
    windows = np.asarray(windows_ms, dtype=float)
    if len(windows) == 0 or not np.isfinite(windows).all() or np.any(windows <= 0) or np.any(windows > duration + 1e-9):
        raise ValueError("windows must be positive and within the generated observation duration")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("audit output must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    for data, labels in iter_shards(root):
        coords = data["detector_coords"]
        # Interior detector t compares rounds t-1 and t, so use end of round t.
        relative_ms = (coords[:, 2] + 1) * circuit["round_duration_ms"]
        absolute_ms = circuit["start_time_ms"] + relative_ms
        interior = (coords[:, 2] > 0) & (coords[:, 2] < circuit["rounds"])
        sites = np.unique(coords[interior, :2], axis=0)
        rates = data["detector_events"].mean(axis=1)
        for window in windows:
            selected = interior & (relative_ms <= window + 1e-9)
            if not selected.any():
                raise ValueError("window contains no interior detector measurements")
            for i, row in labels.iterrows():
                before = selected & (absolute_ms < row.event_onset_ms)
                after = selected & (absolute_ms >= row.event_onset_ms)
                spatial = [rates[i, selected & np.all(coords[:, :2] == site, axis=1)].mean()
                           for site in sites if np.any(selected & np.all(coords[:, :2] == site, axis=1))]
                early = float(rates[i, before].mean()) if before.any() else np.nan
                late = float(rates[i, after].mean()) if after.any() else np.nan
                frames.append({"event": int(row.event), "event_uid": row.event_uid, "window_ms": float(window),
                               "strength_band": int(row.strength_band), "geometry": row.geometry,
                               "propagation_law": row.propagation_law, "epicenter_region": row.epicenter_region,
                               "is_control": bool(row.is_control), "full_field_minimum_t1_us": row.qp_minimum_t1_us,
                               "detector_rate": float(rates[i, selected].mean()), "pre_event_rate": early,
                               "post_event_rate": late, "excess_rate": late - early,
                               "spatial_rate_std": float(np.std(spatial))})
    frame = pd.DataFrame(frames)
    frame.to_csv(destination / "event_window_diagnostics.csv", index=False)
    summary = {"events": manifest["completed_events"], "windows_ms": windows.tolist(), "code": manifest["code"],
               "interpretation": "descriptive prefix/strength diagnostics; not localization accuracy or a causal estimate",
               "groups": []}
    for (window, band), group in frame.groupby(["window_ms", "strength_band"]):
        summary["groups"].append({"window_ms": float(window), "strength_band": int(band), "events": len(group),
                                  "mean_detector_rate": float(group.detector_rate.mean()),
                                  "mean_spatial_rate_std": float(group.spatial_rate_std.mean())})
    _write_json(destination / "audit_summary.json", summary)
    return summary
