#!/usr/bin/env python3
"""Generate stabilizer syndromes and detection events from simulator outputs.

Two data-fault backends are supported.  The legacy response-excess backend maps the
simulated RReCS response phenomenologically.  The physics-informed T1 backend Pauli
twirls the relaxation/dephasing channel obtained from the QP model.  Both backends
then evolve a Pauli frame, measure configurable CSS checks, and add optional
stabilizer-measurement faults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAULI_I = 0
PAULI_X = 1
PAULI_Y = 2
PAULI_Z = 3

SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}
KNOWN_CONFIG_KEYS = {
    "schema_version",
    "backend",
    "graph_maximum_edge_mm",
    "require_css_commutation",
    "x_checks",
    "z_checks",
    "data_error_model",
    "qec_round_duration_ms",
    "fault_probability_backend",
    "fault_probability_model",
    "t1_pauli_model",
    "pauli_probabilities",
    "measurement_error_probability",
    "measurement_error_rate_per_ms",
    "measurement_error_radiation_scale",
    "measurement_error_radiation_rate_per_ms",
    "include_final_boundary",
    "allow_static_response_baseline",
    "output",
    "seed",
}


def graph_repetition_checks(
    coords: np.ndarray, maximum_edge_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pair-parity Z checks for all spatial graph edges within a cutoff."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("data-qubit coordinates must have finite shape (n_data, 2)")
    if not np.isfinite(maximum_edge_mm) or maximum_edge_mm <= 0:
        raise ValueError("graph maximum_edge_mm must be strictly positive")
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    edges = np.argwhere(np.triu((distance > 0) & (distance <= maximum_edge_mm), 1))
    if len(edges) == 0:
        raise ValueError("graph cutoff produced no parity checks")
    checks = np.zeros((len(edges), len(coords)), dtype=np.uint8)
    checks[np.arange(len(edges))[:, None], edges] = 1
    check_coords = coords[edges].mean(axis=1).astype(np.float32)
    return checks, edges.astype(np.int32), check_coords


def _supports_to_matrix(supports: list[list[int]], n_data: int, name: str) -> np.ndarray:
    matrix = np.zeros((len(supports), n_data), dtype=np.uint8)
    for check, support in enumerate(supports):
        if not support:
            raise ValueError(f"{name} check {check} has empty support")
        indices = np.asarray(support, dtype=int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} check {check} contains duplicate data qubits")
        if np.any(indices < 0) or np.any(indices >= n_data):
            raise ValueError(f"{name} check {check} references an invalid data qubit")
        matrix[check, indices] = 1
    return matrix


def _check_coordinates(matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if len(matrix) == 0:
        return np.empty((0, 2), dtype=np.float32)
    weights = matrix.sum(axis=1, keepdims=True)
    return ((matrix @ coords) / weights).astype(np.float32)


def checks_from_config(
    config: dict, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct Hx/Hz and check coordinates from a syndrome configuration."""
    n_data = len(coords)
    backend = config.get("backend", "graph_repetition_proxy")
    if backend == "graph_repetition_proxy":
        hz, edges, z_coords = graph_repetition_checks(
            coords, float(config.get("graph_maximum_edge_mm", 1.01))
        )
        hx = np.zeros((0, n_data), dtype=np.uint8)
        x_coords = np.empty((0, 2), dtype=np.float32)
    elif backend == "custom_css":
        hx = _supports_to_matrix(config.get("x_checks", []), n_data, "X")
        hz = _supports_to_matrix(config.get("z_checks", []), n_data, "Z")
        if len(hx) + len(hz) == 0:
            raise ValueError("custom_css requires at least one X or Z check")
        edges = np.empty((0, 2), dtype=np.int32)
        x_coords = _check_coordinates(hx, coords)
        z_coords = _check_coordinates(hz, coords)
    else:
        raise ValueError(f"unsupported syndrome backend: {backend!r}")

    if config.get("require_css_commutation", True) is not True:
        raise ValueError("require_css_commutation cannot be disabled for a CSS backend")
    if len(hx) and len(hz) and np.any(
        (hx.astype(np.int16) @ hz.astype(np.int16).T) % 2
    ):
        raise ValueError("X and Z check matrices do not commute over GF(2)")
    return hx, hz, x_coords, z_coords, edges


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _validate_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _exclusive_settings(config: dict, first: str, second: str) -> None:
    if first in config and second in config:
        raise ValueError(f"configure only one of {first} and {second}")


def _probability_from_rate(rate_per_ms: float, duration_ms: float) -> float:
    return float(-np.expm1(-float(rate_per_ms) * float(duration_ms)))


def validate_syndrome_config(config: dict, n_data: int) -> None:
    if not isinstance(config, dict):
        raise ValueError("syndrome configuration must be a JSON object")
    unknown = set(config) - KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown syndrome configuration keys: {sorted(unknown)}")
    schema_version = config.get("schema_version", 1)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported syndrome schema_version {schema_version!r}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if not isinstance(n_data, int) or n_data <= 0:
        raise ValueError("n_data must be a positive integer")
    error_model = config.get("data_error_model", "pauli_frame")
    if error_model not in {"pauli_frame", "round_independent"}:
        raise ValueError("data_error_model must be 'pauli_frame' or 'round_independent'")

    duration = config.get("qec_round_duration_ms")
    if duration is not None:
        duration = _validate_nonnegative("qec_round_duration_ms", duration)
        if duration == 0:
            raise ValueError("qec_round_duration_ms must be strictly positive")
    if not isinstance(config.get("include_final_boundary", True), bool):
        raise ValueError("include_final_boundary must be boolean")
    if not isinstance(
        config.get("allow_static_response_baseline", schema_version == 1), bool
    ):
        raise ValueError("allow_static_response_baseline must be boolean")

    fault_backend = config.get("fault_probability_backend", "response_excess")
    if fault_backend not in {"response_excess", "t1_pauli"}:
        raise ValueError(
            "fault_probability_backend must be 'response_excess' or 't1_pauli'"
        )
    if fault_backend == "t1_pauli":
        if schema_version < 3:
            raise ValueError("t1_pauli requires syndrome schema_version 3")
        if duration is None:
            raise ValueError("t1_pauli requires qec_round_duration_ms")
        t1_model = config.get("t1_pauli_model", {})
        if not isinstance(t1_model, dict):
            raise ValueError("t1_pauli_model must be an object")
        allowed_t1 = {"pure_dephasing_time_us"}
        unknown_t1 = set(t1_model) - allowed_t1
        if unknown_t1:
            raise ValueError(f"unknown t1_pauli_model keys: {sorted(unknown_t1)}")
        if "pure_dephasing_time_us" in t1_model:
            _validate_positive(
                "t1_pauli_model.pure_dephasing_time_us",
                t1_model["pure_dephasing_time_us"],
            )
        ignored = {
            key for key in ("fault_probability_model", "pauli_probabilities") if key in config
        }
        if ignored:
            raise ValueError(
                "t1_pauli derives X/Y/Z probabilities directly; remove ignored settings "
                f"{sorted(ignored)}"
            )
    elif "t1_pauli_model" in config:
        raise ValueError(
            "t1_pauli_model is only valid with fault_probability_backend='t1_pauli'"
        )

    fault = config.get("fault_probability_model", {})
    if not isinstance(fault, dict):
        raise ValueError("fault_probability_model must be an object")
    allowed_fault = {
        "baseline_probability",
        "baseline_rate_per_ms",
        "radiation_excess_scale",
        "radiation_excess_rate_per_ms",
        "maximum_probability",
    }
    unknown_fault = set(fault) - allowed_fault
    if unknown_fault:
        raise ValueError(f"unknown fault_probability_model keys: {sorted(unknown_fault)}")
    _exclusive_settings(fault, "baseline_probability", "baseline_rate_per_ms")
    _exclusive_settings(fault, "radiation_excess_scale", "radiation_excess_rate_per_ms")
    if "baseline_rate_per_ms" in fault or "radiation_excess_rate_per_ms" in fault:
        if duration is None:
            raise ValueError("rate-based fault settings require qec_round_duration_ms")
    if "baseline_rate_per_ms" in fault:
        baseline_rate = _validate_nonnegative(
            "fault_probability_model.baseline_rate_per_ms",
            fault["baseline_rate_per_ms"],
        )
        baseline = _probability_from_rate(baseline_rate, duration)
    else:
        baseline = _validate_probability(
            "fault_probability_model.baseline_probability",
            fault.get("baseline_probability", 0.001),
        )
    maximum = _validate_probability(
        "fault_probability_model.maximum_probability",
        fault.get("maximum_probability", 0.25),
    )
    if maximum < baseline:
        raise ValueError("fault maximum_probability cannot be below baseline_probability")
    if "radiation_excess_rate_per_ms" in fault:
        _validate_nonnegative(
            "fault_probability_model.radiation_excess_rate_per_ms",
            fault["radiation_excess_rate_per_ms"],
        )
    else:
        _validate_nonnegative(
            "fault_probability_model.radiation_excess_scale",
            fault.get("radiation_excess_scale", 0.05),
        )

    pauli = config.get("pauli_probabilities", {"x": 1.0, "y": 0.0, "z": 0.0})
    if not isinstance(pauli, dict) or set(pauli) - {"x", "y", "z"}:
        raise ValueError("pauli_probabilities may contain only x, y, and z")
    pauli_values = np.asarray([pauli.get(key, 0.0) for key in ("x", "y", "z")], dtype=float)
    if (
        not np.isfinite(pauli_values).all()
        or np.any(pauli_values < 0)
        or not np.isclose(pauli_values.sum(), 1.0)
    ):
        raise ValueError("conditional X/Y/Z Pauli probabilities must be non-negative and sum to 1")

    _exclusive_settings(config, "measurement_error_probability", "measurement_error_rate_per_ms")
    _exclusive_settings(
        config,
        "measurement_error_radiation_scale",
        "measurement_error_radiation_rate_per_ms",
    )
    if (
        "measurement_error_rate_per_ms" in config
        or "measurement_error_radiation_rate_per_ms" in config
    ) and duration is None:
        raise ValueError("rate-based measurement settings require qec_round_duration_ms")
    if "measurement_error_rate_per_ms" in config:
        _validate_nonnegative(
            "measurement_error_rate_per_ms", config["measurement_error_rate_per_ms"]
        )
    else:
        _validate_probability(
            "measurement_error_probability",
            config.get("measurement_error_probability", 0.001),
        )
    if "measurement_error_radiation_rate_per_ms" in config:
        _validate_nonnegative(
            "measurement_error_radiation_rate_per_ms",
            config["measurement_error_radiation_rate_per_ms"],
        )
    else:
        _validate_nonnegative(
            "measurement_error_radiation_scale",
            config.get("measurement_error_radiation_scale", 0.0),
        )

    output = config.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    unknown_output = set(output) - {"include_intermediate_arrays"}
    if unknown_output:
        raise ValueError(f"unknown output keys: {sorted(unknown_output)}")
    legacy_output_default = schema_version == 1
    if not isinstance(
        output.get("include_intermediate_arrays", legacy_output_default), bool
    ):
        raise ValueError("output.include_intermediate_arrays must be boolean")


def _interpolate_time_axis(
    values: np.ndarray, source_time_ms: np.ndarray, target_time_ms: np.ndarray
) -> np.ndarray:
    """Linearly interpolate arrays shaped (event, time, qubit)."""
    if np.array_equal(source_time_ms, target_time_ms):
        return values
    right = np.searchsorted(source_time_ms, target_time_ms, side="right")
    left = np.clip(right - 1, 0, len(source_time_ms) - 1)
    right = np.clip(right, 0, len(source_time_ms) - 1)
    denominator = source_time_ms[right] - source_time_ms[left]
    alpha = np.divide(
        target_time_ms - source_time_ms[left],
        denominator,
        out=np.zeros_like(target_time_ms, dtype=float),
        where=denominator > 0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)[None, :, None]
    interpolated = values[:, left, :] * (1 - alpha) + values[:, right, :] * alpha
    return interpolated.astype(np.result_type(values.dtype, np.float32), copy=False)


def resample_to_qec_rounds(
    probabilities: np.ndarray,
    baselines: np.ndarray,
    time_ms: np.ndarray,
    round_duration_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample latent response and dynamic baseline onto fixed QEC rounds."""
    probabilities = np.asarray(probabilities)
    baselines = np.asarray(baselines)
    time_ms = np.asarray(time_ms, dtype=float)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (event, time, data_qubit)")
    static_shape = (probabilities.shape[0], probabilities.shape[2])
    if baselines.shape not in {static_shape, probabilities.shape}:
        raise ValueError(
            "baselines must have shape (event, data_qubit) or "
            "(event, time, data_qubit)"
        )
    if not np.isfinite(round_duration_ms) or round_duration_ms <= 0:
        raise ValueError("round_duration_ms must be finite and strictly positive")
    if time_ms.ndim != 1 or len(time_ms) != probabilities.shape[1]:
        raise ValueError("time_ms must match the probability time axis")
    if not np.isfinite(time_ms).all() or len(time_ms) == 0:
        raise ValueError("time_ms must be a non-empty finite array")
    if len(time_ms) == 1:
        return probabilities, baselines, time_ms.astype(np.float32)
    differences = np.diff(time_ms)
    if np.any(differences <= 0):
        raise ValueError("time_ms must be strictly increasing")
    span = float(time_ms[-1] - time_ms[0])
    n_rounds = max(1, int(np.floor(span / round_duration_ms + 1e-10)) + 1)
    round_time_ms = time_ms[0] + np.arange(n_rounds) * round_duration_ms
    round_probabilities = _interpolate_time_axis(probabilities, time_ms, round_time_ms)
    round_baselines = (
        _interpolate_time_axis(baselines, time_ms, round_time_ms)
        if baselines.ndim == 3
        else baselines
    )
    return round_probabilities, round_baselines, round_time_ms.astype(np.float32)


def response_to_fault_probability(
    probabilities: np.ndarray,
    baselines: np.ndarray,
    config: dict,
    *,
    round_duration_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map RReCS response excess to a configurable QEC-round fault probability."""
    probabilities = np.asarray(probabilities, dtype=float)
    baselines = np.asarray(baselines, dtype=float)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (event, round, data_qubit)")
    static_shape = (probabilities.shape[0], probabilities.shape[2])
    if baselines.shape not in {static_shape, probabilities.shape}:
        raise ValueError(
            "baselines must have shape (event, data_qubit) or "
            "(event, round, data_qubit)"
        )
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.isfinite(baselines).all() or np.any((baselines < 0) | (baselines > 1)):
        raise ValueError("baselines must be finite and lie in [0, 1]")

    fault = config.get("fault_probability_model", {})
    if "baseline_rate_per_ms" in fault:
        if round_duration_ms is None:
            raise ValueError("baseline_rate_per_ms requires a QEC round duration")
        baseline_probability = _probability_from_rate(
            fault["baseline_rate_per_ms"], round_duration_ms
        )
    else:
        baseline_probability = float(fault.get("baseline_probability", 0.001))
    maximum_probability = float(fault.get("maximum_probability", 0.25))
    baseline_series = baselines[:, None, :] if baselines.ndim == 2 else baselines
    response_excess = np.maximum(probabilities - baseline_series, 0.0)
    if "radiation_excess_rate_per_ms" in fault:
        if round_duration_ms is None:
            raise ValueError("radiation_excess_rate_per_ms requires a QEC round duration")
        radiation_probability = -np.expm1(
            -float(fault["radiation_excess_rate_per_ms"])
            * round_duration_ms
            * response_excess
        )
    else:
        radiation_scale = float(fault.get("radiation_excess_scale", 0.05))
        radiation_probability = np.clip(radiation_scale * response_excess, 0.0, 1.0)
    fault_probability = 1 - (1 - baseline_probability) * (1 - radiation_probability)
    return (
        np.clip(fault_probability, 0.0, maximum_probability).astype(np.float32),
        response_excess.astype(np.float32),
    )


def t1_to_pauli_probabilities(
    t1_us: np.ndarray,
    round_duration_ms: float,
    *,
    t2_us: np.ndarray | None = None,
    pure_dephasing_time_us: float | None = None,
    include_identity: bool = True,
) -> dict[str, np.ndarray]:
    """Pauli twirl a T1/Tphi relaxation channel for one QEC round.

    The total dephasing time obeys 1/T2 = 1/(2*T1) + 1/Tphi.  Omitting
    ``pure_dephasing_time_us`` uses T2=2*T1, so the channel contains no extra
    pure dephasing beyond relaxation.  The returned arrays share the shape of
    ``t1_us`` and contain mutually exclusive I/X/Y/Z probabilities.
    """
    t1 = np.asarray(t1_us, dtype=np.float32)
    if t1.ndim != 3:
        raise ValueError("t1_us must have shape (event, round, data_qubit)")
    if not np.isfinite(t1).all() or np.any(t1 <= 0):
        raise ValueError("t1_us must be finite and strictly positive")
    duration_us = np.float32(
        _validate_positive("qec_round_duration_ms", round_duration_ms) * 1000
    )
    if t2_us is not None and pure_dephasing_time_us is not None:
        raise ValueError("provide t2_us or pure_dephasing_time_us, not both")
    if t2_us is not None:
        t2 = np.asarray(t2_us, dtype=np.float32)
        if t2.shape != t1.shape or not np.isfinite(t2).all() or np.any(t2 <= 0):
            raise ValueError("t2_us must match t1_us and be finite and positive")
        if np.any(t2 > 2.0 * t1 * (1 + 1e-6)):
            raise ValueError("t2_us must satisfy T2 <= 2*T1 for this Pauli channel")
        inverse_t2 = 1.0 / t2
    else:
        inverse_t2 = 0.5 / t1
    if pure_dephasing_time_us is not None:
        inverse_t2 = inverse_t2 + 1.0 / _validate_positive(
            "pure_dephasing_time_us", pure_dephasing_time_us
        )
    t2 = 1.0 / inverse_t2
    lambda_z = np.exp(-duration_us / t1)
    lambda_xy = np.exp(-duration_us / t2)
    p_x = (1.0 - lambda_z) / 4.0
    p_y = p_x.copy()
    p_z = (1.0 - 2.0 * lambda_xy + lambda_z) / 4.0
    # The analytical channel is non-negative when T2 <= 2*T1.  Clipping only
    # removes floating-point roundoff at very long coherence times.
    p_x = np.maximum(p_x, 0.0)
    p_y = np.maximum(p_y, 0.0)
    p_z = np.maximum(p_z, 0.0)
    result = {
        "x": p_x.astype(np.float32, copy=False),
        "y": p_y.astype(np.float32, copy=False),
        "z": p_z.astype(np.float32, copy=False),
        "t2_us": t2.astype(np.float32),
    }
    if include_identity:
        result["i"] = np.maximum(1.0 - p_x - p_y - p_z, 0.0).astype(
            np.float32, copy=False
        )
    return result


def sample_pauli_channel(
    rng: np.random.Generator, pauli_probability: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a spatially and temporally varying Pauli channel."""
    p_x = np.asarray(pauli_probability["x"], dtype=float)
    p_y = np.asarray(pauli_probability["y"], dtype=float)
    p_z = np.asarray(pauli_probability["z"], dtype=float)
    if p_x.shape != p_y.shape or p_x.shape != p_z.shape or p_x.ndim != 3:
        raise ValueError("Pauli probabilities must share shape (event, round, qubit)")
    if not all(np.isfinite(value).all() for value in (p_x, p_y, p_z)):
        raise ValueError("Pauli probabilities must be finite")
    total = p_x + p_y + p_z
    if np.any(p_x < 0) or np.any(p_y < 0) or np.any(p_z < 0) or np.any(total > 1):
        raise ValueError("Pauli probabilities must be non-negative and sum to at most one")
    draws = rng.random(p_x.shape)
    labels = np.zeros(p_x.shape, dtype=np.uint8)
    labels[draws < p_x] = PAULI_X
    labels[(draws >= p_x) & (draws < p_x + p_y)] = PAULI_Y
    labels[(draws >= p_x + p_y) & (draws < total)] = PAULI_Z
    x_faults = (labels == PAULI_X) | (labels == PAULI_Y)
    z_faults = (labels == PAULI_Z) | (labels == PAULI_Y)
    return labels, x_faults, z_faults


def sample_pauli_faults(
    rng: np.random.Generator, fault_probability: np.ndarray, config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample conditional Pauli labels and return X/Z components."""
    pauli = config.get("pauli_probabilities", {"x": 1.0, "y": 0.0, "z": 0.0})
    conditional = np.asarray([pauli.get(key, 0.0) for key in ("x", "y", "z")], dtype=float)
    faults = rng.random(fault_probability.shape) < fault_probability
    labels = np.zeros(fault_probability.shape, dtype=np.uint8)
    draws = rng.random(fault_probability.shape)
    x_boundary = conditional[0]
    y_boundary = conditional[0] + conditional[1]
    labels[faults & (draws < x_boundary)] = PAULI_X
    labels[faults & (draws >= x_boundary) & (draws < y_boundary)] = PAULI_Y
    labels[faults & (draws >= y_boundary)] = PAULI_Z
    x_faults = (labels == PAULI_X) | (labels == PAULI_Y)
    z_faults = (labels == PAULI_Z) | (labels == PAULI_Y)
    return labels, x_faults, z_faults


def _syndrome(frame: np.ndarray, checks: np.ndarray) -> np.ndarray:
    return (
        np.matmul(frame.astype(np.int16), checks.astype(np.int16).T) % 2
    ).astype(np.uint8)


def detection_events(
    syndrome: np.ndarray, *, include_final_boundary: bool = False
) -> np.ndarray:
    """Temporal differences with an initial and optional final zero boundary."""
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    if syndrome.ndim != 3 or syndrome.shape[1] == 0:
        raise ValueError("syndrome must have non-empty shape (event, round, check)")
    detection = np.empty_like(syndrome)
    detection[:, 0] = syndrome[:, 0]
    detection[:, 1:] = syndrome[:, 1:] ^ syndrome[:, :-1]
    if include_final_boundary:
        detection = np.concatenate((detection, syndrome[:, -1:, :]), axis=1)
    return detection


def syndrome_from_faults(
    x_faults: np.ndarray,
    z_faults: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    data_error_model: str = "pauli_frame",
    x_measurement_flips: np.ndarray | None = None,
    z_measurement_flips: np.ndarray | None = None,
    include_final_boundary: bool = False,
) -> dict[str, np.ndarray]:
    """Low-level deterministic CSS syndrome calculation used by tests and tooling."""
    x_faults = np.asarray(x_faults, dtype=bool)
    z_faults = np.asarray(z_faults, dtype=bool)
    if x_faults.shape != z_faults.shape or x_faults.ndim != 3:
        raise ValueError("X and Z faults must share shape (event, round, data_qubit)")
    if hx.shape[1] != x_faults.shape[2] or hz.shape[1] != x_faults.shape[2]:
        raise ValueError("check matrices and fault arrays disagree on data-qubit count")
    if data_error_model == "pauli_frame":
        x_frame = np.logical_xor.accumulate(x_faults, axis=1)
        z_frame = np.logical_xor.accumulate(z_faults, axis=1)
    elif data_error_model == "round_independent":
        x_frame, z_frame = x_faults, z_faults
    else:
        raise ValueError("unsupported data_error_model")

    # Z checks detect X components; X checks detect Z components.
    ideal_z = _syndrome(x_frame, hz)
    ideal_x = _syndrome(z_frame, hx)
    if z_measurement_flips is None:
        z_measurement_flips = np.zeros_like(ideal_z, dtype=bool)
    if x_measurement_flips is None:
        x_measurement_flips = np.zeros_like(ideal_x, dtype=bool)
    if z_measurement_flips.shape != ideal_z.shape or x_measurement_flips.shape != ideal_x.shape:
        raise ValueError("measurement-flip arrays must match their syndrome arrays")
    measured_z = ideal_z ^ np.asarray(z_measurement_flips, dtype=np.uint8)
    measured_x = ideal_x ^ np.asarray(x_measurement_flips, dtype=np.uint8)
    return {
        "x_data_frame": x_frame.astype(np.uint8),
        "z_data_frame": z_frame.astype(np.uint8),
        "ideal_x_syndromes": ideal_x,
        "ideal_z_syndromes": ideal_z,
        "x_syndromes": measured_x,
        "z_syndromes": measured_z,
        "x_detection_events": detection_events(
            measured_x, include_final_boundary=include_final_boundary
        ),
        "z_detection_events": detection_events(
            measured_z, include_final_boundary=include_final_boundary
        ),
        "x_measurement_flips": np.asarray(x_measurement_flips, dtype=np.uint8),
        "z_measurement_flips": np.asarray(z_measurement_flips, dtype=np.uint8),
    }


def _check_excess(response_excess: np.ndarray, checks: np.ndarray) -> np.ndarray:
    if len(checks) == 0:
        return np.empty((*response_excess.shape[:2], 0), dtype=np.float32)
    weights = checks.sum(axis=1)
    return (np.matmul(response_excess, checks.T) / weights).astype(np.float32)


def generate_syndromes(
    probabilities: np.ndarray,
    baselines: np.ndarray,
    coords: np.ndarray,
    config: dict,
    *,
    seed: int | None = None,
    time_ms: np.ndarray | None = None,
    t1_us: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate Pauli faults, stabilizer measurements, and detection events."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (data_qubit, 2)")
    validate_syndrome_config(config, len(coords))
    include_intermediate_arrays = config.get("output", {}).get(
        "include_intermediate_arrays", config.get("schema_version", 1) == 1
    )
    probabilities = np.asarray(probabilities)
    baselines = np.asarray(baselines)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (event, time, data_qubit)")
    if probabilities.shape[2] != len(coords):
        raise ValueError("probabilities and coords disagree on data-qubit count")
    fault_backend = config.get("fault_probability_backend", "response_excess")
    round_t1_us: np.ndarray | None = None
    if fault_backend == "t1_pauli":
        if t1_us is None:
            raise ValueError(
                "fault_probability_backend='t1_pauli' requires the simulator t1_us array"
            )
        round_t1_us = np.asarray(t1_us, dtype=np.float32)
        if round_t1_us.shape != probabilities.shape:
            raise ValueError("t1_us must match probabilities shape before round resampling")
        if not np.isfinite(round_t1_us).all() or np.any(round_t1_us <= 0):
            raise ValueError("t1_us must be finite and strictly positive")
    allow_static_baseline = config.get(
        "allow_static_response_baseline", config.get("schema_version", 1) == 1
    )
    if baselines.ndim == 2 and not allow_static_baseline:
        raise ValueError(
            "schema version 2 requires time-dependent baseline_probabilities; "
            "regenerate the simulation, or set allow_static_response_baseline=true "
            "only when baseline drift is disabled"
        )
    input_samples = probabilities.shape[1]
    configured_round_duration = config.get("qec_round_duration_ms")
    inferred_round_duration = False
    round_time_ms: np.ndarray | None = None
    round_duration_ms = (
        float(configured_round_duration) if configured_round_duration is not None else None
    )
    if time_ms is not None:
        source_time_ms = np.asarray(time_ms, dtype=float)
        if round_duration_ms is None:
            if len(source_time_ms) < 2:
                raise ValueError(
                    "qec_round_duration_ms is required when the simulator has one time sample"
                )
            round_duration_ms = float(np.median(np.diff(source_time_ms)))
            inferred_round_duration = True
        probabilities, baselines, round_time_ms = resample_to_qec_rounds(
            probabilities,
            baselines,
            source_time_ms,
            round_duration_ms,
        )
        if round_t1_us is not None:
            # Relaxation rates, rather than coherence times, add linearly and
            # are therefore the appropriate quantities to interpolate.
            inverse_t1 = _interpolate_time_axis(
                1.0 / round_t1_us, source_time_ms, round_time_ms
            )
            round_t1_us = 1.0 / inverse_t1
    elif round_duration_ms is not None:
        round_time_ms = (
            np.arange(probabilities.shape[1], dtype=np.float32) * round_duration_ms
        )

    hx, hz, x_check_coords, z_check_coords, graph_edges = checks_from_config(config, coords)
    phenomenological_fault_probability, response_excess = response_to_fault_probability(
        probabilities,
        baselines,
        config,
        round_duration_ms=round_duration_ms,
    )
    rng = np.random.default_rng(config.get("seed", 0) if seed is None else seed)
    pauli_channel: dict[str, np.ndarray] | None = None
    if fault_backend == "t1_pauli":
        assert round_t1_us is not None
        assert round_duration_ms is not None
        t1_model = config.get("t1_pauli_model", {})
        pauli_channel = t1_to_pauli_probabilities(
            round_t1_us,
            round_duration_ms,
            pure_dephasing_time_us=t1_model.get("pure_dephasing_time_us"),
            include_identity=include_intermediate_arrays,
        )
        fault_probability = (
            pauli_channel["x"] + pauli_channel["y"] + pauli_channel["z"]
        ).astype(np.float32)
        pauli_faults, x_faults, z_faults = sample_pauli_channel(rng, pauli_channel)
    else:
        fault_probability = phenomenological_fault_probability
        pauli_faults, x_faults, z_faults = sample_pauli_faults(
            rng, fault_probability, config
        )

    if "measurement_error_rate_per_ms" in config:
        assert round_duration_ms is not None
        measurement_baseline = _probability_from_rate(
            config["measurement_error_rate_per_ms"], round_duration_ms
        )
    else:
        measurement_baseline = float(config.get("measurement_error_probability", 0.001))

    def measurement_probability(checks: np.ndarray) -> np.ndarray:
        exposure = _check_excess(response_excess, checks)
        if "measurement_error_radiation_rate_per_ms" in config:
            assert round_duration_ms is not None
            radiation = -np.expm1(
                -float(config["measurement_error_radiation_rate_per_ms"])
                * round_duration_ms
                * exposure
            )
        else:
            measurement_scale = float(config.get("measurement_error_radiation_scale", 0.0))
            radiation = np.clip(measurement_scale * exposure, 0.0, 1.0)
        return (1 - (1 - measurement_baseline) * (1 - radiation)).astype(np.float32)

    x_measurement_probability = measurement_probability(hx)
    z_measurement_probability = measurement_probability(hz)
    x_measurement_flips = rng.random(x_measurement_probability.shape) < x_measurement_probability
    z_measurement_flips = rng.random(z_measurement_probability.shape) < z_measurement_probability
    result = syndrome_from_faults(
        x_faults,
        z_faults,
        hx,
        hz,
        data_error_model=config.get("data_error_model", "pauli_frame"),
        x_measurement_flips=x_measurement_flips,
        z_measurement_flips=z_measurement_flips,
        include_final_boundary=config.get("include_final_boundary", True),
    )
    result.update(
        {
            "fault_probability": fault_probability,
            "response_excess": response_excess,
            "pauli_faults": pauli_faults,
            "x_faults": x_faults.astype(np.uint8),
            "z_faults": z_faults.astype(np.uint8),
            "x_checks": hx,
            "z_checks": hz,
            "data_coords": coords.astype(np.float32),
            "x_check_coords": x_check_coords,
            "z_check_coords": z_check_coords,
            "graph_edges": graph_edges,
            "x_measurement_error_probability": x_measurement_probability,
            "z_measurement_error_probability": z_measurement_probability,
        }
    )
    if pauli_channel is not None and include_intermediate_arrays:
        assert round_t1_us is not None
        result.update(
            {
                "t1_us": round_t1_us.astype(np.float32),
                "t2_us": pauli_channel["t2_us"],
                "pauli_i_probability": pauli_channel["i"],
                "pauli_x_probability": pauli_channel["x"],
                "pauli_y_probability": pauli_channel["y"],
                "pauli_z_probability": pauli_channel["z"],
            }
        )
    if round_time_ms is not None:
        result["time_ms"] = round_time_ms.astype(np.float32)
        detection_time_ms = round_time_ms
        if config.get("include_final_boundary", True):
            assert round_duration_ms is not None
            detection_time_ms = np.concatenate(
                (round_time_ms, [round_time_ms[-1] + round_duration_ms])
            )
        result["detection_time_ms"] = detection_time_ms.astype(np.float32)
    backend = config.get("backend", "graph_repetition_proxy")
    warnings = [
        "Circuit scheduling, ancilla propagation, reset faults, and leakage are not modeled.",
        "Decoder performance from these data is a stress-test result, not a hardware logical-error prediction.",
    ]
    if fault_backend == "response_excess":
        warnings.insert(
            0,
            "The RReCS response-to-Pauli mapping is phenomenological and is not calibrated to syndrome data.",
        )
    else:
        warnings.insert(
            0,
            "The T1/T2 channel is Pauli twirled; non-unital relaxation and finite-temperature excitation bias are not retained.",
        )
        warnings.append(
            "QP parameters and pure dephasing have not been calibrated against Google circuit-level syndrome data."
        )
        if (
            config.get("measurement_error_radiation_scale", 0.0) != 0
            or config.get("measurement_error_radiation_rate_per_ms", 0.0) != 0
        ):
            warnings.append(
                "Radiation-dependent measurement faults still use the RReCS response proxy because ancilla T1 is unavailable."
            )
    if baselines.ndim == 2:
        warnings.append(
            "Only a static response baseline was supplied; archives with baseline drift should be regenerated to include baseline_probabilities."
        )
    if inferred_round_duration:
        warnings.append(
            "qec_round_duration_ms was absent, so the simulator sample interval was used for legacy compatibility."
        )
    if not config.get("include_final_boundary", True):
        warnings.append("Final zero-boundary detection events are disabled.")
    if backend == "graph_repetition_proxy":
        warnings.append(
            "graph_repetition_proxy measures pair parity on the supplied coordinate graph; it is not a circuit-level QEC layout."
        )
    metadata = {
        "schema_version": 3 if fault_backend == "t1_pauli" else 2,
        "backend": backend,
        "fault_probability_backend": fault_backend,
        "seed": int(config.get("seed", 0) if seed is None else seed),
        "events": int(probabilities.shape[0]),
        "input_samples_per_event": int(input_samples),
        "rounds_per_event": int(probabilities.shape[1]),
        "detection_rounds_per_event": int(
            result["z_detection_events"].shape[1]
            if len(hz)
            else result["x_detection_events"].shape[1]
        ),
        "data_qubits": int(probabilities.shape[2]),
        "x_checks": int(len(hx)),
        "z_checks": int(len(hz)),
        "data_error_model": config.get("data_error_model", "pauli_frame"),
        "include_final_boundary": config.get("include_final_boundary", True),
        "qec_round_duration_ms": round_duration_ms,
        "round_mapping": (
            "latent response linearly interpolated to fixed-duration QEC rounds"
            if time_ms is not None and configured_round_duration is not None
            else "input samples treated as QEC rounds"
        ),
        "response_baseline": (
            "time_dependent_baseline_probabilities"
            if baselines.ndim == 3
            else "static_event_baselines"
        ),
        "mean_fault_probability": float(fault_probability.mean()),
        "mean_pauli_x_probability": (
            float(pauli_channel["x"].mean()) if pauli_channel is not None else None
        ),
        "mean_pauli_y_probability": (
            float(pauli_channel["y"].mean()) if pauli_channel is not None else None
        ),
        "mean_pauli_z_probability": (
            float(pauli_channel["z"].mean()) if pauli_channel is not None else None
        ),
        "mean_t1_us": (
            float(round_t1_us.mean()) if round_t1_us is not None else None
        ),
        "mean_t2_us": (
            float(pauli_channel["t2_us"].mean())
            if pauli_channel is not None
            else None
        ),
        "sampled_fault_fraction": float((pauli_faults != PAULI_I).mean()),
        "x_detection_event_fraction": (
            float(result["x_detection_events"].mean()) if len(hx) else None
        ),
        "z_detection_event_fraction": (
            float(result["z_detection_events"].mean()) if len(hz) else None
        ),
        "pauli_label_encoding": {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z},
        "configuration": config,
        "warnings": warnings,
    }
    if not include_intermediate_arrays:
        intermediate_arrays = {
            "fault_probability",
            "response_excess",
            "x_faults",
            "z_faults",
            "x_data_frame",
            "z_data_frame",
            "ideal_x_syndromes",
            "ideal_z_syndromes",
            "x_measurement_flips",
            "z_measurement_flips",
            "x_measurement_error_probability",
            "z_measurement_error_probability",
            "t1_us",
            "t2_us",
            "pauli_i_probability",
            "pauli_x_probability",
            "pauli_y_probability",
            "pauli_z_probability",
        }
        for name in intermediate_arrays:
            result.pop(name, None)
    metadata["returned_arrays"] = sorted(result)
    return result, metadata


def save_syndrome_output(output: Path, result: dict[str, np.ndarray], metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "syndrome_events.npz", **result)
    (output / "syndrome_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    from .configuration import config_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", default="simulation_results/simulated_events.npz")
    parser.add_argument("--config", default=config_path("syndrome_proxy_t1_pauli"))
    parser.add_argument("--output", default="syndrome_results")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    archive = np.load(args.simulation)
    required = {"probabilities", "baselines", "coords", "time_ms"}
    if config.get("fault_probability_backend", "response_excess") == "t1_pauli":
        required.add("t1_us")
    missing = required - set(archive.files)
    if missing:
        raise ValueError(
            f"simulation archive is missing {sorted(missing)}; rerun qp-ode-simulate "
            "with --syndrome-config or enable output.save_probabilities"
        )
    response_baselines = (
        archive["baseline_probabilities"]
        if "baseline_probabilities" in archive.files
        else archive["baselines"]
    )
    result, metadata = generate_syndromes(
        archive["probabilities"],
        response_baselines,
        archive["coords"],
        config,
        seed=args.seed,
        time_ms=archive["time_ms"],
        t1_us=(archive["t1_us"] if "t1_us" in archive.files else None),
    )
    metadata["source_simulation"] = str(args.simulation)
    metadata["simulator_time_step_ms"] = float(np.median(np.diff(archive["time_ms"])))
    save_syndrome_output(Path(args.output), result, metadata)
    print(
        f"generated {metadata['events']} events x {metadata['rounds_per_event']} rounds; "
        f"X checks={metadata['x_checks']}, Z checks={metadata['z_checks']}"
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
