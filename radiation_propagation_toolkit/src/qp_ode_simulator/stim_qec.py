#!/usr/bin/env python3
"""Run the radiation/QP model through a real Stim QEC circuit.

This module is the circuit-level alternative to the algebraic syndrome proxy in
``syndrome.py``.  A continuous radiation field is evaluated at every
active circuit qubit and converted from T1/T2 to duration-aware Pauli channels.
For gate-sliced runs, each channel is placed after the operations in its TICK
interval and before the next TICK/measurement boundary.  Stim then performs the
ancilla resets, Clifford gates, measurements, detector comparisons, and logical-
observable bookkeeping defined by the generated circuit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .syndrome import t1_to_pauli_probabilities
from .simulator import load_config, simulate


ROLE_DATA = 0
ROLE_X_ANCILLA = 1
ROLE_Z_ANCILLA = 2
ROLE_NAMES = {
    ROLE_DATA: "data",
    ROLE_X_ANCILLA: "x_ancilla",
    ROLE_Z_ANCILLA: "z_ancilla",
}

SUPPORTED_SCHEMA_VERSIONS = {1}
KNOWN_CONFIG_KEYS = {
    "schema_version",
    "backend",
    "task",
    "distance",
    "rounds",
    "round_duration_ms",
    "start_time_ms",
    "window_role",
    "shots_per_event",
    "layout",
    "circuit_noise",
    "gate_schedule",
    "noise_accounting",
    "radiation_channel",
    "decoder",
    "seed",
}


@dataclass(frozen=True)
class StimLayout:
    """Stim template circuit and its compact physical-qubit coordinate table."""

    circuit: Any
    qubit_ids: np.ndarray
    grid_coords: np.ndarray
    physical_coords_mm: np.ndarray
    roles: np.ndarray
    detector_coords: np.ndarray


def _import_stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError(
            "Stim is required for circuit-level syndrome generation. "
            "Install the QEC dependencies with: pip install 'qp-ode-simulator[qec]'"
        ) from exc
    return stim


def _probability(name: str, value: Any) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def validate_stim_config(config: dict) -> None:
    """Validate the configuration before building a potentially large circuit."""
    if not isinstance(config, dict):
        raise ValueError("Stim configuration must be a JSON object")
    unknown = set(config) - KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown Stim configuration keys: {sorted(unknown)}")
    if config.get("schema_version", 1) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported Stim schema_version")
    if config.get("backend", "stim_surface_code") != "stim_surface_code":
        raise ValueError("backend must be 'stim_surface_code'")
    task = config.get("task", "surface_code:rotated_memory_z")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty Stim generated-circuit task")
    for name in ("distance", "rounds", "shots_per_event"):
        value = config.get(name, 1 if name == "shots_per_event" else None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    round_duration = float(config.get("round_duration_ms", np.nan))
    start_time = float(config.get("start_time_ms", np.nan))
    if not np.isfinite(round_duration) or round_duration <= 0:
        raise ValueError("round_duration_ms must be finite and strictly positive")
    if not np.isfinite(start_time):
        raise ValueError("start_time_ms must be finite")
    window_role = config.get("window_role", "unspecified")
    if window_role not in {"unspecified", "onset_and_peak", "long_recovery"}:
        raise ValueError(
            "window_role must be 'unspecified', 'onset_and_peak', or "
            "'long_recovery'"
        )

    layout = config.get("layout", {})
    if not isinstance(layout, dict):
        raise ValueError("layout must be an object")
    allowed_layout = {"qubit_pitch_mm", "rotation_degrees", "center_mm"}
    unknown_layout = set(layout) - allowed_layout
    if unknown_layout:
        raise ValueError(f"unknown layout keys: {sorted(unknown_layout)}")
    pitch = float(layout.get("qubit_pitch_mm", 1.0))
    rotation = float(layout.get("rotation_degrees", 0.0))
    center = np.asarray(layout.get("center_mm", [0.0, 0.0]), dtype=float)
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("layout.qubit_pitch_mm must be strictly positive")
    if not np.isfinite(rotation):
        raise ValueError("layout.rotation_degrees must be finite")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("layout.center_mm must contain two finite coordinates")

    circuit_noise = config.get("circuit_noise", {})
    if not isinstance(circuit_noise, dict):
        raise ValueError("circuit_noise must be an object")
    allowed_noise = {
        "after_clifford_depolarization",
        "before_round_data_depolarization",
        "before_measure_flip_probability",
        "after_reset_flip_probability",
    }
    unknown_noise = set(circuit_noise) - allowed_noise
    if unknown_noise:
        raise ValueError(f"unknown circuit_noise keys: {sorted(unknown_noise)}")
    for name, value in circuit_noise.items():
        _probability(f"circuit_noise.{name}", value)

    noise_accounting = config.get("noise_accounting", {})
    if not isinstance(noise_accounting, dict):
        raise ValueError("noise_accounting must be an object")
    allowed_accounting = {"depolarizing_channels_exclude_t1_t2"}
    unknown_accounting = set(noise_accounting) - allowed_accounting
    if unknown_accounting:
        raise ValueError(
            f"unknown noise_accounting keys: {sorted(unknown_accounting)}"
        )
    excludes_t1_t2 = noise_accounting.get(
        "depolarizing_channels_exclude_t1_t2"
    )
    if excludes_t1_t2 is not None and not isinstance(excludes_t1_t2, bool):
        raise ValueError(
            "noise_accounting.depolarizing_channels_exclude_t1_t2 must be boolean"
        )
    depolarizing_is_nonzero = any(
        float(circuit_noise.get(name, 0.0)) > 0
        for name in (
            "after_clifford_depolarization",
            "before_round_data_depolarization",
        )
    )
    if depolarizing_is_nonzero and excludes_t1_t2 is not True:
        raise ValueError(
            "nonzero circuit depolarization must be explicitly declared residual "
            "noise with noise_accounting.depolarizing_channels_exclude_t1_t2=true; "
            "otherwise the QP-derived T1/T2 channel may be counted twice"
        )

    gate_schedule = config.get("gate_schedule", {})
    if not isinstance(gate_schedule, dict):
        raise ValueError("gate_schedule must be an object")
    allowed_schedule = {"mode", "tick_durations_ns"}
    unknown_schedule = set(gate_schedule) - allowed_schedule
    if unknown_schedule:
        raise ValueError(f"unknown gate_schedule keys: {sorted(unknown_schedule)}")
    schedule_mode = gate_schedule.get("mode", "uniform")
    if schedule_mode not in {"uniform", "explicit"}:
        raise ValueError("gate_schedule.mode must be 'uniform' or 'explicit'")
    if schedule_mode == "explicit":
        durations = np.asarray(gate_schedule.get("tick_durations_ns", []), dtype=float)
        if durations.ndim != 1 or len(durations) == 0:
            raise ValueError(
                "gate_schedule.tick_durations_ns must be a non-empty list"
            )
        if not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("gate_schedule.tick_durations_ns must be finite and positive")
        expected_ns = round_duration * 1e6
        if not np.isclose(durations.sum(), expected_ns, rtol=1e-9, atol=1e-6):
            raise ValueError(
                "gate_schedule.tick_durations_ns must sum to round_duration_ms"
            )

    radiation = config.get("radiation_channel", {})
    if not isinstance(radiation, dict):
        raise ValueError("radiation_channel must be an object")
    allowed_radiation = {
        "targets",
        "pure_dephasing_time_us",
        "timing",
        "coherent_phase_approximation",
    }
    unknown_radiation = set(radiation) - allowed_radiation
    if unknown_radiation:
        raise ValueError(
            f"unknown radiation_channel keys: {sorted(unknown_radiation)}"
        )
    if radiation.get("targets", "all") not in {"all", "data"}:
        raise ValueError("radiation_channel.targets must be 'all' or 'data'")
    if radiation.get("timing", "gate_slices") not in {
        "gate_slices",
        "round_boundary",
    }:
        raise ValueError(
            "radiation_channel.timing must be 'gate_slices' or 'round_boundary'"
        )
    if "pure_dephasing_time_us" in radiation:
        tphi = float(radiation["pure_dephasing_time_us"])
        if not np.isfinite(tphi) or tphi <= 0:
            raise ValueError(
                "radiation_channel.pure_dephasing_time_us must be strictly positive"
            )
    phase = radiation.get("coherent_phase_approximation", {})
    if not isinstance(phase, dict):
        raise ValueError(
            "radiation_channel.coherent_phase_approximation must be an object"
        )
    unknown_phase = set(phase) - {"enabled", "method"}
    if unknown_phase:
        raise ValueError(
            "unknown radiation_channel.coherent_phase_approximation keys: "
            f"{sorted(unknown_phase)}"
        )
    if not isinstance(phase.get("enabled", False), bool):
        raise ValueError(
            "radiation_channel.coherent_phase_approximation.enabled must be boolean"
        )
    if phase.get("enabled", False):
        if phase.get("method", "pauli_twirl") != "pauli_twirl":
            raise ValueError("only coherent phase method 'pauli_twirl' is supported")
        if radiation.get("timing", "gate_slices") != "gate_slices":
            raise ValueError(
                "coherent phase approximation requires radiation_channel.timing='gate_slices'"
            )

    decoder = config.get("decoder", {})
    if not isinstance(decoder, dict):
        raise ValueError("decoder must be an object")
    allowed_decoder = {"enabled", "mode", "allow_oracle", "baseline_reference"}
    if set(decoder) - allowed_decoder:
        raise ValueError(
            f"unknown decoder keys: {sorted(set(decoder) - allowed_decoder)}"
        )
    if not isinstance(decoder.get("enabled", False), bool):
        raise ValueError("decoder.enabled must be boolean")
    decoder_mode = decoder.get("mode", "fixed_circuit")
    if decoder_mode not in {"fixed_circuit", "oracle_event"}:
        raise ValueError("decoder.mode must be 'fixed_circuit' or 'oracle_event'")
    if not isinstance(decoder.get("allow_oracle", False), bool):
        raise ValueError("decoder.allow_oracle must be boolean")
    if decoder.get("baseline_reference", "pre_event_median") not in {
        "pre_event_median",
        "maximum_t1",
    }:
        raise ValueError(
            "decoder.baseline_reference must be 'pre_event_median' or 'maximum_t1'"
        )
    if (
        decoder.get("enabled", False)
        and decoder_mode == "oracle_event"
        and not decoder.get("allow_oracle", False)
    ):
        raise ValueError(
            "oracle_event decoding requires decoder.allow_oracle=true; "
            "it is an optimistic upper bound, not a deployable decoder"
        )


def _instruction_qubits(instruction: Any) -> set[int]:
    return {
        int(target.value)
        for target in instruction.targets_copy()
        if target.is_qubit_target
    }


def _physical_coordinates(grid_coords: np.ndarray, layout_config: dict) -> np.ndarray:
    centered = grid_coords - 0.5 * (
        np.min(grid_coords, axis=0) + np.max(grid_coords, axis=0)
    )
    angle = np.deg2rad(float(layout_config.get("rotation_degrees", 0.0)))
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    pitch = float(layout_config.get("qubit_pitch_mm", 1.0))
    center = np.asarray(layout_config.get("center_mm", [0.0, 0.0]), dtype=float)
    return centered @ rotation.T * pitch + center


def build_stim_layout(config: dict) -> StimLayout:
    """Build a generated code circuit and classify its active qubits."""
    validate_stim_config(config)
    stim = _import_stim()
    circuit_noise = config.get("circuit_noise", {})
    circuit = stim.Circuit.generated(
        config.get("task", "surface_code:rotated_memory_z"),
        distance=int(config["distance"]),
        rounds=int(config["rounds"]),
        **{
            name: float(circuit_noise.get(name, 0.0))
            for name in (
                "after_clifford_depolarization",
                "before_round_data_depolarization",
                "before_measure_flip_probability",
                "after_reset_flip_probability",
            )
        },
    )

    coordinate_by_qubit: dict[int, tuple[float, float]] = {}
    measured_reset_qubits: set[int] = set()
    hadamard_qubits: set[int] = set()
    for instruction in circuit.flattened():
        if instruction.name == "QUBIT_COORDS":
            targets = list(_instruction_qubits(instruction))
            args = instruction.gate_args_copy()
            if len(targets) == 1 and len(args) >= 2:
                coordinate_by_qubit[targets[0]] = (float(args[0]), float(args[1]))
        elif instruction.name in {"MR", "MRX", "MRY"}:
            measured_reset_qubits.update(_instruction_qubits(instruction))
        elif instruction.name == "H":
            hadamard_qubits.update(_instruction_qubits(instruction))
    if not coordinate_by_qubit:
        raise ValueError("generated circuit does not contain QUBIT_COORDS annotations")

    qubit_ids = np.asarray(sorted(coordinate_by_qubit), dtype=np.int32)
    grid_coords = np.asarray(
        [coordinate_by_qubit[int(qubit)] for qubit in qubit_ids], dtype=np.float64
    )
    roles = np.full(len(qubit_ids), ROLE_DATA, dtype=np.uint8)
    for index, qubit in enumerate(qubit_ids):
        if int(qubit) in measured_reset_qubits:
            roles[index] = (
                ROLE_X_ANCILLA
                if int(qubit) in hadamard_qubits
                else ROLE_Z_ANCILLA
            )

    raw_detector_coords = circuit.get_detector_coordinates()
    detector_coords = np.zeros((circuit.num_detectors, 3), dtype=np.float32)
    for detector, coordinate in raw_detector_coords.items():
        values = np.asarray(coordinate[:3], dtype=np.float32)
        detector_coords[int(detector), : len(values)] = values
    return StimLayout(
        circuit=circuit,
        qubit_ids=qubit_ids,
        grid_coords=grid_coords,
        physical_coords_mm=_physical_coordinates(grid_coords, config.get("layout", {})),
        roles=roles,
        detector_coords=detector_coords,
    )


def interpolate_inverse_t1(
    t1_us: np.ndarray, source_time_ms: np.ndarray, target_time_ms: np.ndarray
) -> np.ndarray:
    """Interpolate relaxation rate (1/T1), then convert it back to T1."""
    t1 = np.asarray(t1_us, dtype=np.float64)
    source = np.asarray(source_time_ms, dtype=np.float64)
    target = np.asarray(target_time_ms, dtype=np.float64)
    if t1.ndim != 3 or t1.shape[1] != len(source):
        raise ValueError("t1_us must have shape (event, source_time, qubit)")
    if len(source) == 0 or np.any(np.diff(source) <= 0):
        raise ValueError("source_time_ms must be non-empty and strictly increasing")
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("target_time_ms must be a finite one-dimensional array")
    tolerance = max(1e-12, 1e-9 * max(abs(source[0]), abs(source[-1]), 1.0))
    if target.min() < source[0] - tolerance or target.max() > source[-1] + tolerance:
        raise ValueError("QEC round times lie outside the simulated T1 time interval")
    if not np.isfinite(t1).all() or np.any(t1 <= 0):
        raise ValueError("t1_us must be finite and strictly positive")

    rate = 1.0 / t1
    result_rate = np.empty((t1.shape[0], len(target), t1.shape[2]), dtype=np.float64)
    for event in range(t1.shape[0]):
        for qubit in range(t1.shape[2]):
            result_rate[event, :, qubit] = np.interp(
                target, source, rate[event, :, qubit]
            )
    return (1.0 / result_rate).astype(np.float32)


def average_t1_over_rounds(
    t1_us: np.ndarray,
    source_time_ms: np.ndarray,
    round_start_time_ms: np.ndarray,
    round_duration_ms: float,
) -> np.ndarray:
    """Return the effective T1 from the integrated relaxation rate per round.

    The channel exponent is determined by ``integral dt/T1(t)``.  Averaging
    ``1/T1`` therefore preserves the correct decay probability even while the
    radiation pulse is changing within a QEC round.
    """
    t1 = np.asarray(t1_us, dtype=np.float64)
    source = np.asarray(source_time_ms, dtype=np.float64)
    starts = np.asarray(round_start_time_ms, dtype=np.float64)
    duration = float(round_duration_ms)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("round_duration_ms must be finite and strictly positive")
    if t1.ndim != 3 or t1.shape[1] != len(source):
        raise ValueError("t1_us must have shape (event, source_time, qubit)")
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("source_time_ms must contain at least two increasing values")
    if not np.isfinite(t1).all() or np.any(t1 <= 0):
        raise ValueError("t1_us must be finite and strictly positive")
    ends = starts + duration
    tolerance = max(1e-12, 1e-9 * max(abs(source[0]), abs(source[-1]), 1.0))
    if starts.min() < source[0] - tolerance or ends.max() > source[-1] + tolerance:
        raise ValueError("QEC round intervals lie outside the simulated T1 time interval")

    rate = 1.0 / t1
    source_step = np.diff(source)
    cumulative = np.zeros_like(rate)
    cumulative[:, 1:, :] = np.cumsum(
        0.5
        * (rate[:, :-1, :] + rate[:, 1:, :])
        * source_step[None, :, None],
        axis=1,
    )

    def integrated_rate(times: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(source, times, side="right") - 1
        indices = np.clip(indices, 0, len(source) - 2)
        offset = times - source[indices]
        interval = source[indices + 1] - source[indices]
        r0 = rate[:, indices, :]
        r1 = rate[:, indices + 1, :]
        slope = (r1 - r0) / interval[None, :, None]
        return (
            cumulative[:, indices, :]
            + r0 * offset[None, :, None]
            + 0.5 * slope * np.square(offset)[None, :, None]
        )

    average_rate = (integrated_rate(ends) - integrated_rate(starts)) / duration
    return (1.0 / average_rate).astype(np.float32)


def inject_round_pauli_channels(
    circuit: Any,
    qubit_ids: np.ndarray,
    pauli_x: np.ndarray,
    pauli_y: np.ndarray,
    pauli_z: np.ndarray,
) -> Any:
    """Insert one location-dependent Pauli channel before each QEC round."""
    stim = _import_stim()
    x = np.asarray(pauli_x, dtype=float)
    y = np.asarray(pauli_y, dtype=float)
    z = np.asarray(pauli_z, dtype=float)
    expected = (x.shape[0], len(qubit_ids)) if x.ndim == 2 else None
    if x.ndim != 2 or y.shape != x.shape or z.shape != x.shape:
        raise ValueError("Pauli arrays must share shape (round, active_qubit)")
    if expected != x.shape:
        raise ValueError("Pauli arrays and qubit_ids disagree")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError("Pauli probabilities must be finite")
    if np.any(x < 0) or np.any(y < 0) or np.any(z < 0) or np.any(x + y + z > 1 + 1e-7):
        raise ValueError("invalid mutually exclusive Pauli probabilities")

    output = stim.Circuit()
    round_index = 0
    at_round_start = True
    for instruction in circuit.flattened():
        if instruction.name == "TICK" and at_round_start:
            if round_index >= len(x):
                raise ValueError("circuit contains more QEC rounds than the Pauli arrays")
            _append_location_dependent_pauli_channels(
                output, qubit_ids, x[round_index], y[round_index], z[round_index]
            )
            round_index += 1
            at_round_start = False
        output.append(instruction)
        if instruction.name in {"MR", "MRX", "MRY"}:
            at_round_start = True
    if round_index != len(x):
        raise ValueError(
            f"expected {len(x)} QEC rounds but found {round_index} in the circuit"
        )
    return output


def _append_location_dependent_pauli_channels(
    output: Any,
    qubit_ids: np.ndarray,
    pauli_x: np.ndarray,
    pauli_y: np.ndarray,
    pauli_z: np.ndarray,
) -> None:
    """Append one instruction per unique probability triple, not per qubit."""
    grouped_targets: dict[tuple[float, float, float], list[int]] = {}
    for column, qubit in enumerate(qubit_ids):
        probabilities = (
            float(pauli_x[column]),
            float(pauli_y[column]),
            float(pauli_z[column]),
        )
        if sum(probabilities) > 0:
            grouped_targets.setdefault(probabilities, []).append(int(qubit))
    for probabilities, targets in grouped_targets.items():
        output.append("PAULI_CHANNEL_1", targets, probabilities)


def round_tick_counts(circuit: Any, expected_rounds: int) -> np.ndarray:
    """Count scheduled TICK intervals in each ancilla-measurement round."""
    counts: list[int] = []
    awaiting_round_start = True
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            if awaiting_round_start:
                counts.append(0)
                awaiting_round_start = False
            counts[-1] += 1
        if instruction.name in {"MR", "MRX", "MRY"}:
            awaiting_round_start = True
    result = np.asarray(counts, dtype=np.int32)
    if len(result) != expected_rounds:
        raise ValueError(
            f"expected {expected_rounds} QEC rounds but found {len(result)} in the circuit"
        )
    if np.any(result <= 0):
        raise ValueError("each QEC round must contain at least one TICK interval")
    return result


def gate_slice_pauli_probabilities(
    round_t1_us: np.ndarray,
    round_duration_ms: float,
    tick_counts: np.ndarray,
    *,
    round_t2_us: np.ndarray | None = None,
    pure_dephasing_time_us: float | None = None,
) -> dict[str, np.ndarray]:
    """Return the channel used at every TICK within each circuit round.

    Repeating this smaller channel ``tick_counts[round]`` times composes to the
    same round-integrated Pauli-twirled T1/T2 channel when no gates intervene.
    Placing the pieces between gates additionally captures propagation timing.
    """
    t1 = np.asarray(round_t1_us, dtype=np.float32)
    counts = np.asarray(tick_counts, dtype=np.int32)
    if t1.ndim != 3 or counts.shape != (t1.shape[1],) or np.any(counts <= 0):
        raise ValueError("tick_counts must contain one positive value per QEC round")
    result = {
        name: np.empty_like(t1, dtype=np.float32)
        for name in ("i", "x", "y", "z", "t2_us")
    }
    for count in np.unique(counts):
        selected = counts == count
        channel = t1_to_pauli_probabilities(
            t1[:, selected, :],
            float(round_duration_ms) / int(count),
            t2_us=(
                None
                if round_t2_us is None
                else np.asarray(round_t2_us, dtype=np.float32)[:, selected, :]
            ),
            pure_dephasing_time_us=pure_dephasing_time_us,
        )
        for name in result:
            result[name][:, selected, :] = channel[name]
    return result


def gate_slice_durations_ms(
    round_duration_ms: float,
    tick_counts: np.ndarray,
    gate_schedule: dict | None = None,
) -> np.ndarray:
    """Return a padded [round, slice] duration table for the Stim TICK schedule."""
    counts = np.asarray(tick_counts, dtype=np.int32)
    if counts.ndim != 1 or np.any(counts <= 0):
        raise ValueError("tick_counts must contain positive values")
    maximum = int(counts.max())
    durations = np.zeros((len(counts), maximum), dtype=np.float64)
    schedule = {} if gate_schedule is None else gate_schedule
    if schedule.get("mode", "uniform") == "explicit":
        explicit = np.asarray(schedule["tick_durations_ns"], dtype=float) * 1e-6
        if np.any(counts != len(explicit)):
            raise ValueError(
                "explicit gate schedule length must match the TICK count in every round"
            )
        durations[:, : len(explicit)] = explicit
    else:
        for index, count in enumerate(counts):
            durations[index, :count] = float(round_duration_ms) / int(count)
    return durations


def _compose_independent_z_channel(
    channel: dict[str, np.ndarray], z_probability: np.ndarray
) -> dict[str, np.ndarray]:
    """Compose a Pauli channel with an independent Z channel exactly."""
    q = np.asarray(z_probability, dtype=np.float32)
    i = channel["i"]
    x = channel["x"]
    y = channel["y"]
    z = channel["z"]
    return {
        **channel,
        "i": ((1 - q) * i + q * z).astype(np.float32),
        "x": ((1 - q) * x + q * y).astype(np.float32),
        "y": ((1 - q) * y + q * x).astype(np.float32),
        "z": ((1 - q) * z + q * i).astype(np.float32),
    }


def scheduled_gate_slice_pauli_probabilities(
    round_t1_us: np.ndarray,
    slice_duration_ms: np.ndarray,
    tick_counts: np.ndarray,
    *,
    round_t2_us: np.ndarray | None = None,
    round_frequency_shift_ghz: np.ndarray | None = None,
    pure_dephasing_time_us: float | None = None,
    include_coherent_phase_approximation: bool = False,
) -> dict[str, np.ndarray]:
    """Build the exact dissipative/phase-approximated channel for every TICK."""
    t1 = np.asarray(round_t1_us, dtype=np.float32)
    counts = np.asarray(tick_counts, dtype=np.int32)
    durations = np.asarray(slice_duration_ms, dtype=float)
    if t1.ndim != 3 or durations.shape[0] != t1.shape[1]:
        raise ValueError("round T1 and gate-slice duration tables disagree")
    maximum = durations.shape[1]
    shape = (t1.shape[0], t1.shape[1], maximum, t1.shape[2])
    result = {
        name: np.zeros(shape, dtype=np.float32)
        for name in ("i", "x", "y", "z", "t2_us", "phase_z_probability")
    }
    # Equal-duration slices share the same elementwise channel calculation.
    # Batching them avoids tens of thousands of Python calls for long windows,
    # without changing channel placement, floating-point formulas, or sampling.
    if counts.shape != (t1.shape[1],) or np.any(counts < 0) or np.any(counts > maximum):
        raise ValueError("invalid gate-slice tick counts")
    active = np.arange(maximum)[None, :] < counts[:, None]
    for value in np.unique(durations[active]):
        rr, ss = np.nonzero(active & (durations == value))
        duration = float(value)
        channel = t1_to_pauli_probabilities(
            t1[:, rr, :], duration,
            t2_us=None if round_t2_us is None else np.asarray(round_t2_us, dtype=np.float32)[:, rr, :],
            pure_dephasing_time_us=pure_dephasing_time_us,
        )
        if include_coherent_phase_approximation:
            if round_frequency_shift_ghz is None:
                raise ValueError("coherent phase approximation requires frequency_shift_ghz")
            shift = np.asarray(round_frequency_shift_ghz, dtype=float)[:, rr, :]
            phase_z = np.sin(np.pi * shift * (duration * 1e6)) ** 2
            channel = _compose_independent_z_channel(channel, phase_z)
            result["phase_z_probability"][:, rr, ss, :] = phase_z
        for name in ("i", "x", "y", "z", "t2_us"):
            result[name][:, rr, ss, :] = channel[name]
    return result


def inject_gate_slice_pauli_channels(
    circuit: Any,
    qubit_ids: np.ndarray,
    pauli_x: np.ndarray,
    pauli_y: np.ndarray,
    pauli_z: np.ndarray,
    tick_counts: np.ndarray,
) -> Any:
    """Insert one repeated channel at the end of every TICK-delimited slice."""
    stim = _import_stim()
    x = np.asarray(pauli_x, dtype=float)
    y = np.asarray(pauli_y, dtype=float)
    z = np.asarray(pauli_z, dtype=float)
    counts = np.asarray(tick_counts, dtype=np.int32)
    if x.ndim != 2 or y.shape != x.shape or z.shape != x.shape:
        raise ValueError("Pauli arrays must share shape (round, active_qubit)")
    if x.shape != (len(counts), len(qubit_ids)):
        raise ValueError("Pauli arrays, tick_counts, and qubit_ids disagree")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError("Pauli probabilities must be finite")
    if np.any(x < 0) or np.any(y < 0) or np.any(z < 0) or np.any(x + y + z > 1 + 1e-7):
        raise ValueError("invalid mutually exclusive Pauli probabilities")

    output = stim.Circuit()
    round_index = -1
    slice_index = 0
    slice_active = False
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            if not slice_active:
                round_index += 1
                slice_index = 0
                slice_active = True
            else:
                if round_index >= len(counts) or slice_index >= counts[round_index]:
                    raise ValueError("circuit TICK schedule disagrees with tick_counts")
                _append_location_dependent_pauli_channels(
                    output, qubit_ids, x[round_index], y[round_index], z[round_index]
                )
                slice_index += 1
            output.append(instruction)
            continue
        if instruction.name in {"MR", "MRX", "MRY"}:
            if not slice_active or round_index < 0:
                raise ValueError("circuit measurement occurred outside a QEC round")
            if round_index >= len(counts) or slice_index >= counts[round_index]:
                raise ValueError("circuit TICK schedule disagrees with tick_counts")
            _append_location_dependent_pauli_channels(
                output, qubit_ids, x[round_index], y[round_index], z[round_index]
            )
            slice_index += 1
            if slice_index != counts[round_index]:
                raise ValueError("circuit round ended before all TICK channels were inserted")
            output.append(instruction)
            slice_active = False
            continue
        output.append(instruction)
    if round_index + 1 != len(counts):
        raise ValueError("circuit contains fewer QEC rounds than tick_counts")
    return output


def inject_scheduled_gate_slice_pauli_channels(
    circuit: Any,
    qubit_ids: np.ndarray,
    pauli_x: np.ndarray,
    pauli_y: np.ndarray,
    pauli_z: np.ndarray,
    tick_counts: np.ndarray,
) -> Any:
    """Insert a separately parameterized channel at each gate-slice end.

    A slice begins after a TICK.  Its channel is appended after the operations in
    that slice and immediately before the next TICK or ancilla measurement.  This
    matches the conventional placement of a gate-duration noise channel after the
    corresponding ideal gate instead of accidentally conjugating it through that
    gate.
    """
    stim = _import_stim()
    x = np.asarray(pauli_x, dtype=float)
    y = np.asarray(pauli_y, dtype=float)
    z = np.asarray(pauli_z, dtype=float)
    counts = np.asarray(tick_counts, dtype=np.int32)
    if x.ndim != 3 or y.shape != x.shape or z.shape != x.shape:
        raise ValueError("scheduled Pauli arrays must share shape (round, slice, qubit)")
    if x.shape[0] != len(counts) or x.shape[2] != len(qubit_ids):
        raise ValueError("scheduled Pauli arrays, tick_counts, and qubit_ids disagree")
    if np.any(x < 0) or np.any(y < 0) or np.any(z < 0) or np.any(x + y + z > 1 + 1e-7):
        raise ValueError("invalid scheduled mutually exclusive Pauli probabilities")

    from .circuit_building import CircuitAccumulator
    output = CircuitAccumulator()
    round_index = -1
    slice_index = 0
    slice_active = False
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            if not slice_active:
                round_index += 1
                slice_index = 0
                slice_active = True
            else:
                if round_index >= len(counts) or slice_index >= counts[round_index]:
                    raise ValueError(
                        "circuit TICK schedule disagrees with scheduled channels"
                    )
                _append_location_dependent_pauli_channels(
                    output,
                    qubit_ids,
                    x[round_index, slice_index],
                    y[round_index, slice_index],
                    z[round_index, slice_index],
                )
                slice_index += 1
            output.append(instruction)
            continue
        if instruction.name in {"MR", "MRX", "MRY"}:
            if not slice_active or round_index < 0:
                raise ValueError("circuit measurement occurred outside a QEC round")
            if round_index >= len(counts) or slice_index >= counts[round_index]:
                raise ValueError("circuit TICK schedule disagrees with scheduled channels")
            _append_location_dependent_pauli_channels(
                output,
                qubit_ids,
                x[round_index, slice_index],
                y[round_index, slice_index],
                z[round_index, slice_index],
            )
            slice_index += 1
            if slice_index != counts[round_index]:
                raise ValueError("circuit round ended before all scheduled channels")
            output.append(instruction)
            slice_active = False
            continue
        output.append(instruction)
    if round_index + 1 != len(counts):
        raise ValueError("circuit contains fewer QEC rounds than scheduled channels")
    return output.finish()


def _build_matching(circuit: Any) -> Any:
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "PyMatching is required when decoder.enabled=true. "
            "Install the QEC dependencies with: pip install 'qp-ode-simulator[qec]'"
        ) from exc
    detector_error_model = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    return pymatching.Matching.from_detector_error_model(detector_error_model)


def _decode(matching: Any, detector_events: np.ndarray) -> np.ndarray:
    prediction = np.asarray(matching.decode_batch(detector_events), dtype=np.uint8)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    return prediction


def generate_circuit_syndromes(
    t1_us: np.ndarray,
    simulator_time_ms: np.ndarray,
    layout: StimLayout,
    config: dict,
    *,
    t2_us: np.ndarray | None = None,
    frequency_shift_ghz: np.ndarray | None = None,
    event_onset_ms: np.ndarray | None = None,
    baseline_t1_us: np.ndarray | None = None,
    baseline_t2_us: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Create circuit-level detector samples from event-wise QP-derived T1."""
    validate_stim_config(config)
    t1 = np.asarray(t1_us, dtype=np.float32)
    if t1.ndim != 3 or t1.shape[2] != len(layout.qubit_ids):
        raise ValueError("t1_us and the active Stim layout qubits disagree")
    t2 = None if t2_us is None else np.asarray(t2_us, dtype=np.float32)
    if t2 is not None and t2.shape != t1.shape:
        raise ValueError("t2_us must have the same shape as t1_us")
    frequency_shift = (
        None
        if frequency_shift_ghz is None
        else np.asarray(frequency_shift_ghz, dtype=np.float32)
    )
    if frequency_shift is not None and frequency_shift.shape != t1.shape:
        raise ValueError("frequency_shift_ghz must have the same shape as t1_us")
    if config.get("radiation_channel", {}).get("targets", "all") == "data":
        # T1/T2 fields include ordinary hardware decay. Restrict only the excess
        # radiation response, not all relaxation, to data qubits.
        def baseline(values: np.ndarray, supplied: np.ndarray | None, name: str) -> np.ndarray:
            if supplied is None:
                onsets = np.asarray(event_onset_ms, dtype=float)
                if onsets.shape != (len(values),) or not np.isfinite(onsets).all():
                    raise ValueError(f"data-only radiation requires {name} or pre-event samples")
                rows = []
                for event, onset in enumerate(onsets):
                    before = np.asarray(simulator_time_ms) <= onset
                    if not before.any():
                        raise ValueError(f"no pre-event samples; provide {name}")
                    rows.append(np.median(values[event, before], axis=0))
                supplied = np.asarray(rows)
            reference = np.asarray(supplied, dtype=np.float32)
            if reference.shape != (values.shape[0], values.shape[2]):
                raise ValueError(f"{name} must have shape (event, qubit)")
            if not np.isfinite(reference).all() or np.any(reference <= 0):
                raise ValueError(f"{name} must be finite and positive")
            return reference

        ancilla = layout.roles != ROLE_DATA
        reference_t1 = baseline(t1, baseline_t1_us, "baseline_t1_us")
        t1 = t1.copy()
        t1[:, :, ancilla] = reference_t1[:, None, ancilla]
        if t2 is not None:
            reference_t2 = baseline(t2, baseline_t2_us, "baseline_t2_us")
            t2 = t2.copy()
            t2[:, :, ancilla] = reference_t2[:, None, ancilla]
        if frequency_shift is not None:
            frequency_shift = frequency_shift.copy()
            frequency_shift[:, :, ancilla] = 0
    rounds = int(config["rounds"])
    round_duration_ms = float(config["round_duration_ms"])
    round_start_time_ms = (
        float(config["start_time_ms"]) + np.arange(rounds) * round_duration_ms
    )
    round_time_ms = round_start_time_ms + 0.5 * round_duration_ms
    round_t1_us = average_t1_over_rounds(
        t1,
        simulator_time_ms,
        round_start_time_ms,
        round_duration_ms,
    )
    round_t2_us = (
        None
        if t2 is None
        else average_t1_over_rounds(
            t2,
            simulator_time_ms,
            round_start_time_ms,
            round_duration_ms,
        )
    )
    round_frequency_shift_ghz = None
    if frequency_shift is not None:
        round_frequency_shift_ghz = np.empty(
            (frequency_shift.shape[0], rounds, frequency_shift.shape[2]),
            dtype=np.float32,
        )
        source_time = np.asarray(simulator_time_ms, dtype=float)
        for event in range(frequency_shift.shape[0]):
            for qubit in range(frequency_shift.shape[2]):
                round_frequency_shift_ghz[event, :, qubit] = np.interp(
                    round_time_ms,
                    source_time,
                    frequency_shift[event, :, qubit],
                )

    radiation = config.get("radiation_channel", {})
    round_pauli = t1_to_pauli_probabilities(
        round_t1_us,
        round_duration_ms,
        t2_us=round_t2_us,
        pure_dephasing_time_us=radiation.get("pure_dephasing_time_us"),
    )
    injection_timing = radiation.get("timing", "gate_slices")
    tick_counts = round_tick_counts(layout.circuit, rounds)
    slice_duration_ms = gate_slice_durations_ms(
        round_duration_ms,
        tick_counts,
        config.get("gate_schedule", {}),
    )
    phase_config = radiation.get("coherent_phase_approximation", {})
    phase_approximation_enabled = bool(phase_config.get("enabled", False))
    scheduled_pauli = None
    if injection_timing == "gate_slices":
        injection_pauli = gate_slice_pauli_probabilities(
            round_t1_us,
            round_duration_ms,
            tick_counts,
            round_t2_us=round_t2_us,
            pure_dephasing_time_us=radiation.get("pure_dephasing_time_us"),
        )
        scheduled_pauli = scheduled_gate_slice_pauli_probabilities(
            round_t1_us,
            slice_duration_ms,
            tick_counts,
            round_t2_us=round_t2_us,
            round_frequency_shift_ghz=round_frequency_shift_ghz,
            pure_dephasing_time_us=radiation.get("pure_dephasing_time_us"),
            include_coherent_phase_approximation=phase_approximation_enabled,
        )
    else:
        injection_pauli = round_pauli

    events = t1.shape[0]
    shots = int(config.get("shots_per_event", 1))
    detector_events = np.empty(
        (events, shots, layout.circuit.num_detectors), dtype=np.uint8
    )
    measurement_records = np.empty(
        (events, shots, layout.circuit.num_measurements), dtype=np.uint8
    )
    logical_flips = np.empty(
        (events, shots, layout.circuit.num_observables), dtype=np.uint8
    )
    decoder_config = config.get("decoder", {})
    decoder_enabled = bool(decoder_config.get("enabled", False))
    decoder_mode = decoder_config.get("mode", "fixed_circuit")
    decoder_predictions = np.empty_like(logical_flips) if decoder_enabled else None
    fixed_matching = None
    fixed_decoder_circuit = None
    fixed_decoder_reference_used = None
    if decoder_enabled and decoder_mode == "fixed_circuit":
        reference_mode = decoder_config.get("baseline_reference", "pre_event_median")
        baseline_samples_t1 = []
        baseline_samples_t2 = []
        if reference_mode == "pre_event_median" and event_onset_ms is not None:
            onsets = np.asarray(event_onset_ms, dtype=float)
            if onsets.shape != (events,):
                raise ValueError("event_onset_ms must contain one value per event")
            source_time = np.asarray(simulator_time_ms, dtype=float)
            for event, onset in enumerate(onsets):
                before = source_time < onset
                if np.any(before):
                    baseline_samples_t1.append(t1[event, before, :])
                    if t2 is not None:
                        baseline_samples_t2.append(t2[event, before, :])
        if baseline_samples_t1:
            baseline_t1 = np.median(np.concatenate(baseline_samples_t1, axis=0), axis=0)
            baseline_t2 = (
                None
                if t2 is None
                else np.median(np.concatenate(baseline_samples_t2, axis=0), axis=0)
            )
        else:
            baseline_t1 = np.max(t1, axis=(0, 1))
            baseline_t2 = None if t2 is None else np.max(t2, axis=(0, 1))
            reference_mode = "maximum_t1"
        fixed_decoder_reference_used = reference_mode
        decoder_t1 = np.broadcast_to(
            baseline_t1[None, None, :], (1, rounds, len(layout.qubit_ids))
        ).copy()
        decoder_t2 = (
            None
            if baseline_t2 is None
            else np.broadcast_to(
                baseline_t2[None, None, :], decoder_t1.shape
            ).copy()
        )
        if injection_timing == "gate_slices":
            decoder_scheduled = scheduled_gate_slice_pauli_probabilities(
                decoder_t1,
                slice_duration_ms,
                tick_counts,
                round_t2_us=decoder_t2,
                pure_dephasing_time_us=radiation.get("pure_dephasing_time_us"),
            )
            fixed_decoder_circuit = inject_scheduled_gate_slice_pauli_channels(
                layout.circuit,
                layout.qubit_ids,
                decoder_scheduled["x"][0],
                decoder_scheduled["y"][0],
                decoder_scheduled["z"][0],
                tick_counts,
            )
        else:
            decoder_pauli = t1_to_pauli_probabilities(
                decoder_t1,
                round_duration_ms,
                t2_us=decoder_t2,
                pure_dephasing_time_us=radiation.get("pure_dephasing_time_us"),
            )
            fixed_decoder_circuit = inject_round_pauli_channels(
                layout.circuit,
                layout.qubit_ids,
                decoder_pauli["x"][0],
                decoder_pauli["y"][0],
                decoder_pauli["z"][0],
            )
        fixed_matching = _build_matching(fixed_decoder_circuit)
    root_seed = int(config.get("seed", 0) if seed is None else seed)
    first_annotated_circuit = None
    for event in range(events):
        if injection_timing == "gate_slices":
            assert scheduled_pauli is not None
            annotated = inject_scheduled_gate_slice_pauli_channels(
                layout.circuit,
                layout.qubit_ids,
                scheduled_pauli["x"][event],
                scheduled_pauli["y"][event],
                scheduled_pauli["z"][event],
                tick_counts,
            )
        else:
            annotated = inject_round_pauli_channels(
                layout.circuit,
                layout.qubit_ids,
                injection_pauli["x"][event],
                injection_pauli["y"][event],
                injection_pauli["z"][event],
            )
        if first_annotated_circuit is None:
            first_annotated_circuit = annotated
        event_seed = int(
            np.random.SeedSequence([root_seed, event]).generate_state(1, dtype=np.uint64)[0]
            % np.uint64(2**63 - 1)
        )
        sampled_measurements = annotated.compile_sampler(seed=event_seed).sample(
            shots=shots
        )
        sampled_detectors, sampled_observables = (
            annotated.compile_m2d_converter().convert(
                measurements=sampled_measurements,
                separate_observables=True,
            )
        )
        measurement_records[event] = sampled_measurements.astype(np.uint8)
        detector_events[event] = sampled_detectors.astype(np.uint8)
        logical_flips[event] = sampled_observables.astype(np.uint8)
        if decoder_predictions is not None:
            matching = (
                fixed_matching
                if fixed_matching is not None
                else _build_matching(annotated)
            )
            decoder_predictions[event] = _decode(matching, detector_events[event])

    result = {
        "measurement_records": measurement_records,
        "detector_events": detector_events,
        "logical_observable_flips": logical_flips,
        "round_start_time_ms": round_start_time_ms.astype(np.float64),
        "round_time_ms": round_time_ms.astype(np.float64),
        "round_t1_us": round_t1_us,
        "pauli_i_probability": round_pauli["i"],
        "pauli_x_probability": round_pauli["x"],
        "pauli_y_probability": round_pauli["y"],
        "pauli_z_probability": round_pauli["z"],
        "t2_us": round_pauli["t2_us"],
        "injection_pauli_i_probability": injection_pauli["i"],
        "injection_pauli_x_probability": injection_pauli["x"],
        "injection_pauli_y_probability": injection_pauli["y"],
        "injection_pauli_z_probability": injection_pauli["z"],
        "injection_tick_count": tick_counts,
        "gate_slice_duration_ms": slice_duration_ms,
        "circuit_qubit_ids": layout.qubit_ids,
        "circuit_grid_coords": layout.grid_coords.astype(np.float32),
        "circuit_physical_coords_mm": layout.physical_coords_mm.astype(np.float32),
        "circuit_qubit_roles": layout.roles,
        "detector_coords": layout.detector_coords,
    }
    if round_t2_us is not None:
        result["round_t2_us"] = round_t2_us
    if round_frequency_shift_ghz is not None:
        result["round_frequency_shift_ghz"] = round_frequency_shift_ghz
    if scheduled_pauli is not None:
        for output_name, source_name in (
            ("gate_slice_pauli_i_probability", "i"),
            ("gate_slice_pauli_x_probability", "x"),
            ("gate_slice_pauli_y_probability", "y"),
            ("gate_slice_pauli_z_probability", "z"),
            ("gate_slice_phase_z_probability", "phase_z_probability"),
        ):
            result[output_name] = scheduled_pauli[source_name]
    if decoder_predictions is not None:
        result["decoder_predictions"] = decoder_predictions
        result["decoder_failures"] = np.any(
            decoder_predictions != logical_flips, axis=2
        ).astype(np.uint8)
    metadata = {
        "schema_version": 2,
        "backend": "stim_surface_code",
        "task": config.get("task", "surface_code:rotated_memory_z"),
        "distance": int(config["distance"]),
        "rounds": rounds,
        "round_duration_ms": round_duration_ms,
        "start_time_ms": float(config["start_time_ms"]),
        "window_role": config.get("window_role", "unspecified"),
        "events": events,
        "shots_per_event": shots,
        "active_qubits": len(layout.qubit_ids),
        "data_qubits": int(np.count_nonzero(layout.roles == ROLE_DATA)),
        "x_ancillas": int(np.count_nonzero(layout.roles == ROLE_X_ANCILLA)),
        "z_ancillas": int(np.count_nonzero(layout.roles == ROLE_Z_ANCILLA)),
        "detectors": int(layout.circuit.num_detectors),
        "measurements": int(layout.circuit.num_measurements),
        "logical_observables": int(layout.circuit.num_observables),
        "role_codes": {str(key): value for key, value in ROLE_NAMES.items()},
        "radiation_targets": radiation.get("targets", "all"),
        "radiation_target_semantics": "restrict excess radiation; retain baseline T1/T2 on every qubit",
        "noise_accounting": {
            "qp_channel_includes_baseline_t1_t2": True,
            "depolarizing_channels_exclude_t1_t2": bool(
                config.get("noise_accounting", {}).get(
                    "depolarizing_channels_exclude_t1_t2", False
                )
            ),
            "measurement_and_reset_faults_are_separate": True,
        },
        "relaxation_channel_model": "pauli_twirled_t1_t2_ptgad",
        "channel_approximation_validation": {
            "reference": "qec_channel_validation/REPORT_JA.md",
            "exact_reference_model": "nonunital generalized amplitude damping plus dephasing",
            "minimum_round_t1_us": float(np.min(round_t1_us)),
            "median_round_t1_us": float(np.median(round_t1_us)),
            "fraction_round_qubits_below_50us": float(np.mean(round_t1_us < 50.0)),
            "fraction_round_qubits_below_30us": float(np.mean(round_t1_us < 30.0)),
            "strong_burst_warning": bool(np.any(round_t1_us < 50.0)),
            "interpretation": (
                "PTGAD is a scalable stress-test approximation; exact validation "
                "shows state-dependent detector differences during strong T1 bursts."
            ),
        },
        "radiation_injection_timing": injection_timing,
        "gate_schedule_mode": config.get("gate_schedule", {}).get("mode", "uniform"),
        "gate_slice_channel_placement": "after_moment_before_next_tick_or_measurement",
        "gate_slice_durations_ns": (
            slice_duration_ms[0, : tick_counts[0]] * 1e6
        ).tolist(),
        "coherent_phase_approximation_enabled": phase_approximation_enabled,
        "coherent_phase_approximation_method": (
            phase_config.get("method", "pauli_twirl")
            if phase_approximation_enabled
            else None
        ),
        "injection_tick_counts": tick_counts.tolist(),
        "decoder_enabled": decoder_enabled,
        "decoder_mode": decoder_mode if decoder_enabled else None,
        "decoder_baseline_reference": (
            fixed_decoder_reference_used
            if fixed_matching is not None
            else None
        ),
        "fixed_decoder_includes_baseline_t1_t2": bool(fixed_matching is not None),
        "decoder_is_oracle": bool(decoder_enabled and decoder_mode == "oracle_event"),
        "seed": root_seed,
        "model_scope": [
            "The inverse T1 rate is integrated over each QEC round.",
            (
                "The round channel is split across TICK intervals and placed after each interval's operations."
                if injection_timing == "gate_slices"
                else "The Pauli-twirled channel is inserted at the round boundary."
            ),
            "Stim executes resets, Clifford gates, measurements, detectors, and logical observables.",
            "The layout is generic and is not restricted to the 26-qubit Google observation set.",
        ],
        "known_limitations": [
            "T1/T2 are round-averaged; gate durations control sub-round channel placement.",
            (
                "Frequency shifts use an explicitly requested per-slice Pauli twirl, not exact coherent evolution."
                if phase_approximation_enabled
                else "Coherent frequency shifts are not injected unless explicitly requested."
            ),
            "Non-unital amplitude-damping asymmetry is not represented; quantified limits are saved in channel_approximation_validation.",
            "Microscopic phonon transport is not simulated.",
        ],
    }
    assert first_annotated_circuit is not None
    metadata["first_annotated_circuit"] = str(first_annotated_circuit)
    return result, metadata


def save_circuit_syndrome_output(
    output: Path,
    result: dict[str, np.ndarray],
    metadata: dict,
    *,
    template_circuit: Any,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "stim_syndrome_events.npz", **result)
    serializable_metadata = dict(metadata)
    circuit_text = serializable_metadata.pop("first_annotated_circuit")
    (output / "stim_template_circuit.stim").write_text(str(template_circuit) + "\n")
    (output / "stim_first_event_circuit.stim").write_text(circuit_text + "\n")
    (output / "stim_syndrome_metadata.json").write_text(
        json.dumps(serializable_metadata, indent=2) + "\n"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def collect_run_manifest(
    repository: Path,
    *,
    argv: list[str],
    source_paths: list[Path],
    simulator_config: dict,
    stim_config: dict,
    template_circuit: Any,
) -> dict:
    """Collect enough immutable identifiers to audit a generated dataset."""
    repository = repository.resolve()

    def git_output(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git_output("status", "--porcelain=v1")
    source_hashes = {}
    for path in source_paths:
        resolved = path.resolve()
        try:
            name = str(resolved.relative_to(repository))
        except ValueError:
            name = str(resolved)
        source_hashes[name] = _sha256_bytes(resolved.read_bytes())

    packages = {}
    for distribution in ("numpy", "pandas", "scipy", "stim", "pymatching"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    circuit_text = str(template_circuit) + "\n"
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(argv),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": [] if not status else status.splitlines(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "source_sha256": source_hashes,
        "resolved_simulator_config_sha256": _canonical_json_sha256(simulator_config),
        "resolved_stim_config_sha256": _canonical_json_sha256(stim_config),
        "template_circuit_sha256": _sha256_bytes(circuit_text.encode()),
        "seeds": {
            "simulation": int(simulator_config["seed"]),
            "syndrome": int(stim_config.get("seed", 0)),
        },
    }


def uses_peak_matched_qp(config: dict) -> bool:
    """Return whether event strength is post-hoc matched to a sampled peak."""
    return bool(config.get("temporal_population", {}).get("enabled", False))


def uses_fixed_hardware(config: dict) -> bool:
    """Return whether one immutable device calibration is shared by all events.

    QP-induced T1/T2 changes are still event dependent.  This predicate concerns
    only the no-radiation device/qubit calibration that feeds the QP response.
    """
    hardware = config.get("hardware_hierarchy", {})
    if not hardware.get("enabled", False) or hardware.get("mode") != "fixed_device":
        return False
    drift_keys = (
        "run_t1_log_sd",
        "run_t2_log_sd",
        "run_frequency_sd_ghz",
        "event_t1_log_sd",
        "event_t2_log_sd",
        "event_frequency_sd_ghz",
    )
    return all(float(hardware.get(key, 0.0)) == 0.0 for key in drift_keys)


def main() -> None:
    from .configuration import config_path

    parser = argparse.ArgumentParser(
        description="Generate natural circuit-level QEC syndrome data from the QP ODE model"
    )
    parser.add_argument("--config", default=config_path("simulator_base"))
    parser.add_argument("--profile", default=config_path("qp_ode_generic"))
    parser.add_argument("--stim-config", default=config_path("stim_surface_code_d3"))
    parser.add_argument("--output", default="simulation_stim_surface_code")
    parser.add_argument("--n-events", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--syndrome-seed", type=int)
    parser.add_argument(
        "--allow-peak-matched-qp",
        action="store_true",
        help=(
            "allow a profile that adjusts QP generation to a sampled target peak; "
            "such runs are phenomenological and cannot validate absolute QP/T1 scale"
        ),
    )
    parser.add_argument(
        "--allow-varying-hardware",
        action="store_true",
        help=(
            "allow device_population mode or run/event T1/T2/frequency drift; "
            "the default requires one immutable device calibration across events"
        ),
    )
    args = parser.parse_args()

    stim_config = json.loads(Path(args.stim_config).read_text())
    if args.syndrome_seed is not None:
        stim_config["seed"] = args.syndrome_seed
    layout = build_stim_layout(stim_config)
    simulator_config = load_config(Path(args.config), Path(args.profile))
    if args.n_events is not None:
        simulator_config["n_events"] = args.n_events
    if args.seed is not None:
        simulator_config["seed"] = args.seed
    if simulator_config.get("temporal_model", {}).get("backend") != "qp_ode":
        raise ValueError("circuit-level T1 conversion requires a qp_ode simulator profile")
    peak_matched_qp = uses_peak_matched_qp(simulator_config)
    if peak_matched_qp and not args.allow_peak_matched_qp:
        raise ValueError(
            "this profile post-hoc matches QP generation to a sampled target peak; "
            "use a non-peak-matched profile for circuit prediction, or pass "
            "--allow-peak-matched-qp for an explicitly phenomenological stress test"
        )
    fixed_hardware = uses_fixed_hardware(simulator_config)
    if not fixed_hardware and not args.allow_varying_hardware:
        raise ValueError(
            "circuit dataset generation requires hardware_hierarchy.enabled=true, "
            "mode='fixed_device', and zero run/event hardware drift; use "
            "--allow-varying-hardware only for an explicitly labeled hardware-"
            "population or drift experiment"
        )
    simulator_config.setdefault("output", {})["save_probabilities"] = True
    simulator_config["output"]["save_physics_diagnostics"] = True

    run_manifest = collect_run_manifest(
        Path(__file__).resolve().parent,
        argv=sys.argv,
        source_paths=[
            Path(__file__),
            Path(__file__).resolve().parent / "simulator.py",
            Path(__file__).resolve().parent / "syndrome.py",
            Path(args.config),
            Path(args.profile),
            Path(args.stim_config),
        ],
        simulator_config=simulator_config,
        stim_config=stim_config,
        template_circuit=layout.circuit,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (
        observations,
        probabilities,
        time_ms,
        parameters,
        baselines,
        baseline_probabilities,
        physics_arrays,
    ) = simulate(
        simulator_config,
        coords=layout.physical_coords_mm,
        include_baseline_probabilities=True,
        include_physics_arrays=True,
    )
    latent_arrays = {
        "observations": observations,
        "probabilities": probabilities,
        "baseline_probabilities": baseline_probabilities,
        "time_ms": time_ms,
        "coords": layout.physical_coords_mm,
        "baselines": baselines,
        "circuit_qubit_ids": layout.qubit_ids,
        "circuit_qubit_roles": layout.roles,
        "event_onset_ms": parameters["event_onset_ms"].to_numpy(dtype=np.float64),
        "event_is_control": parameters["is_control"].to_numpy(dtype=np.uint8),
        **physics_arrays,
    }
    np.savez_compressed(output / "simulated_events.npz", **latent_arrays)
    parameters.to_csv(output / "true_parameters.csv", index=False)
    (output / "resolved_config.json").write_text(
        json.dumps(simulator_config, indent=2) + "\n"
    )
    (output / "resolved_stim_config.json").write_text(
        json.dumps(stim_config, indent=2) + "\n"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )

    result, metadata = generate_circuit_syndromes(
        physics_arrays["t1_us"],
        time_ms,
        layout,
        stim_config,
        t2_us=physics_arrays.get("t2_us"),
        frequency_shift_ghz=physics_arrays.get("frequency_shift_ghz"),
        event_onset_ms=parameters["event_onset_ms"].to_numpy(dtype=np.float64),
    )
    result["event_onset_ms"] = parameters["event_onset_ms"].to_numpy(dtype=np.float64)
    result["event_is_control"] = parameters["is_control"].to_numpy(dtype=np.uint8)
    metadata["control_events"] = int(parameters["is_control"].sum())
    metadata["qp_peak_matched"] = peak_matched_qp
    metadata["hardware_fixed_across_events"] = fixed_hardware
    metadata["qp_absolute_hardware_calibrated"] = bool(
        simulator_config.get("qp_model_provenance", {}).get(
            "absolute_hardware_calibration", False
        )
    )
    metadata["qp_generation_semantics"] = (
        "qp_generation_rate_per_us is a local density-like proxy sampled at each "
        "qubit coordinate; its sum over qubits is not deposited particle energy"
    )
    metadata["particle_energy_conservation_modeled"] = False
    metadata["source_simulation"] = str(output / "simulated_events.npz")
    metadata["run_manifest"] = str(output / "run_manifest.json")
    save_circuit_syndrome_output(
        output, result, metadata, template_circuit=layout.circuit
    )
    print(
        f"generated {len(parameters)} events "
        f"({int(parameters['is_control'].sum())} controls) on "
        f"{len(layout.qubit_ids)} active circuit qubits"
    )
    print(
        f"Stim output: {stim_config['rounds']} rounds, "
        f"{stim_config.get('shots_per_event', 1)} shot(s) per event, "
        f"{layout.circuit.num_detectors} detectors"
    )
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
