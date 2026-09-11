#!/usr/bin/env python3
"""Deterministic validation of the Stim PTGAD approximation.

The production syndrome path converts T1/T2 into a Pauli-twirled generalized
amplitude-damping (PTGAD) channel so that Stim can sample large stabilizer
circuits efficiently.  This module measures what that approximation discards.

It uses a three-qubit repeated parity-check primitive (two data qubits and one
reset/measured ancilla) and propagates its density matrix exactly.  The same
gate schedule is run with either:

* a non-unital generalized amplitude-damping plus dephasing Kraus channel; or
* the unital PTGAD Pauli channel used by ``stim_qec.py``.

All measurement branches are enumerated, so the comparison has no shot noise.
This is a channel-approximation validation, not a replacement for a future
full-device comparison against measured syndrome data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .syndrome import t1_to_pauli_probabilities


I2 = np.eye(2, dtype=np.complex128)
X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
H = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

STATE_VECTORS = {
    "0": np.asarray([1, 0], dtype=np.complex128),
    "1": np.asarray([0, 1], dtype=np.complex128),
    "+": np.asarray([1, 1], dtype=np.complex128) / np.sqrt(2),
    "-": np.asarray([1, -1], dtype=np.complex128) / np.sqrt(2),
    "+i": np.asarray([1, 1j], dtype=np.complex128) / np.sqrt(2),
    "-i": np.asarray([1, -1j], dtype=np.complex128) / np.sqrt(2),
}

CASE_DEFINITIONS = {
    "z_memory_00": {"basis": "z", "states": ("0", "0", "0"), "bad": "1"},
    "z_memory_11": {"basis": "z", "states": ("1", "1", "0"), "bad": "0"},
    "x_memory_pp": {"basis": "x", "states": ("+", "+", "0"), "bad": "-"},
    "x_memory_mm": {"basis": "x", "states": ("-", "-", "0"), "bad": "+"},
}


def _validate_channel_inputs(
    t1_us: float,
    t2_us: float,
    duration_us: float,
    equilibrium_excited_population: float,
) -> tuple[float, float, float, float]:
    values = np.asarray(
        [t1_us, t2_us, duration_us, equilibrium_excited_population], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("channel parameters must be finite")
    t1, t2, duration, p_excited = map(float, values)
    if t1 <= 0 or t2 <= 0 or duration < 0:
        raise ValueError("T1/T2 must be positive and duration must be non-negative")
    if t2 > 2.0 * t1 * (1 + 1e-10):
        raise ValueError("T2 must satisfy T2 <= 2*T1")
    if not 0 <= p_excited <= 1:
        raise ValueError("equilibrium_excited_population must lie in [0, 1]")
    return t1, t2, duration, p_excited


def exact_gad_dephasing_kraus(
    t1_us: float,
    t2_us: float,
    duration_us: float,
    equilibrium_excited_population: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Return exact generalized-amplitude-damping plus dephasing Kraus operators."""
    t1, t2, duration, p_excited = _validate_channel_inputs(
        t1_us, t2_us, duration_us, equilibrium_excited_population
    )
    gamma = -np.expm1(-duration / t1)
    survival = np.sqrt(max(1.0 - gamma, 0.0))
    p_ground = 1.0 - p_excited
    amplitude = (
        np.sqrt(p_ground)
        * np.asarray([[1, 0], [0, survival]], dtype=np.complex128),
        np.sqrt(p_ground)
        * np.asarray([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128),
        np.sqrt(p_excited)
        * np.asarray([[survival, 0], [0, 1]], dtype=np.complex128),
        np.sqrt(p_excited)
        * np.asarray([[0, 0], [np.sqrt(gamma), 0]], dtype=np.complex128),
    )
    gamma_phi = max(1.0 / t2 - 0.5 / t1, 0.0)
    phase_flip_probability = -0.5 * np.expm1(-duration * gamma_phi)
    phase = (
        np.sqrt(1.0 - phase_flip_probability) * I2,
        np.sqrt(phase_flip_probability) * Z,
    )
    return tuple(d @ a for d in phase for a in amplitude if np.any(a))


def ptgad_kraus(
    t1_us: float,
    t2_us: float,
    duration_us: float,
) -> tuple[np.ndarray, ...]:
    """Return the Pauli Kraus representation used by the production Stim path."""
    _validate_channel_inputs(t1_us, t2_us, duration_us, 0.0)
    channel = t1_to_pauli_probabilities(
        np.asarray([[[t1_us]]], dtype=np.float32),
        duration_us / 1000.0,
        t2_us=np.asarray([[[t2_us]]], dtype=np.float32),
    )
    probabilities = [
        float(channel[name][0, 0, 0]) for name in ("i", "x", "y", "z")
    ]
    return tuple(
        np.sqrt(probability) * operator
        for probability, operator in zip(probabilities, (I2, X, Y, Z))
        if probability > 0
    )


def _tensor_product(operators: Iterable[np.ndarray]) -> np.ndarray:
    result = np.asarray([[1.0]], dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def _expand_single_qubit(operator: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    if not 0 <= qubit < n_qubits:
        raise ValueError("qubit index is out of range")
    return _tensor_product(
        operator if index == qubit else I2 for index in range(n_qubits)
    )


def _cnot(n_qubits: int, control: int, target: int) -> np.ndarray:
    if control == target:
        raise ValueError("CNOT control and target must differ")
    dimension = 1 << n_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    for column in range(dimension):
        row = column ^ target_mask if column & control_mask else column
        matrix[row, column] = 1.0
    return matrix


def _apply_unitary(
    branches: dict[tuple[int, ...], np.ndarray], unitary: np.ndarray
) -> dict[tuple[int, ...], np.ndarray]:
    adjoint = unitary.conj().T
    return {record: unitary @ rho @ adjoint for record, rho in branches.items()}


def _apply_local_noise(
    branches: dict[tuple[int, ...], np.ndarray],
    kraus: tuple[np.ndarray, ...],
    n_qubits: int,
) -> dict[tuple[int, ...], np.ndarray]:
    expanded_by_qubit = [
        tuple(_expand_single_qubit(operator, qubit, n_qubits) for operator in kraus)
        for qubit in range(n_qubits)
    ]
    result = branches
    for expanded in expanded_by_qubit:
        next_result: dict[tuple[int, ...], np.ndarray] = {}
        for record, rho in result.items():
            transformed = np.zeros_like(rho)
            for operator in expanded:
                transformed += operator @ rho @ operator.conj().T
            next_result[record] = transformed
        result = next_result
    return result


def _measure_and_reset(
    branches: dict[tuple[int, ...], np.ndarray], ancilla: int, n_qubits: int
) -> dict[tuple[int, ...], np.ndarray]:
    projectors = (
        _expand_single_qubit(np.diag([1.0, 0.0]), ancilla, n_qubits),
        _expand_single_qubit(np.diag([0.0, 1.0]), ancilla, n_qubits),
    )
    reset_x = _expand_single_qubit(X, ancilla, n_qubits)
    result: dict[tuple[int, ...], np.ndarray] = {}
    for record, rho in branches.items():
        for outcome, projector in enumerate(projectors):
            projected = projector @ rho @ projector
            if float(np.trace(projected).real) <= 1e-15:
                continue
            if outcome:
                projected = reset_x @ projected @ reset_x
            result[record + (outcome,)] = projected
    return result


def _product_density(states: tuple[str, ...]) -> np.ndarray:
    vector = _tensor_product(STATE_VECTORS[state][:, None] for state in states)[:, 0]
    return np.outer(vector, vector.conj())


def _channel_for_model(
    model: str,
    t1_us: float,
    t2_us: float,
    duration_us: float,
    equilibrium_excited_population: float,
) -> tuple[np.ndarray, ...]:
    if model == "exact_gad":
        return exact_gad_dephasing_kraus(
            t1_us,
            t2_us,
            duration_us,
            equilibrium_excited_population,
        )
    if model == "ptgad":
        return ptgad_kraus(t1_us, t2_us, duration_us)
    raise ValueError(f"unknown channel model: {model}")


def _run_parity_round(
    branches: dict[tuple[int, ...], np.ndarray],
    *,
    basis: str,
    model: str,
    t1_us: float,
    t2_us: float,
    equilibrium_excited_population: float,
    cycle_duration_us: float,
) -> dict[tuple[int, ...], np.ndarray]:
    n_qubits = 3
    ancilla = 2

    def noise(duration_us: float) -> None:
        nonlocal branches
        branches = _apply_local_noise(
            branches,
            _channel_for_model(
                model,
                t1_us,
                t2_us,
                duration_us,
                equilibrium_excited_population,
            ),
            n_qubits,
        )

    if basis == "z":
        # Preparation/idle, two data-to-ancilla CNOTs, then readout/idle.
        durations = np.asarray([0.03, 0.04, 0.04], dtype=float)
        readout = cycle_duration_us - float(durations.sum())
        if readout <= 0:
            raise ValueError("cycle_duration_us is too short for the Z-check schedule")
        noise(float(durations[0]))
        branches = _apply_unitary(branches, _cnot(n_qubits, 0, ancilla))
        noise(float(durations[1]))
        branches = _apply_unitary(branches, _cnot(n_qubits, 1, ancilla))
        noise(float(durations[2]))
        noise(readout)
    elif basis == "x":
        # Ancilla H, two ancilla-to-data CNOTs, H, then readout/idle.
        durations = np.asarray([0.03, 0.04, 0.04, 0.03], dtype=float)
        readout = cycle_duration_us - float(durations.sum())
        if readout <= 0:
            raise ValueError("cycle_duration_us is too short for the X-check schedule")
        branches = _apply_unitary(branches, _expand_single_qubit(H, ancilla, n_qubits))
        noise(float(durations[0]))
        branches = _apply_unitary(branches, _cnot(n_qubits, ancilla, 0))
        noise(float(durations[1]))
        branches = _apply_unitary(branches, _cnot(n_qubits, ancilla, 1))
        noise(float(durations[2]))
        branches = _apply_unitary(branches, _expand_single_qubit(H, ancilla, n_qubits))
        noise(float(durations[3]))
        noise(readout)
    else:
        raise ValueError("basis must be 'x' or 'z'")
    return _measure_and_reset(branches, ancilla, n_qubits)


def _detectors(record: tuple[int, ...]) -> tuple[int, ...]:
    if not record:
        return ()
    return (record[0],) + tuple(
        record[index] ^ record[index - 1] for index in range(1, len(record))
    )


def _measurement_distribution(
    branches: dict[tuple[int, ...], np.ndarray]
) -> dict[tuple[int, ...], float]:
    result: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for record, rho in branches.items():
        result[_detectors(record)] += max(float(np.trace(rho).real), 0.0)
    total = sum(result.values())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise RuntimeError(f"measurement branch probability sums to {total}, not 1")
    return {key: value / total for key, value in result.items()}


def _data_memory_error(
    branches: dict[tuple[int, ...], np.ndarray], bad_state: str
) -> float:
    bad = STATE_VECTORS[bad_state]
    projector = _expand_single_qubit(np.outer(bad, bad.conj()), 0, 3)
    return float(
        sum(np.trace(projector @ rho).real for rho in branches.values())
    )


def simulate_parity_memory(
    model: str,
    case: str,
    *,
    t1_us: float,
    t2_us: float,
    rounds: int,
    cycle_duration_us: float,
    equilibrium_excited_population: float,
) -> dict:
    """Run a deterministic repeated parity-check experiment."""
    if case not in CASE_DEFINITIONS:
        raise ValueError(f"unknown validation case: {case}")
    if not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    definition = CASE_DEFINITIONS[case]
    branches = {(): _product_density(definition["states"])}
    for _ in range(rounds):
        branches = _run_parity_round(
            branches,
            basis=definition["basis"],
            model=model,
            t1_us=t1_us,
            t2_us=t2_us,
            equilibrium_excited_population=equilibrium_excited_population,
            cycle_duration_us=cycle_duration_us,
        )
    distribution = _measurement_distribution(branches)
    bit_mean = np.zeros(rounds, dtype=float)
    second_moment = np.zeros((rounds, rounds), dtype=float)
    expected_weight = 0.0
    for detector_record, probability in distribution.items():
        bits = np.asarray(detector_record, dtype=float)
        bit_mean += probability * bits
        second_moment += probability * np.outer(bits, bits)
        expected_weight += probability * float(bits.sum())
    return {
        "distribution": distribution,
        "detector_probability": bit_mean,
        "detector_covariance": second_moment - np.outer(bit_mean, bit_mean),
        "expected_syndrome_weight": expected_weight,
        "data_memory_error_probability": _data_memory_error(
            branches, definition["bad"]
        ),
    }


def _apply_single_qubit_kraus(
    rho: np.ndarray, kraus: tuple[np.ndarray, ...]
) -> np.ndarray:
    return sum((operator @ rho @ operator.conj().T for operator in kraus), np.zeros_like(rho))


def single_qubit_channel_distance(
    t1_us: float,
    t2_us: float,
    duration_us: float,
    equilibrium_excited_population: float,
) -> dict[str, float]:
    """Compare exact and PTGAD outputs on the six cardinal pure states."""
    exact = exact_gad_dephasing_kraus(
        t1_us, t2_us, duration_us, equilibrium_excited_population
    )
    twirled = ptgad_kraus(t1_us, t2_us, duration_us)
    distances = {}
    for name, vector in STATE_VECTORS.items():
        rho = np.outer(vector, vector.conj())
        difference = _apply_single_qubit_kraus(rho, exact) - _apply_single_qubit_kraus(
            rho, twirled
        )
        distances[name] = 0.5 * float(np.ptp(np.linalg.eigvalsh(difference)))
    mixed = 0.5 * I2
    exact_mixed = _apply_single_qubit_kraus(mixed, exact)
    return {
        "maximum_cardinal_state_trace_distance": max(distances.values()),
        "mean_cardinal_state_trace_distance": float(np.mean(list(distances.values()))),
        "exact_nonunital_z_translation": float(np.trace(Z @ exact_mixed).real),
    }


def compare_models(
    case: str,
    *,
    t1_us: float,
    t2_us: float,
    rounds: int,
    cycle_duration_us: float,
    equilibrium_excited_population: float,
) -> dict[str, float | str]:
    exact = simulate_parity_memory(
        "exact_gad",
        case,
        t1_us=t1_us,
        t2_us=t2_us,
        rounds=rounds,
        cycle_duration_us=cycle_duration_us,
        equilibrium_excited_population=equilibrium_excited_population,
    )
    twirled = simulate_parity_memory(
        "ptgad",
        case,
        t1_us=t1_us,
        t2_us=t2_us,
        rounds=rounds,
        cycle_duration_us=cycle_duration_us,
        equilibrium_excited_population=equilibrium_excited_population,
    )
    support = set(exact["distribution"]) | set(twirled["distribution"])
    tv = 0.5 * sum(
        abs(exact["distribution"].get(key, 0.0) - twirled["distribution"].get(key, 0.0))
        for key in support
    )
    channel = single_qubit_channel_distance(
        t1_us,
        t2_us,
        cycle_duration_us,
        equilibrium_excited_population,
    )
    exact_detector = exact["detector_probability"]
    twirled_detector = twirled["detector_probability"]
    return {
        "case": case,
        "basis": CASE_DEFINITIONS[case]["basis"],
        "t1_us": float(t1_us),
        "t2_us": float(t2_us),
        "t2_over_t1": float(t2_us / t1_us),
        "equilibrium_excited_population": float(equilibrium_excited_population),
        "rounds": int(rounds),
        "cycle_duration_us": float(cycle_duration_us),
        **channel,
        "detector_distribution_tv": float(tv),
        "exact_mean_detector_probability": float(exact_detector.mean()),
        "ptgad_mean_detector_probability": float(twirled_detector.mean()),
        "maximum_detector_probability_abs_error": float(
            np.max(np.abs(exact_detector - twirled_detector))
        ),
        "exact_expected_syndrome_weight": float(exact["expected_syndrome_weight"]),
        "ptgad_expected_syndrome_weight": float(twirled["expected_syndrome_weight"]),
        "syndrome_weight_abs_error": float(
            abs(exact["expected_syndrome_weight"] - twirled["expected_syndrome_weight"])
        ),
        "detector_covariance_max_abs_error": float(
            np.max(
                np.abs(
                    exact["detector_covariance"] - twirled["detector_covariance"]
                )
            )
        ),
        "exact_data_memory_error_probability": float(
            exact["data_memory_error_probability"]
        ),
        "ptgad_data_memory_error_probability": float(
            twirled["data_memory_error_probability"]
        ),
        "data_memory_error_abs_error": float(
            abs(
                exact["data_memory_error_probability"]
                - twirled["data_memory_error_probability"]
            )
        ),
    }


def validate_config(config: dict) -> None:
    required = (
        "t1_us",
        "t2_over_t1",
        "equilibrium_excited_population",
        "rounds",
        "cycle_duration_us",
        "cases",
        "screening_thresholds",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("validation config is missing: " + ", ".join(missing))
    if any(float(value) <= 0 for value in config["t1_us"]):
        raise ValueError("all t1_us values must be positive")
    if any(not 0 < float(value) <= 2 for value in config["t2_over_t1"]):
        raise ValueError("t2_over_t1 values must lie in (0, 2]")
    if any(
        not 0 <= float(value) <= 1
        for value in config["equilibrium_excited_population"]
    ):
        raise ValueError("equilibrium populations must lie in [0, 1]")
    unknown_cases = set(config["cases"]) - set(CASE_DEFINITIONS)
    if unknown_cases:
        raise ValueError(f"unknown validation cases: {sorted(unknown_cases)}")


def run_validation(config: dict) -> tuple[pd.DataFrame, dict]:
    validate_config(config)
    rows = []
    for t1 in config["t1_us"]:
        for ratio in config["t2_over_t1"]:
            for population in config["equilibrium_excited_population"]:
                for case in config["cases"]:
                    rows.append(
                        compare_models(
                            case,
                            t1_us=float(t1),
                            t2_us=float(t1) * float(ratio),
                            rounds=int(config["rounds"]),
                            cycle_duration_us=float(config["cycle_duration_us"]),
                            equilibrium_excited_population=float(population),
                        )
                    )
    frame = pd.DataFrame(rows)
    thresholds = config["screening_thresholds"]
    mappings = {
        "channel_state": "maximum_cardinal_state_trace_distance",
        "detector_tv": "detector_distribution_tv",
        "detector_probability": "maximum_detector_probability_abs_error",
        "data_memory": "data_memory_error_abs_error",
    }
    screening_columns = []
    for threshold_name, metric in mappings.items():
        column = f"passes_{threshold_name}_screen"
        frame[column] = frame[metric] <= float(thresholds[threshold_name])
        screening_columns.append(column)
    frame["passes_all_screens"] = frame[screening_columns].all(axis=1)

    by_t1 = []
    for t1, group in frame.groupby("t1_us", sort=False):
        by_t1.append(
            {
                "t1_us": float(t1),
                "cases": int(len(group)),
                "all_cases_pass": bool(group["passes_all_screens"].all()),
                "pass_fraction": float(group["passes_all_screens"].mean()),
                "worst_channel_state_trace_distance": float(
                    group["maximum_cardinal_state_trace_distance"].max()
                ),
                "worst_detector_distribution_tv": float(
                    group["detector_distribution_tv"].max()
                ),
                "worst_detector_probability_abs_error": float(
                    group["maximum_detector_probability_abs_error"].max()
                ),
                "worst_data_memory_error_abs_error": float(
                    group["data_memory_error_abs_error"].max()
                ),
            }
        )
    summary = {
        "schema_version": 1,
        "comparison": "exact non-unital GAD+dephasing versus unital PTGAD",
        "circuit": "two-data/one-ancilla repeated X/Z parity-check primitive",
        "deterministic": True,
        "monte_carlo_shots": 0,
        "screening_thresholds": thresholds,
        "threshold_interpretation": (
            "engineering screens for this project, not universal physical "
            "accuracy tolerances"
        ),
        "rows": int(len(frame)),
        "overall_pass_fraction": float(frame["passes_all_screens"].mean()),
        "by_t1": by_t1,
        "known_limitations": [
            "The validation circuit is a three-qubit parity-check primitive, not a full surface code.",
            "It isolates the T1/T2 channel and does not include leakage, crosstalk, or control noise.",
            "A hardware claim still requires comparison with measured detector data.",
        ],
    }
    return frame, summary


def make_figure(frame: pd.DataFrame, output: Path, thresholds: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("maximum_cardinal_state_trace_distance", "Max state trace distance", "channel_state"),
        ("detector_distribution_tv", "Detector distribution TV", "detector_tv"),
        ("maximum_detector_probability_abs_error", "Max detector probability error", "detector_probability"),
        ("data_memory_error_abs_error", "Data-memory error difference", "data_memory"),
    )
    grouped = frame.groupby("t1_us", sort=True)
    t1 = np.asarray(sorted(frame["t1_us"].unique()), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, label, threshold_name) in zip(axes.flat, metrics):
        worst = grouped[metric].max().reindex(t1).to_numpy()
        median = grouped[metric].median().reindex(t1).to_numpy()
        axis.plot(t1, worst, "o-", linewidth=2.3, label="worst case")
        axis.plot(t1, median, "s--", linewidth=1.8, label="median case")
        axis.axhline(
            float(thresholds[threshold_name]),
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="project screen",
        )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("T1 (us; stronger burst to the right)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(
        "QEC channel validation: exact amplitude damping versus Stim PTGAD",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output / "qec_channel_validation.png", dpi=220)
    fig.savefig(output / "qec_channel_validation.pdf")
    plt.close(fig)


def write_report(summary: dict, output: Path) -> None:
    lines = [
        "# QEC channel検証結果",
        "",
        "同じT1/T2と1 us QEC cycleを使い、非unitalなexact generalized amplitude dampingと、",
        "Stim経路で使用するunitalなPTGADを3-qubit反復parity-checkで比較した。全measurement",
        "branchを密度行列で列挙しているため、結果にshot noiseはない。",
        "",
        "| T1 (us) | 全case通過 | 通過率 | worst detector TV | worst detector p error | worst memory error |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_t1"]:
        lines.append(
            "| {t1_us:g} | {passed} | {pass_fraction:.1%} | "
            "{worst_detector_distribution_tv:.4f} | "
            "{worst_detector_probability_abs_error:.4f} | "
            "{worst_data_memory_error_abs_error:.4f} |".format(
                passed="yes" if row["all_cases_pass"] else "no", **row
            )
        )
    lines.extend(
        [
            "",
            "判定thresholdはこのprojectのscreening基準であり、普遍的な物理許容値ではない。",
            "強いburstで失敗する場合、PTGADを実機忠実なchannelと呼ばず、高速stress-test近似として",
            "使用範囲を限定する。実機精度の最終確認にはmeasured detector dataが必要である。",
            "",
            "- `qec_channel_validation.csv`: 全parameter/caseの数値",
            "- `qec_channel_validation.png`: T1に対する近似誤差",
            "- `qec_channel_validation_summary.json`: 判定と制約",
        ]
    )
    (output / "REPORT_JA.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    from .configuration import config_path

    parser = argparse.ArgumentParser(
        description="Validate exact GAD against the Stim PTGAD approximation"
    )
    parser.add_argument("--config", default=config_path("channel_validation"))
    parser.add_argument("--output", default="qec_channel_validation")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, summary = run_validation(config)
    frame.to_csv(output / "qec_channel_validation.csv", index=False)
    (output / "qec_channel_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    make_figure(frame, output, config["screening_thresholds"])
    write_report(summary, output)
    print(f"validated {len(frame)} exact/PTGAD circuit cases")
    print(f"screening pass fraction: {summary['overall_pass_fraction']:.1%}")
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
