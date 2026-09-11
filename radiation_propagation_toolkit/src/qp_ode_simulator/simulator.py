#!/usr/bin/env python3
"""Core QP-ODE radiation-event simulator.

The output observations have shape (event, time, qubit), with 1 denoting an error.
This is deliberately a phenomenological model of the measured qubit response, not a
microscopic phonon transport calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, lambertw

from .layouts import DEFAULT_COORDS


ELECTRON_VOLT_J = 1.602176634e-19
HBAR_J_S = 1.054571817e-34


def _bounds(value) -> tuple[float, float]:
    """Return the numeric lower and upper bounds of a scalar/range setting."""
    values = np.asarray(value if isinstance(value, list) else [value], dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"configuration value must contain finite numbers: {value!r}")
    if isinstance(value, list) and len(value) == 2 and values[0] > values[1]:
        raise ValueError(f"configuration range must be ordered [low, high]: {value!r}")
    return float(values.min()), float(values.max())


def validate_config(config: dict, coords: np.ndarray = DEFAULT_COORDS) -> None:
    """Validate invariants that would otherwise fail deep inside the simulator."""
    coordinates = np.asarray(coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("coords must have shape (n_qubits, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("coords must contain only finite values")

    n_events = config.get("n_events")
    if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events <= 0:
        raise ValueError("n_events must be a positive integer")
    device_seed = config.get("device_seed")
    if device_seed is not None and (
        not isinstance(device_seed, int) or isinstance(device_seed, bool) or device_seed < 0
    ):
        raise ValueError("device_seed must be a non-negative integer")
    time = config.get("time", {})
    start = float(time.get("start_ms", np.nan))
    end = float(time.get("end_ms", np.nan))
    step = float(time.get("dt_ms", np.nan))
    if not np.isfinite([start, end, step]).all() or step <= 0 or end <= start:
        raise ValueError("time requires finite start_ms < end_ms and dt_ms > 0")

    event_timing = config.get("event_timing", {})
    if not isinstance(event_timing, dict):
        raise ValueError("event_timing must be an object")
    unknown_timing = set(event_timing) - {"onset_time_ms", "control_fraction"}
    if unknown_timing:
        raise ValueError(f"unknown event_timing keys: {sorted(unknown_timing)}")
    onset_lower, onset_upper = _bounds(event_timing.get("onset_time_ms", 0.0))
    if onset_lower < start or onset_upper >= end:
        raise ValueError("event_timing.onset_time_ms must lie inside the simulation interval")
    control_lower, control_upper = _bounds(event_timing.get("control_fraction", 0.0))
    if control_lower < 0 or control_upper > 1:
        raise ValueError("event_timing.control_fraction must lie in [0, 1]")

    positive_settings = (
        ("shape.initial_beta", config["shape"]["initial_beta"]),
        ("shape.late_beta", config["shape"]["late_beta"]),
        ("shape.beta_transition_ms", config["shape"]["beta_transition_ms"]),
        ("shape.axis_ratio", config["shape"]["axis_ratio"]),
        ("propagation.front_width_ms", config["propagation"]["front_width_ms"]),
        ("range.initial_lambda_mm", config["range"]["initial_lambda_mm"]),
        ("range.maximum_lambda_mm", config["range"]["maximum_lambda_mm"]),
        ("range.maximum_distance_mm", config["range"]["maximum_distance_mm"]),
        ("range.spread_time_ms", config["range"]["spread_time_ms"]),
        ("range.cutoff_width_mm", config["range"]["cutoff_width_mm"]),
        ("strength.rise_time_ms", config["strength"]["rise_time_ms"]),
        ("strength.decay_time_ms", config["strength"]["decay_time_ms"]),
    )
    for name, value in positive_settings:
        lower, _ = _bounds(value)
        if lower <= 0:
            raise ValueError(f"{name} must be strictly positive")
    peak_time_lower, _ = _bounds(config["strength"]["peak_time_ms"])
    if peak_time_lower < 0:
        raise ValueError("strength.peak_time_ms must be non-negative")

    propagation = config["propagation"]
    laws = propagation["law_choices"]
    laws = laws if isinstance(laws, list) else [laws]
    unsupported = set(laws) - {"ballistic", "diffusive"}
    if unsupported:
        raise ValueError(f"unsupported propagation laws: {sorted(unsupported)}")
    if "ballistic" in laws:
        lower, _ = _bounds(
            propagation.get("ballistic_speed_m_per_s", propagation.get("apparent_speed_m_per_s"))
        )
        if lower <= 0:
            raise ValueError("ballistic speed must be strictly positive")
    if "diffusive" in laws:
        lower, _ = _bounds(
            propagation.get(
                "diffusion_coefficient_mm2_per_ms", propagation.get("apparent_speed_m_per_s")
            )
        )
        if lower <= 0:
            raise ValueError("diffusion coefficient must be strictly positive")

    probability_settings = [
        ("strength.maximum_amplitude", config["strength"]["maximum_amplitude"]),
        ("strength.prompt_fraction", config["strength"].get("prompt_fraction", 0.0)),
        ("noise.missing_qubit_probability", config["noise"]["missing_qubit_probability"]),
        ("noise.baseline_drift_fraction", config["noise"]["baseline_drift_fraction"]),
    ]
    temporal_population = config.get("temporal_population", {})
    if temporal_population.get("enabled", False):
        probability_settings.append(
            ("temporal_population.peak_fraction_bounds", temporal_population["peak_fraction_bounds"])
        )
    for name, value in probability_settings:
        lower, upper = _bounds(value)
        if lower < 0 or upper > 1:
            raise ValueError(f"{name} must lie in [0, 1]")

    region_probabilities = np.asarray(
        list(config["epicenter"]["region_probabilities"].values()), dtype=float
    )
    if "reference_bounds_mm" in config["epicenter"]:
        bounds = np.asarray(config["epicenter"]["reference_bounds_mm"], dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("epicenter.reference_bounds_mm must be finite increasing x/y intervals")
    if (
        len(region_probabilities) == 0
        or not np.isfinite(region_probabilities).all()
        or np.any(region_probabilities < 0)
        or region_probabilities.sum() <= 0
    ):
        raise ValueError("epicenter region probabilities must be finite, non-negative, and nonzero")

    source = config.get("source_model", {})
    source_choices = source.get("source_count_choices", [1])
    source_choices = source_choices if isinstance(source_choices, list) else [source_choices]
    if not source_choices or set(source_choices) - {1, 2}:
        raise ValueError("source_count_choices may contain only 1 or 2")
    source_probabilities = source.get("source_count_probabilities")
    if source_probabilities is not None:
        probabilities = np.asarray(source_probabilities, dtype=float)
        if (
            probabilities.shape != (len(source_choices),)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "source_count_probabilities must match source_count_choices and sum to 1"
            )

    temporal_model = config.get("temporal_model", {})
    temporal_backend = temporal_model.get("backend", "empirical")
    if temporal_backend not in {"empirical", "qp_ode"}:
        raise ValueError("temporal_model.backend must be 'empirical' or 'qp_ode'")
    if temporal_backend == "qp_ode":
        qp = temporal_model.get("qp_ode", {})
        required_positive = (
            "trapping_rate_per_us",
            "generation_scale_per_us",
            "baseline_t1_us",
            "measurement_exposure_us",
            "al_gap_uev",
            "qubit_frequency_ghz",
        )
        for key in required_positive:
            if key not in qp:
                raise ValueError(f"temporal_model.qp_ode.{key} is required")
            lower, _ = _bounds(qp[key])
            if lower <= 0:
                raise ValueError(
                    f"temporal_model.qp_ode.{key} must be strictly positive"
                )
        if "recombination_rate_per_us" not in qp:
            raise ValueError(
                "temporal_model.qp_ode.recombination_rate_per_us is required"
            )
        recombination_lower, _ = _bounds(qp["recombination_rate_per_us"])
        if recombination_lower < 0:
            raise ValueError(
                "temporal_model.qp_ode.recombination_rate_per_us must be non-negative"
            )
        if "baseline_t2_us" in qp:
            lower, _ = _bounds(qp["baseline_t2_us"])
            if lower <= 0:
                raise ValueError(
                    "temporal_model.qp_ode.baseline_t2_us must be strictly positive"
                )
        dephasing = qp.get("qp_dephasing", {})
        if not isinstance(dephasing, dict):
            raise ValueError("temporal_model.qp_ode.qp_dephasing must be an object")
        unknown_dephasing = set(dephasing) - {"enabled", "charging_energy_ghz"}
        if unknown_dephasing:
            raise ValueError(
                "unknown temporal_model.qp_ode.qp_dephasing keys: "
                f"{sorted(unknown_dephasing)}"
            )
        if not isinstance(dephasing.get("enabled", False), bool):
            raise ValueError("temporal_model.qp_ode.qp_dephasing.enabled must be boolean")
        if dephasing.get("enabled", False):
            lower, _ = _bounds(dephasing.get("charging_energy_ghz", np.nan))
            if lower <= 0:
                raise ValueError(
                    "temporal_model.qp_ode.qp_dephasing.charging_energy_ghz "
                    "must be strictly positive"
                )
        frequency_shift = qp.get("frequency_shift", {})
        if not isinstance(frequency_shift, dict):
            raise ValueError("temporal_model.qp_ode.frequency_shift must be an object")
        unknown_shift = set(frequency_shift) - {"enabled", "coefficient_a"}
        if unknown_shift:
            raise ValueError(
                "unknown temporal_model.qp_ode.frequency_shift keys: "
                f"{sorted(unknown_shift)}"
            )
        if not isinstance(frequency_shift.get("enabled", False), bool):
            raise ValueError("temporal_model.qp_ode.frequency_shift.enabled must be boolean")
        if frequency_shift.get("enabled", False):
            lower, _ = _bounds(frequency_shift.get("coefficient_a", np.nan))
            if lower <= 0:
                raise ValueError(
                    "temporal_model.qp_ode.frequency_shift.coefficient_a "
                    "must be strictly positive"
                )
        maximum_radiation_probability = qp.get(
            "maximum_radiation_probability", 0.999
        )
        lower, upper = _bounds(maximum_radiation_probability)
        if lower <= 0 or upper > 1:
            raise ValueError(
                "temporal_model.qp_ode.maximum_radiation_probability must lie in (0, 1]"
            )
        source_decay_mode = qp.get("source_decay_mode", "legacy_response_decay")
        if source_decay_mode not in {
            "legacy_response_decay",
            "independent_lognormal",
        }:
            raise ValueError(
                "temporal_model.qp_ode.source_decay_mode must be "
                "'legacy_response_decay' or 'independent_lognormal'"
            )
        if source_decay_mode == "independent_lognormal":
            source_lifetime = qp.get("source_lifetime_ms", {})
            required = ("log_mean", "log_sd", "bounds_ms")
            missing = [key for key in required if key not in source_lifetime]
            if missing:
                raise ValueError(
                    "temporal_model.qp_ode.source_lifetime_ms is missing: "
                    + ", ".join(missing)
                )
            log_mean = float(source_lifetime["log_mean"])
            log_sd = float(source_lifetime["log_sd"])
            lower, upper = _bounds(source_lifetime["bounds_ms"])
            if not np.isfinite([log_mean, log_sd]).all() or log_sd < 0:
                raise ValueError(
                    "temporal_model.qp_ode.source_lifetime_ms requires finite "
                    "log_mean and non-negative log_sd"
                )
            if lower <= 0 or upper <= lower:
                raise ValueError(
                    "temporal_model.qp_ode.source_lifetime_ms.bounds_ms must be "
                    "strictly positive and ordered"
                )
        hardware = config.get("hardware_hierarchy", {})
        if not isinstance(hardware, dict):
            raise ValueError("hardware_hierarchy must be an object")
        allowed_hardware = {
            "enabled",
            "mode",
            "qubit_t1_log_sd",
            "qubit_t2_log_sd",
            "qubit_frequency_sd_ghz",
            "run_t1_log_sd",
            "run_t2_log_sd",
            "run_frequency_sd_ghz",
            "event_t1_log_sd",
            "event_t2_log_sd",
            "event_frequency_sd_ghz",
        }
        unknown_hardware = set(hardware) - allowed_hardware
        if unknown_hardware:
            raise ValueError(
                f"unknown hardware_hierarchy keys: {sorted(unknown_hardware)}"
            )
        if not isinstance(hardware.get("enabled", False), bool):
            raise ValueError("hardware_hierarchy.enabled must be boolean")
        if hardware.get("mode", "fixed_device") not in {
            "fixed_device",
            "device_population",
        }:
            raise ValueError(
                "hardware_hierarchy.mode must be 'fixed_device' or "
                "'device_population'"
            )
        for key in allowed_hardware - {"enabled", "mode"}:
            value = float(hardware.get(key, 0.0))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"hardware_hierarchy.{key} must be finite and non-negative")
    if config.get("output", {}).get("save_physics_diagnostics", False):
        if temporal_backend != "qp_ode":
            raise ValueError(
                "output.save_physics_diagnostics requires temporal_model.backend='qp_ode'"
            )


def draw(rng: np.random.Generator, value):
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            return float(rng.uniform(value[0], value[1]))
        return rng.choice(value).item()
    return value


def draw_truncated_lognormal(
    rng: np.random.Generator,
    *,
    log_mean: float,
    log_sd: float,
    bounds: tuple[float, float],
) -> float:
    """Draw a lognormal value without accumulating probability at the bounds."""
    lower, upper = map(float, bounds)
    if log_sd == 0:
        return float(np.clip(np.exp(log_mean), lower, upper))
    for _ in range(10_000):
        value = float(np.exp(rng.normal(log_mean, log_sd)))
        if lower <= value <= upper:
            return value
    raise RuntimeError("failed to sample the truncated lognormal distribution")


def sample_event_schedule(
    config: dict, n_events: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample event onsets and an exact-size no-radiation control group.

    A dedicated random stream keeps timing/control changes from perturbing the
    already validated event-shape, spatial, and QP parameter streams.
    """
    timing = config.get("event_timing", {})
    if not timing:
        return np.zeros(n_events, dtype=float), np.zeros(n_events, dtype=bool)
    rng = np.random.default_rng(int(seed) ^ 0x54494D45)
    onset_setting = timing.get("onset_time_ms", 0.0)
    onsets = np.asarray([draw(rng, onset_setting) for _ in range(n_events)], dtype=float)
    fraction = float(timing.get("control_fraction", 0.0))
    control_count = int(np.floor(fraction * n_events + 0.5))
    controls = np.zeros(n_events, dtype=bool)
    if control_count:
        controls[rng.choice(n_events, size=control_count, replace=False)] = True
    return onsets, controls


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_profile(profile_path: Path, seen: set[Path] | None = None) -> dict:
    """Load a profile with optional single/multi-level base_profile inheritance."""
    resolved = profile_path.resolve()
    seen = set() if seen is None else set(seen)
    if resolved in seen:
        raise ValueError(f"cyclic profile inheritance at {profile_path}")
    seen.add(resolved)
    profile = json.loads(profile_path.read_text())
    base_name = profile.pop("base_profile", None)
    if base_name is None:
        return profile
    base = _load_profile(profile_path.parent / base_name, seen)
    return deep_merge(base, profile)


def load_config(config_path: Path, profile_path: Path | None = None) -> dict:
    config = json.loads(config_path.read_text())
    embedded = config.get("device_profile")
    selected = profile_path
    if selected is None and embedded:
        selected = config_path.parent / embedded
    if selected is not None:
        config = deep_merge(config, _load_profile(selected))
        config["resolved_device_profile"] = str(selected)
    return config


def qp_relaxation_coefficient_per_us(
    al_gap_uev: float | np.ndarray, qubit_frequency_ghz: float | np.ndarray
) -> float | np.ndarray:
    """Return Delta_Gamma1 / x_qp in inverse microseconds.

    This is the quasiparticle-induced transmon relaxation relation used by
    Yelton et al. (arXiv:2402.15471), with SI inputs converted at the boundary.
    """
    delta_joule = np.asarray(al_gap_uev, dtype=float) * 1e-6 * ELECTRON_VOLT_J
    omega_per_s = 2 * np.pi * np.asarray(qubit_frequency_ghz, dtype=float) * 1e9
    coefficient_per_s = np.sqrt(2 * omega_per_s * delta_joule / HBAR_J_S) / np.pi
    result = coefficient_per_s * 1e-6
    return float(result) if result.ndim == 0 else result


def qp_dephasing_rate_per_us(
    x_qp: np.ndarray, charging_energy_ghz: float
) -> np.ndarray:
    """Return the QP-induced pure-dephasing rate from Baity et al. Appendix B.

    ``charging_energy_ghz`` is E_C / h.  The algebraically equivalent form below
    avoids directly exponentiating Lambert-W at very small QP densities.
    """
    density = np.maximum(np.asarray(x_qp, dtype=float), 0.0)
    result = np.zeros_like(density)
    positive = density > 0
    if np.any(positive):
        x = density[positive]
        w = np.real(lambertw(4 * np.pi / np.square(x)))
        ec_over_hbar_per_us = 2 * np.pi * float(charging_energy_ghz) * 1e3
        result[positive] = (
            ec_over_hbar_per_us * x / (np.pi ** 1.5 * np.sqrt(w))
        )
    return result


def _centered_lognormal(
    rng: np.random.Generator, size: int, log_sd: float
) -> np.ndarray:
    if log_sd == 0:
        return np.ones(size, dtype=float)
    values = rng.lognormal(mean=-0.5 * log_sd**2, sigma=log_sd, size=size)
    return values / values.mean()


def sample_qp_device_hardware(
    rng: np.random.Generator, config: dict, n_qubits: int, device_id: int
) -> dict[str, np.ndarray | float | int]:
    """Sample one persistent superconducting-device calibration state."""
    qp = config["temporal_model"]["qp_ode"]
    hierarchy = config.get("hardware_hierarchy", {})
    mean_t1 = float(draw(rng, qp["baseline_t1_us"]))
    mean_t2 = float(draw(rng, qp.get("baseline_t2_us", 2.0 * mean_t1)))
    mean_frequency = float(draw(rng, qp["qubit_frequency_ghz"]))
    gap = float(draw(rng, qp["al_gap_uev"]))
    t1 = mean_t1 * _centered_lognormal(
        rng, n_qubits, float(hierarchy.get("qubit_t1_log_sd", 0.0))
    )
    t2 = mean_t2 * _centered_lognormal(
        rng, n_qubits, float(hierarchy.get("qubit_t2_log_sd", 0.0))
    )
    t2 = np.minimum(t2, 2.0 * t1)
    frequency = mean_frequency + rng.normal(
        0.0,
        float(hierarchy.get("qubit_frequency_sd_ghz", 0.0)),
        size=n_qubits,
    )
    frequency += mean_frequency - frequency.mean()
    if np.any(frequency <= 0):
        raise ValueError("hardware hierarchy produced a non-positive qubit frequency")
    return {
        "device_id": int(device_id),
        "baseline_t1_us": t1,
        "baseline_t2_us": t2,
        "qubit_frequency_ghz": frequency,
        "al_gap_uev": gap,
    }


def _qp_exact_step(
    current: np.ndarray,
    generation_rate_per_us: np.ndarray,
    dt_us: float,
    trapping_rate_per_us: float,
    recombination_rate_per_us: float,
) -> np.ndarray:
    """Advance dx/dt = g - s*x - r*x^2 for constant g without Euler instability."""
    current = np.asarray(current, dtype=float)
    generation = np.maximum(np.asarray(generation_rate_per_us, dtype=float), 0.0)
    if dt_us <= 0:
        return current.copy()
    s = float(trapping_rate_per_us)
    r = float(recombination_rate_per_us)
    if r == 0:
        equilibrium = generation / s
        return np.maximum(
            equilibrium + (current - equilibrium) * np.exp(-s * dt_us), 0.0
        )

    discriminant = np.sqrt(s * s + 4 * r * generation)
    positive_root = (discriminant - s) / (2 * r)
    negative_root = -(discriminant + s) / (2 * r)
    ratio = (current - positive_root) / (current - negative_root)
    ratio *= np.exp(-discriminant * dt_us)
    denominator = 1 - ratio
    advanced = np.divide(
        positive_root - ratio * negative_root,
        denominator,
        out=np.zeros_like(current),
        where=np.abs(denominator) > np.finfo(float).eps,
    )
    return np.maximum(advanced, 0.0)


def solve_qp_dynamics(
    time_ms: np.ndarray,
    generation_rate_per_us: np.ndarray,
    trapping_rate_per_us: float,
    recombination_rate_per_us: float,
) -> np.ndarray:
    """Solve independent local QP-density ODEs for arrays shaped (time, qubit)."""
    time_ms = np.asarray(time_ms, dtype=float)
    generation = np.asarray(generation_rate_per_us, dtype=float)
    if generation.ndim != 2 or generation.shape[0] != len(time_ms):
        raise ValueError("generation rate must have shape (time, qubit)")
    if len(time_ms) == 0 or not np.isfinite(time_ms).all():
        raise ValueError("time_ms must be a non-empty finite array")
    if len(time_ms) > 1 and np.any(np.diff(time_ms) <= 0):
        raise ValueError("time_ms must be strictly increasing")
    if not np.isfinite(generation).all() or np.any(generation < 0):
        raise ValueError("generation rate must be finite and non-negative")
    if trapping_rate_per_us <= 0 or recombination_rate_per_us < 0:
        raise ValueError("QP trapping must be positive and recombination non-negative")

    density = np.zeros_like(generation, dtype=float)
    for index in range(1, len(time_ms)):
        interval_start = time_ms[index - 1]
        interval_end = time_ms[index]
        if interval_end <= 0:
            continue
        active_start = max(interval_start, 0.0)
        dt_us = (interval_end - active_start) * 1000.0
        if interval_start < 0:
            interval_generation = 0.5 * generation[index]
        else:
            interval_generation = 0.5 * (
                generation[index - 1] + generation[index]
            )
        density[index] = _qp_exact_step(
            density[index - 1],
            interval_generation,
            dt_us,
            trapping_rate_per_us,
            recombination_rate_per_us,
        )
    return density


def spatial_covariance(coords: np.ndarray, log_sd: float, correlation_length: float) -> np.ndarray:
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    covariance = log_sd**2 * np.exp(-0.5 * np.square(distance / correlation_length))
    covariance.flat[:: len(coords) + 1] += 1e-8
    return covariance


def spatial_log_field(
    rng: np.random.Generator, coords: np.ndarray, log_sd: float, correlation_length: float
) -> np.ndarray:
    if log_sd <= 0:
        return np.zeros(len(coords))
    field = rng.multivariate_normal(
        np.zeros(len(coords)), spatial_covariance(coords, log_sd, correlation_length)
    )
    return field - field.mean()


def quadratic_coordinate_field(coords: np.ndarray, coefficients: list[float]) -> np.ndarray:
    """Evaluate a smooth, layout-aware log-sensitivity field.

    The five coefficients multiply x, y, x^2-1, y^2-1, and x*y after each
    coordinate axis is standardized.  This deliberately cannot memorize one
    free correction per qubit; it represents only a smooth device-scale trend.
    """
    coefficients_array = np.asarray(coefficients, dtype=float)
    if coefficients_array.shape != (5,):
        raise ValueError("quadratic device response requires exactly 5 coefficients")
    centered = coords - coords.mean(axis=0)
    scale = coords.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x, y = (centered / scale).T
    basis = np.column_stack((x, y, x * x - 1.0, y * y - 1.0, x * y))
    field = basis @ coefficients_array
    return field - field.mean()


def sample_low_rank_spatial_field(
    rng: np.random.Generator, config: dict, n_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a compact device-specific event field from learned covariance modes."""
    if not config.get("enabled", False):
        return np.zeros(n_qubits), np.empty(0)
    modes = np.asarray(config["modes"], dtype=float)
    coefficient_sd = np.asarray(config["coefficient_sd"], dtype=float)
    if modes.ndim != 2 or modes.shape[1] != n_qubits:
        raise ValueError("low-rank spatial modes must have shape (rank, n_qubits)")
    if coefficient_sd.shape != (modes.shape[0],):
        raise ValueError("one low-rank coefficient SD is required per mode")
    coefficients = rng.normal(0.0, coefficient_sd)
    field = coefficients @ modes
    field -= field.mean()
    maximum = float(config.get("maximum_abs_log_sensitivity", 0.75))
    return np.clip(field, -maximum, maximum), coefficients


def time_correlated_residual(
    rng: np.random.Generator, coords: np.ndarray, time_ms: np.ndarray, config: dict
) -> np.ndarray:
    if not config.get("enabled", False):
        return np.zeros((len(time_ms), len(coords)), dtype=np.float32)
    log_sd = draw(rng, config["initial_log_sd"])
    correlation_length = draw(rng, config["correlation_length_mm"])
    decay = draw(rng, config["decay_time_ms"])
    late_fraction = draw(rng, config["late_fraction"])
    initial = spatial_log_field(rng, coords, log_sd, correlation_length)
    late = spatial_log_field(rng, coords, log_sd * late_fraction, correlation_length)
    positive = np.maximum(time_ms, 0)[:, None]
    initial_weight = np.exp(-positive / decay)
    residual = initial_weight * initial[None, :] + (1 - initial_weight) * late[None, :]
    residual[time_ms < 0] = 0
    return residual.astype(np.float32)


def sample_temporal_population(
    rng: np.random.Generator, config: dict, baseline_fraction: float
) -> tuple[float, float] | None:
    """Sample peak fraction and decay conditional on an event's baseline.

    The calibrated variables are (logit baseline, logit peak, log decay).
    Conditioning lets a generic run hierarchy preserve their measured
    correlations without storing any individual event.
    """
    if not config.get("enabled", False):
        return None
    mean = np.asarray(config["joint_transformed_mean"], dtype=float)
    covariance = np.asarray(config["joint_transformed_covariance"], dtype=float)
    baseline_logit = np.log(baseline_fraction / (1 - baseline_fraction))
    cross = covariance[1:, 0]
    conditional_mean = mean[1:] + cross / covariance[0, 0] * (baseline_logit - mean[0])
    conditional_covariance = covariance[1:, 1:] - np.outer(cross, cross) / covariance[0, 0]
    conditional_covariance.flat[::3] += 1e-8
    peak_logit, log_decay = rng.multivariate_normal(conditional_mean, conditional_covariance)
    peak = float(expit(peak_logit))
    decay = float(np.exp(log_decay))
    peak_bounds = config.get("peak_fraction_bounds", [0.02, 0.98])
    decay_bounds = config.get("decay_time_bounds_ms", [2.0, 150.0])
    peak = float(np.clip(peak, peak_bounds[0], peak_bounds[1]))
    decay = float(np.clip(decay, decay_bounds[0], decay_bounds[1]))
    return max(peak, baseline_fraction + 0.01), decay


def sample_epicenter(
    rng: np.random.Generator,
    config: dict,
    coords: np.ndarray = DEFAULT_COORDS,
) -> tuple[float, float, str]:
    coords = np.asarray(coords, dtype=float)
    bounds = np.asarray(
        [
            [coords[:, 0].min(), coords[:, 0].max()],
            [coords[:, 1].min(), coords[:, 1].max()],
        ]
    )
    if "reference_bounds_mm" in config:
        bounds = np.asarray(config["reference_bounds_mm"], dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("invalid epicenter reference bounds")
    names = list(config["region_probabilities"])
    probabilities = np.asarray(list(config["region_probabilities"].values()), dtype=float)
    probabilities /= probabilities.sum()
    region = str(rng.choice(names, p=probabilities))
    if region == "inside":
        point = rng.uniform(bounds[:, 0], bounds[:, 1])
    else:
        side = int(rng.integers(4))
        margin = draw(rng, config["outside_margin_mm"])
        point = rng.uniform(bounds[:, 0], bounds[:, 1])
        axis, high = divmod(side, 2)
        point[axis] = bounds[axis, high]
        if region == "outside":
            point[axis] += margin if high else -margin
    return float(point[0]), float(point[1]), region


def elliptical_distance(
    coords: np.ndarray, center: np.ndarray, axis_ratio: float, angle_degrees: float
) -> np.ndarray:
    delta = coords - center
    angle = np.radians(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    major = cosine * delta[:, 0] + sine * delta[:, 1]
    minor = -sine * delta[:, 0] + cosine * delta[:, 1]
    # Area-preserving transform: ratio=1 is circular.
    return np.sqrt((major / np.sqrt(axis_ratio)) ** 2 + (minor * np.sqrt(axis_ratio)) ** 2)


def sample_parameters(
    rng: np.random.Generator,
    config: dict,
    *,
    physics_rng: np.random.Generator | None = None,
    source_rng: np.random.Generator | None = None,
    coords: np.ndarray = DEFAULT_COORDS,
) -> dict:
    row, column, region = sample_epicenter(rng, config["epicenter"], coords)
    geometry = draw(rng, config["shape"]["geometry_choices"])
    propagation_law = draw(rng, config["propagation"]["law_choices"])
    propagation = config["propagation"]
    if propagation_law == "ballistic":
        ballistic_speed = draw(
            rng,
            propagation.get(
                "ballistic_speed_m_per_s", propagation["apparent_speed_m_per_s"]
            ),
        )
        diffusion_coefficient = np.nan
    else:
        ballistic_speed = np.nan
        diffusion_coefficient = draw(
            rng,
            propagation.get(
                "diffusion_coefficient_mm2_per_ms", propagation["apparent_speed_m_per_s"]
            ),
        )
    source_config = config.get("source_model", {})
    source_choices = source_config.get("source_count_choices", [1])
    source_probabilities = source_config.get("source_count_probabilities")
    source_count = int(rng.choice(source_choices, p=source_probabilities))
    secondary_row = secondary_col = secondary_strength = np.nan
    if source_count == 2:
        separation = draw(rng, source_config.get("separation_mm", [0.5, 3.0]))
        source_angle = rng.uniform(0, 2 * np.pi)
        secondary_row = row + separation * np.cos(source_angle)
        secondary_col = column + separation * np.sin(source_angle)
        secondary_strength = draw(rng, source_config.get("secondary_strength", [0.2, 0.8]))
    reflection_config = config.get("boundary_reflection", {})
    reflection_enabled = (
        rng.random() < reflection_config["enabled_probability"]
        if "enabled_probability" in reflection_config
        else bool(reflection_config.get("enabled", False))
    )
    initial_lambda = draw(rng, config["range"]["initial_lambda_mm"])
    maximum_lambda = max(initial_lambda, draw(rng, config["range"]["maximum_lambda_mm"]))
    halo_config = config.get("halo", {})
    halo_enabled = (
        rng.random() < halo_config.get("enabled_probability", 0.0)
        if "enabled_probability" in halo_config
        else bool(halo_config.get("enabled", False))
    )
    temporal_config = config.get("temporal_model", {})
    temporal_backend = temporal_config.get("backend", "empirical")
    parameters = {
        "epicenter_row": row,
        "epicenter_col": column,
        "epicenter_region": region,
        "geometry": geometry,
        "initial_beta": draw(rng, config["shape"]["initial_beta"]),
        "late_beta": draw(rng, config["shape"]["late_beta"]),
        "beta_transition_ms": draw(rng, config["shape"]["beta_transition_ms"]),
        "axis_ratio": draw(rng, config["shape"]["axis_ratio"]) if geometry == "elliptical" else 1.0,
        "angle_degrees": draw(rng, config["shape"]["angle_degrees"]),
        "source_count": source_count,
        "secondary_epicenter_row": secondary_row,
        "secondary_epicenter_col": secondary_col,
        "secondary_source_strength": secondary_strength,
        "reflection_enabled": reflection_enabled,
        "reflection_coefficient": draw(rng, reflection_config.get("coefficient", 0.0)),
        "reflection_boundary_margin_mm": draw(
            rng, reflection_config.get("boundary_margin_mm", 0.0)
        ),
        "halo_enabled": halo_enabled,
        "halo_strength": draw(rng, halo_config.get("strength", 0.0)),
        "halo_lambda_multiplier": draw(rng, halo_config.get("lambda_multiplier", 1.0)),
        "halo_beta": draw(rng, halo_config.get("beta", 1.0)),
        "halo_front_width_multiplier": draw(
            rng, halo_config.get("front_width_multiplier", 1.0)
        ),
        "propagation_law": propagation_law,
        "apparent_speed_m_per_s": ballistic_speed,
        "diffusion_coefficient_mm2_per_ms": diffusion_coefficient,
        "front_width_ms": draw(rng, config["propagation"]["front_width_ms"]),
        "arrival_jitter_sd_us": draw(
            rng, config["propagation"].get("arrival_jitter_sd_us", 0.0)
        ),
        "arrival_jitter_correlation_length_mm": draw(
            rng,
            config["propagation"].get("arrival_jitter_correlation_length_mm", 1.0),
        ),
        "initial_lambda_mm": initial_lambda,
        "maximum_lambda_mm": maximum_lambda,
        "maximum_distance_mm": draw(rng, config["range"]["maximum_distance_mm"]),
        "spread_time_ms": draw(rng, config["range"]["spread_time_ms"]),
        "cutoff_width_mm": draw(rng, config["range"]["cutoff_width_mm"]),
        "maximum_amplitude": draw(rng, config["strength"]["maximum_amplitude"]),
        "prompt_fraction": draw(rng, config["strength"].get("prompt_fraction", 0.0)),
        "rise_time_ms": draw(rng, config["strength"]["rise_time_ms"]),
        "peak_time_ms": draw(rng, config["strength"]["peak_time_ms"]),
        "decay_time_ms": draw(rng, config["strength"]["decay_time_ms"]),
        "baseline_total_errors": draw(rng, config["noise"]["baseline_total_errors"]),
        "baseline_drift_fraction": draw(rng, config["noise"]["baseline_drift_fraction"]),
        "temporal_backend": temporal_backend,
    }
    if temporal_backend == "qp_ode":
        qp = temporal_config["qp_ode"]
        qp_rng = rng if physics_rng is None else physics_rng
        source_decay_mode = qp.get("source_decay_mode", "legacy_response_decay")
        parameters.update(
            {
                "qp_trapping_rate_per_us": draw(
                    qp_rng, qp["trapping_rate_per_us"]
                ),
                "qp_recombination_rate_per_us": draw(
                    qp_rng, qp["recombination_rate_per_us"]
                ),
                "qp_generation_scale_per_us": draw(
                    qp_rng, qp["generation_scale_per_us"]
                ),
                "qp_baseline_t1_us": draw(qp_rng, qp["baseline_t1_us"]),
                "qp_baseline_t2_us": draw(
                    qp_rng,
                    qp.get("baseline_t2_us", np.nan),
                ),
                "qp_measurement_exposure_us": draw(
                    qp_rng, qp["measurement_exposure_us"]
                ),
                "qp_al_gap_uev": draw(qp_rng, qp["al_gap_uev"]),
                "qp_qubit_frequency_ghz": draw(
                    qp_rng, qp["qubit_frequency_ghz"]
                ),
                "qp_maximum_radiation_probability": draw(
                    qp_rng, qp.get("maximum_radiation_probability", 0.999)
                ),
                "qp_source_decay_mode": source_decay_mode,
                "qp_dephasing_enabled": bool(
                    qp.get("qp_dephasing", {}).get("enabled", False)
                ),
                "qp_charging_energy_ghz": float(
                    qp.get("qp_dephasing", {}).get("charging_energy_ghz", 0.0)
                ),
                "qp_frequency_shift_enabled": bool(
                    qp.get("frequency_shift", {}).get("enabled", False)
                ),
                "qp_frequency_shift_coefficient_a": float(
                    qp.get("frequency_shift", {}).get("coefficient_a", 0.0)
                ),
            }
        )
        if "baseline_t2_us" not in qp:
            parameters["qp_baseline_t2_us"] = (
                2.0 * float(parameters["qp_baseline_t1_us"])
            )
        if source_decay_mode == "independent_lognormal":
            source_distribution = qp["source_lifetime_ms"]
            lifetime_rng = qp_rng if source_rng is None else source_rng
            parameters["qp_source_lifetime_ms"] = draw_truncated_lognormal(
                lifetime_rng,
                log_mean=float(source_distribution["log_mean"]),
                log_sd=float(source_distribution["log_sd"]),
                bounds=_bounds(source_distribution["bounds_ms"]),
            )
    return parameters


def qp_response_from_drive(
    time_ms: np.ndarray,
    drive: np.ndarray,
    parameters: dict,
    *,
    drive_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Convert a local non-negative QP-generation proxy into relaxation.

    ``drive`` samples a local density-like field at each requested coordinate.
    Its sum over qubits is not deposited energy and must not be normalized by
    the number of coordinates. Absolute energy accounting requires a separate
    transport backend with electrode area/volume and conversion efficiency.
    """
    generation = (
        np.maximum(np.asarray(drive, dtype=float), 0.0)
        * float(parameters["qp_generation_scale_per_us"])
        * float(drive_scale)
    )
    density = solve_qp_dynamics(
        time_ms,
        generation,
        float(parameters["qp_trapping_rate_per_us"]),
        float(parameters["qp_recombination_rate_per_us"]),
    )
    frequency = np.asarray(
        parameters.get(
            "_qp_qubit_frequency_by_qubit_ghz",
            parameters["qp_qubit_frequency_ghz"],
        ),
        dtype=float,
    )
    baseline_t1 = np.asarray(
        parameters.get(
            "_qp_baseline_t1_by_qubit_us", parameters["qp_baseline_t1_us"]
        ),
        dtype=float,
    )
    baseline_t2 = np.asarray(
        parameters.get(
            "_qp_baseline_t2_by_qubit_us",
            parameters.get("qp_baseline_t2_us", 2.0 * baseline_t1),
        ),
        dtype=float,
    )
    coefficient = np.asarray(
        qp_relaxation_coefficient_per_us(
            float(parameters["qp_al_gap_uev"]), frequency
        ),
        dtype=float,
    )
    delta_gamma1 = density * coefficient
    baseline_rate = 1.0 / baseline_t1
    t1_us = 1.0 / (baseline_rate + delta_gamma1)
    baseline_gamma_phi = np.maximum(
        1.0 / baseline_t2 - 0.5 / baseline_t1,
        0.0,
    )
    if parameters.get("qp_dephasing_enabled", False):
        delta_gamma_phi = qp_dephasing_rate_per_us(
            density, float(parameters["qp_charging_energy_ghz"])
        )
    else:
        delta_gamma_phi = np.zeros_like(density)
    t2_us = 1.0 / (0.5 / t1_us + baseline_gamma_phi + delta_gamma_phi)
    if parameters.get("qp_frequency_shift_enabled", False):
        frequency_shift_ghz = (
            -float(parameters["qp_frequency_shift_coefficient_a"])
            * density
            * frequency
        )
    else:
        frequency_shift_ghz = np.zeros_like(density)
    radiation_probability = -np.expm1(
        -delta_gamma1 * float(parameters["qp_measurement_exposure_us"])
    )
    radiation_probability = np.clip(
        radiation_probability,
        0.0,
        float(parameters["qp_maximum_radiation_probability"]),
    )
    return radiation_probability, {
        "qp_generation_rate_per_us": generation.astype(np.float32),
        "x_qp": density.astype(np.float32),
        "delta_gamma1_per_us": delta_gamma1.astype(np.float32),
        "delta_gamma_phi_per_us": delta_gamma_phi.astype(np.float32),
        "t1_us": t1_us.astype(np.float32),
        "t2_us": t2_us.astype(np.float32),
        "frequency_shift_ghz": frequency_shift_ghz.astype(np.float32),
        "qp_relaxation_coefficient_per_us": coefficient,
        "qp_generation_event_scale": float(drive_scale),
    }


def event_probability(
    time_ms: np.ndarray,
    parameters: dict,
    baseline: np.ndarray,
    sensitivity: np.ndarray,
    residual: np.ndarray | None = None,
    arrival_jitter_ms: np.ndarray | None = None,
    coords: np.ndarray = DEFAULT_COORDS,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | float]]:
    event_time_ms = np.asarray(time_ms, dtype=float) - float(
        parameters.get("event_onset_ms", 0.0)
    )
    positive_time = np.maximum(event_time_ms, 0)[:, None]
    temporal_backend = parameters.get("temporal_backend", "empirical")

    peak = parameters["peak_time_ms"]
    slow_rise = 1 - np.exp(-positive_time / parameters["rise_time_ms"])
    normalization = 1 - np.exp(-peak / parameters["rise_time_ms"])
    if normalization <= np.finfo(float).eps:
        normalized_rise = (event_time_ms[:, None] >= 0).astype(float)
    else:
        normalized_rise = np.minimum(slow_rise / normalization, 1.0)
    rise = parameters.get("prompt_fraction", 0.0) + (
        1 - parameters.get("prompt_fraction", 0.0)
    ) * normalized_rise
    amplitude = parameters["maximum_amplitude"] * rise
    source_lifetime_ms = parameters["decay_time_ms"]
    if (
        temporal_backend == "qp_ode"
        and parameters.get("qp_source_decay_mode") == "independent_lognormal"
    ):
        source_lifetime_ms = parameters["qp_source_lifetime_ms"]
    amplitude *= np.exp(
        -np.maximum(positive_time - peak, 0) / source_lifetime_ms
    )
    amplitude[event_time_ms < 0] = 0

    spread_fraction = 1 - np.exp(-positive_time / parameters["spread_time_ms"])
    scale = parameters["initial_lambda_mm"] + (
        parameters["maximum_lambda_mm"] - parameters["initial_lambda_mm"]
    ) * spread_fraction
    beta_fraction = 1 - np.exp(-positive_time / parameters["beta_transition_ms"])
    beta = parameters["initial_beta"] + (
        parameters["late_beta"] - parameters["initial_beta"]
    ) * beta_fraction
    centers = [np.asarray([parameters["epicenter_row"], parameters["epicenter_col"]])]
    weights = [1.0]
    if parameters.get("source_count", 1) == 2:
        centers.append(
            np.asarray(
                [parameters["secondary_epicenter_row"], parameters["secondary_epicenter_col"]]
            )
        )
        weights.append(parameters["secondary_source_strength"])
    component_centers = list(centers)
    component_weights = list(weights)
    if parameters.get("reflection_enabled", False):
        margin = parameters.get("reflection_boundary_margin_mm", 0.0)
        row_low, row_high = coords[:, 0].min() - margin, coords[:, 0].max() + margin
        col_low, col_high = coords[:, 1].min() - margin, coords[:, 1].max() + margin
        reflected_centers, reflected_weights = [], []
        for center, weight in zip(centers, weights):
            reflected_centers.extend(
                [
                    np.asarray([2 * row_low - center[0], center[1]]),
                    np.asarray([2 * row_high - center[0], center[1]]),
                    np.asarray([center[0], 2 * col_low - center[1]]),
                    np.asarray([center[0], 2 * col_high - center[1]]),
                ]
            )
            reflected_weights.extend(
                [weight * parameters["reflection_coefficient"]] * 4
            )
        component_centers.extend(reflected_centers)
        component_weights.extend(reflected_weights)

    spatial_survival = np.ones((len(time_ms), len(coords)))
    for center, weight in zip(component_centers, component_weights):
        distance = elliptical_distance(
            coords, center, parameters["axis_ratio"], parameters["angle_degrees"]
        )
        if parameters["propagation_law"] == "ballistic":
            speed = parameters["apparent_speed_m_per_s"]
            arrival_ms = distance / speed
        else:
            coefficient = parameters.get("diffusion_coefficient_mm2_per_ms")
            if coefficient is None or not np.isfinite(coefficient):
                coefficient = parameters["apparent_speed_m_per_s"]
            arrival_ms = np.square(distance) / coefficient
        if arrival_jitter_ms is not None:
            arrival_ms = np.maximum(arrival_ms + arrival_jitter_ms, 0)
        front = expit(
            (event_time_ms[:, None] - arrival_ms[None, :])
            / parameters["front_width_ms"]
        )
        radial_profile = np.exp(-np.power(distance[None, :] / scale, beta))
        cutoff = expit(
            (parameters["maximum_distance_mm"] - distance) / parameters["cutoff_width_mm"]
        )[None, :]
        component = np.clip(weight * front * radial_profile * cutoff, 0, 0.999)
        if parameters.get("halo_enabled", False):
            halo_front = expit(
                (event_time_ms[:, None] - arrival_ms[None, :])
                / (
                    parameters["front_width_ms"]
                    * parameters.get("halo_front_width_multiplier", 1.0)
                )
            )
            halo_scale = scale * parameters.get("halo_lambda_multiplier", 1.0)
            halo_profile = np.exp(
                -np.power(distance[None, :] / halo_scale, parameters.get("halo_beta", 1.0))
            )
            halo = np.clip(
                weight
                * parameters.get("halo_strength", 0.0)
                * halo_front
                * halo_profile
                * cutoff,
                0,
                0.999,
            )
            component = 1 - (1 - component) * (1 - halo)
        spatial_survival *= 1 - component
    spatial_response = 1 - spatial_survival
    mixing = parameters.get("global_spatial_mixing_fraction", 0.0)
    if mixing:
        # A device observation layer: a fraction of the local disturbance is
        # seen as chip-wide common-mode response.  The time-dependent spatial
        # mean is preserved, so this changes shape without changing strength.
        spatial_mean = spatial_response.mean(axis=1, keepdims=True)
        spatial_response = (1 - mixing) * spatial_response + mixing * spatial_mean
    residual_multiplier = 1.0 if residual is None else np.exp(residual)
    unscaled_drive = amplitude * spatial_response * sensitivity[None, :] * residual_multiplier
    if bool(parameters.get("is_control", False)):
        unscaled_drive = np.zeros_like(unscaled_drive)

    drifting_baseline = drifting_baseline_probability(
        time_ms, baseline, parameters["baseline_drift_fraction"]
    )

    target_peak = (
        None
        if bool(parameters.get("is_control", False))
        else parameters.get("target_peak_error_fraction")
    )

    def disturbance_for_scale(
        scale_factor: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        if temporal_backend == "empirical":
            return np.clip(unscaled_drive * scale_factor, 0, 0.999), {}
        if temporal_backend == "qp_ode":
            return qp_response_from_drive(
                event_time_ms, unscaled_drive, parameters, drive_scale=scale_factor
            )
        raise ValueError(f"unsupported temporal backend: {temporal_backend!r}")

    event_scale = 1.0
    if target_peak is not None:
        early = (event_time_ms >= 0) & (event_time_ms < 5.0)
        if early.any():
            early_baseline = drifting_baseline[early]

            def peak_probability(scale_factor: float) -> float:
                disturbance, _ = disturbance_for_scale(scale_factor)
                combined = 1 - (1 - early_baseline) * (1 - disturbance[early])
                return float(combined.mean(axis=1).max())

            # Background drift can occasionally put the requested raw peak below
            # the no-event probability. In that case, zero event amplitude is the
            # closest physically attainable result.
            target_peak = max(float(target_peak), peak_probability(0.0))

            low, high = 0.0, 1.0
            # Distant/outside sources can have a very small unscaled response.
            # The former cap of 1024 silently left some requested peaks
            # unattained even though the monotone probability curve could reach
            # them at a larger scale.
            while peak_probability(high) < target_peak and high < 2**40:
                high *= 2
            for _ in range(24):
                middle = (low + high) / 2
                if peak_probability(middle) < target_peak:
                    low = middle
                else:
                    high = middle
            event_scale = (low + high) / 2
    event, diagnostics = disturbance_for_scale(event_scale)
    probability = 1 - (1 - drifting_baseline) * (1 - event)
    if return_diagnostics:
        diagnostics = dict(diagnostics)
        diagnostics["radiation_probability"] = event.astype(np.float32)
        diagnostics["event_scale"] = float(event_scale)
        return probability, diagnostics
    return probability


def drifting_baseline_probability(
    time_ms: np.ndarray, baseline: np.ndarray, drift_fraction: float
) -> np.ndarray:
    """Return the time-dependent no-event response probability.

    Keeping this component explicit prevents baseline drift from being mistaken for
    radiation-induced response by downstream observation layers.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    drift = float(drift_fraction) * np.sin(
        2 * np.pi * (time_ms - time_ms.min()) / max(np.ptp(time_ms), 1e-9)
    )
    return np.clip(baseline[None, :] * (1 + drift[:, None]), 1e-5, 0.95)


def simulate(
    config: dict,
    *,
    coords: np.ndarray = DEFAULT_COORDS,
    include_baseline_probabilities: bool = False,
    include_physics_arrays: bool = False,
) -> tuple:
    coords = np.asarray(coords, dtype=float)
    validate_config(config, coords)
    rng = np.random.default_rng(config["seed"])
    physics_rng = np.random.default_rng(int(config["seed"]) ^ 0x51504F44)
    source_rng = np.random.default_rng(int(config["seed"]) ^ 0x534F5552)
    hardware_rng = np.random.default_rng(int(config.get("device_seed", config["seed"])) ^ 0x48415244)
    device_rng = (
        rng if "device_seed" not in config
        else np.random.default_rng(int(config["device_seed"]) ^ 0x44455649)
    )
    event_onsets_ms, control_events = sample_event_schedule(
        config, int(config["n_events"]), int(config["seed"])
    )
    time_config = config["time"]
    time_ms = np.arange(time_config["start_ms"], time_config["end_ms"], time_config["dt_ms"])
    observations = np.empty((config["n_events"], len(time_ms), len(coords)), dtype=np.uint8)
    save_probability = bool(config["output"].get("save_probabilities", False))
    probabilities = np.empty(observations.shape, dtype=np.float16) if save_probability else None
    baseline_probabilities = (
        np.empty(observations.shape, dtype=np.float16)
        if save_probability and include_baseline_probabilities
        else None
    )
    temporal_backend = config.get("temporal_model", {}).get("backend", "empirical")
    save_physics = bool(config["output"].get("save_physics_diagnostics", False))
    physics_arrays: dict[str, np.ndarray] = {}
    if save_physics:
        physics_shape = observations.shape
        physics_arrays = {
            name: np.empty(physics_shape, dtype=np.float32)
            for name in (
                "qp_generation_rate_per_us",
                "x_qp",
                "delta_gamma1_per_us",
                "delta_gamma_phi_per_us",
                "t1_us",
                "t2_us",
                "frequency_shift_ghz",
                "radiation_probability",
            )
        }
    baselines = np.empty((config["n_events"], len(coords)), dtype=np.float32)
    rows = []

    noise = config["noise"]
    hierarchy = config.get("hierarchical_sensitivity", {})
    if hierarchy.get("enabled", False):
        device_log_field = spatial_log_field(
            device_rng,
            coords,
            float(hierarchy["device_log_sd"]),
            float(hierarchy["correlation_length_mm"]),
        )
        device_sensitivity = np.exp(device_log_field)
        device_sensitivity /= device_sensitivity.mean()
    else:
        device_sensitivity = np.ones(len(coords))
    device_response = config.get("device_spatial_response", {})
    if device_response.get("enabled", False):
        model = device_response.get("model", "quadratic_log_field")
        if model != "quadratic_log_field":
            raise ValueError(f"unsupported device spatial response model: {model}")
        calibrated_field = quadratic_coordinate_field(
            coords, device_response["coefficients"]
        )
        maximum = float(device_response.get("maximum_abs_log_sensitivity", 0.75))
        calibrated_field = np.clip(calibrated_field, -maximum, maximum)
        device_sensitivity *= np.exp(calibrated_field)
        device_sensitivity /= device_sensitivity.mean()
    run_hierarchy = config.get("run_hierarchy", {})
    hardware_hierarchy = config.get("hardware_hierarchy", {})
    hardware_enabled = bool(
        temporal_backend == "qp_ode" and hardware_hierarchy.get("enabled", False)
    )
    hardware_mode = hardware_hierarchy.get("mode", "fixed_device")
    fixed_hardware = (
        sample_qp_device_hardware(hardware_rng, config, len(coords), device_id=0)
        if hardware_enabled and hardware_mode == "fixed_device"
        else None
    )
    current_hardware = fixed_hardware
    current_hardware_run_id = None
    hardware_device_count = 0 if fixed_hardware is not None else -1
    run_t1_factor = 1.0
    run_t2_factor = 1.0
    run_frequency_offset_ghz = 0.0
    run_id = -1
    events_left_in_run = 0
    run_baseline_logit = None
    for event in range(config["n_events"]):
        parameters = sample_parameters(
            rng,
            config,
            physics_rng=physics_rng,
            source_rng=source_rng,
            coords=coords,
        )
        parameters["event_onset_ms"] = float(event_onsets_ms[event])
        parameters["is_control"] = bool(control_events[event])
        mixing_config = config.get("observation_spatial_mixing", {})
        parameters["global_spatial_mixing_fraction"] = (
            draw(rng, mixing_config.get("global_fraction", 0.0))
            if mixing_config.get("enabled", False)
            else 0.0
        )
        if run_hierarchy.get("enabled", False):
            if events_left_in_run == 0:
                run_id += 1
                run_distribution = run_hierarchy.get("events_per_run_distribution")
                if run_distribution == "shifted_poisson":
                    minimum = int(run_hierarchy.get("minimum_events_per_run", 1))
                    maximum = int(run_hierarchy.get("maximum_events_per_run", 20))
                    mean = float(run_hierarchy["mean_events_per_run"])
                    events_left_in_run = int(
                        np.clip(minimum + rng.poisson(max(mean - minimum, 0.0)), minimum, maximum)
                    )
                elif run_distribution is None:
                    choices = run_hierarchy.get("events_per_run_choices", [1])
                    events_left_in_run = int(rng.choice(choices))
                else:
                    raise ValueError(f"unsupported events_per_run_distribution: {run_distribution}")
                run_baseline_logit = float(
                    rng.normal(
                        run_hierarchy["baseline_logit_mean"],
                        run_hierarchy["baseline_run_logit_sd"],
                    )
                )
            assert run_baseline_logit is not None
            event_baseline_logit = run_baseline_logit + rng.normal(
                0, run_hierarchy["baseline_event_logit_sd"]
            )
            baseline_fraction = float(expit(event_baseline_logit))
            parameters["baseline_total_errors"] = baseline_fraction * len(coords)
            temporal_sample = sample_temporal_population(
                rng, config.get("temporal_population", {}), baseline_fraction
            )
            if temporal_sample is not None and not parameters["is_control"]:
                parameters["target_peak_error_fraction"], parameters["decay_time_ms"] = temporal_sample
            events_left_in_run -= 1
        else:
            run_id = event
        if hardware_enabled:
            if current_hardware_run_id != run_id:
                current_hardware_run_id = run_id
                if hardware_mode == "device_population":
                    hardware_device_count += 1
                    current_hardware = sample_qp_device_hardware(
                        hardware_rng,
                        config,
                        len(coords),
                        device_id=hardware_device_count,
                    )
                assert current_hardware is not None
                run_t1_factor = float(
                    hardware_rng.lognormal(
                        mean=-0.5
                        * float(hardware_hierarchy.get("run_t1_log_sd", 0.0)) ** 2,
                        sigma=float(hardware_hierarchy.get("run_t1_log_sd", 0.0)),
                    )
                )
                run_t2_factor = float(
                    hardware_rng.lognormal(
                        mean=-0.5
                        * float(hardware_hierarchy.get("run_t2_log_sd", 0.0)) ** 2,
                        sigma=float(hardware_hierarchy.get("run_t2_log_sd", 0.0)),
                    )
                )
                run_frequency_offset_ghz = float(
                    hardware_rng.normal(
                        0.0,
                        float(
                            hardware_hierarchy.get("run_frequency_sd_ghz", 0.0)
                        ),
                    )
                )
            assert current_hardware is not None
            event_t1_sd = float(hardware_hierarchy.get("event_t1_log_sd", 0.0))
            event_t2_sd = float(hardware_hierarchy.get("event_t2_log_sd", 0.0))
            event_t1_factor = float(
                hardware_rng.lognormal(-0.5 * event_t1_sd**2, event_t1_sd)
            )
            event_t2_factor = float(
                hardware_rng.lognormal(-0.5 * event_t2_sd**2, event_t2_sd)
            )
            event_frequency_offset = float(
                hardware_rng.normal(
                    0.0,
                    float(
                        hardware_hierarchy.get("event_frequency_sd_ghz", 0.0)
                    ),
                )
            )
            hardware_t1 = (
                np.asarray(current_hardware["baseline_t1_us"], dtype=float)
                * run_t1_factor
                * event_t1_factor
            )
            hardware_t2 = (
                np.asarray(current_hardware["baseline_t2_us"], dtype=float)
                * run_t2_factor
                * event_t2_factor
            )
            hardware_t2 = np.minimum(hardware_t2, 2.0 * hardware_t1)
            hardware_frequency = (
                np.asarray(current_hardware["qubit_frequency_ghz"], dtype=float)
                + run_frequency_offset_ghz
                + event_frequency_offset
            )
            parameters.update(
                {
                    "qp_hardware_mode": hardware_mode,
                    "qp_device_id": int(current_hardware["device_id"]),
                    "qp_baseline_t1_us": float(hardware_t1.mean()),
                    "qp_baseline_t2_us": float(hardware_t2.mean()),
                    "qp_qubit_frequency_ghz": float(hardware_frequency.mean()),
                    "qp_al_gap_uev": float(current_hardware["al_gap_uev"]),
                    "qp_baseline_t1_min_us": float(hardware_t1.min()),
                    "qp_baseline_t1_max_us": float(hardware_t1.max()),
                    "qp_baseline_t2_min_us": float(hardware_t2.min()),
                    "qp_baseline_t2_max_us": float(hardware_t2.max()),
                    "qp_baseline_t1_by_qubit_us": ";".join(
                        f"{value:.8g}" for value in hardware_t1
                    ),
                    "qp_baseline_t2_by_qubit_us": ";".join(
                        f"{value:.8g}" for value in hardware_t2
                    ),
                    "qp_qubit_frequency_by_qubit_ghz": ";".join(
                        f"{value:.8g}" for value in hardware_frequency
                    ),
                    "_qp_baseline_t1_by_qubit_us": hardware_t1,
                    "_qp_baseline_t2_by_qubit_us": hardware_t2,
                    "_qp_qubit_frequency_by_qubit_ghz": hardware_frequency,
                }
            )
        raw_baseline = rng.lognormal(
            mean=0, sigma=noise["qubit_baseline_log_sd"], size=len(coords)
        )
        baseline = raw_baseline / raw_baseline.sum() * parameters["baseline_total_errors"]
        baseline = np.clip(baseline, 0.005, 0.6)
        event_sensitivity_sd = float(
            hierarchy.get("event_log_sd", noise["qubit_sensitivity_log_sd"])
        )
        event_sensitivity = rng.lognormal(
            mean=-event_sensitivity_sd**2 / 2,
            sigma=event_sensitivity_sd,
            size=len(coords),
        )
        sensitivity = device_sensitivity * event_sensitivity
        low_rank_field, low_rank_coefficients = sample_low_rank_spatial_field(
            rng, config.get("low_rank_spatial_variation", {}), len(coords)
        )
        sensitivity *= np.exp(low_rank_field)
        sensitivity /= sensitivity.mean()
        residual_config = config.get("spatial_residual", {"enabled": False})
        residual = time_correlated_residual(
            rng,
            coords,
            time_ms - parameters["event_onset_ms"],
            residual_config,
        )
        arrival_jitter_ms = spatial_log_field(
            rng,
            coords,
            parameters.get("arrival_jitter_sd_us", 0.0) / 1000,
            parameters.get("arrival_jitter_correlation_length_mm", 1.0),
        )
        if "target_peak_error_fraction" in parameters:
            sampled_target = float(parameters["target_peak_error_fraction"])
            drifting_baseline = drifting_baseline_probability(
                time_ms, baseline, parameters["baseline_drift_fraction"]
            )
            event_time_ms = time_ms - parameters["event_onset_ms"]
            early = (event_time_ms >= 0) & (event_time_ms < 5.0)
            baseline_peak = (
                float(drifting_baseline[early].mean(axis=1).max())
                if early.any()
                else float(baseline.mean())
            )
            parameters["sampled_target_peak_error_fraction"] = sampled_target
            parameters["target_peak_error_fraction"] = max(sampled_target, baseline_peak)
        probability_result = event_probability(
            time_ms,
            parameters,
            baseline,
            sensitivity,
            residual=residual,
            arrival_jitter_ms=arrival_jitter_ms,
            coords=coords,
            return_diagnostics=temporal_backend == "qp_ode",
        )
        if temporal_backend == "qp_ode":
            probability, physics = probability_result
            parameters.update(
                {
                    "qp_peak_density": float(np.max(physics["x_qp"])),
                    "qp_minimum_t1_us": float(np.min(physics["t1_us"])),
                    "qp_peak_generation_rate_per_us": float(
                        np.max(physics["qp_generation_rate_per_us"])
                    ),
                    "qp_relaxation_coefficient_per_us": float(
                        np.mean(physics["qp_relaxation_coefficient_per_us"])
                    ),
                    "qp_generation_event_scale": float(
                        physics["qp_generation_event_scale"]
                    ),
                }
            )
            if save_physics:
                for name, values in physics_arrays.items():
                    values[event] = np.asarray(physics[name], dtype=np.float32)
        else:
            probability = probability_result
        drifting_baseline = drifting_baseline_probability(
            time_ms, baseline, parameters["baseline_drift_fraction"]
        )
        event_time_ms = time_ms - parameters["event_onset_ms"]
        early = (event_time_ms >= 0) & (event_time_ms < 5.0)
        parameters["achieved_peak_error_fraction"] = (
            float(probability[early].mean(axis=1).max()) if early.any() else np.nan
        )
        missing = rng.random(len(coords)) < noise["missing_qubit_probability"]
        if missing.any():
            probability[:, missing] = drifting_baseline[:, missing]
        observations[event] = rng.binomial(1, probability).astype(np.uint8)
        baselines[event] = baseline
        if probabilities is not None:
            probabilities[event] = probability.astype(np.float16)
            if baseline_probabilities is not None:
                baseline_probabilities[event] = drifting_baseline.astype(np.float16)
        rows.append(
            {
                "event": event,
                "run_id": run_id,
                **{key: value for key, value in parameters.items() if not key.startswith("_")},
                "spatial_residual_enabled": bool(residual_config.get("enabled", False)),
                "low_rank_spatial_coefficients": ";".join(
                    f"{value:.8g}" for value in low_rank_coefficients
                ),
                "missing_qubits": ";".join(map(str, np.flatnonzero(missing))),
            }
        )
    result = (
        observations,
        probabilities,
        time_ms,
        pd.DataFrame(rows),
        baselines,
    )
    if include_baseline_probabilities:
        result = (*result, baseline_probabilities)
    if include_physics_arrays:
        result = (*result, physics_arrays)
    return result


def main() -> None:
    from .configuration import config_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path("simulator_base"))
    parser.add_argument("--profile", default=config_path("qp_ode_generic"))
    parser.add_argument("--output", default="simulation_results")
    parser.add_argument("--n-events", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--syndrome-config",
        help="optional syndrome observation-layer JSON; writes syndrome_events.npz",
    )
    parser.add_argument("--syndrome-seed", type=int)
    args = parser.parse_args()
    if args.syndrome_seed is not None and not args.syndrome_config:
        parser.error("--syndrome-seed requires --syndrome-config")
    config = load_config(Path(args.config), Path(args.profile) if args.profile else None)
    if args.n_events is not None:
        config["n_events"] = args.n_events
    if args.seed is not None:
        config["seed"] = args.seed
    syndrome_config = None
    if args.syndrome_config:
        from .syndrome import validate_syndrome_config

        syndrome_config = json.loads(Path(args.syndrome_config).read_text())
        validate_syndrome_config(syndrome_config, len(DEFAULT_COORDS))
        # The syndrome observation layer needs the latent response probabilities,
        # not only the already sampled RReCS measurement bits.
        config["output"]["save_probabilities"] = True
        if (
            syndrome_config.get("fault_probability_backend", "response_excess")
            == "t1_pauli"
        ):
            if config.get("temporal_model", {}).get("backend") != "qp_ode":
                raise ValueError(
                    "the t1_pauli syndrome backend requires a qp_ode simulator profile"
                )
            config["output"]["save_physics_diagnostics"] = True

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
        config,
        include_baseline_probabilities=True,
        include_physics_arrays=True,
    )
    arrays = {
        "observations": observations,
        "time_ms": time_ms,
        "coords": DEFAULT_COORDS,
        "baselines": baselines,
    }
    if probabilities is not None:
        arrays["probabilities"] = probabilities
        assert baseline_probabilities is not None
        arrays["baseline_probabilities"] = baseline_probabilities
    arrays.update(physics_arrays)
    np.savez_compressed(output / "simulated_events.npz", **arrays)
    parameters.to_csv(output / "true_parameters.csv", index=False)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.syndrome_config:
        from .syndrome import generate_syndromes, save_syndrome_output

        assert syndrome_config is not None
        syndrome_result, syndrome_metadata = generate_syndromes(
            probabilities,
            baseline_probabilities,
            DEFAULT_COORDS,
            syndrome_config,
            seed=args.syndrome_seed,
            time_ms=time_ms,
            t1_us=physics_arrays.get("t1_us"),
        )
        syndrome_metadata["source_simulation"] = str(output / "simulated_events.npz")
        syndrome_metadata["simulator_time_step_ms"] = float(np.median(np.diff(time_ms)))
        save_syndrome_output(output, syndrome_result, syndrome_metadata)
    print(
        f"generated {len(parameters)} events x {len(time_ms)} times x "
        f"{len(DEFAULT_COORDS)} qubits"
    )
    if args.syndrome_config:
        print(
            "generated syndrome layer: "
            f"X checks={syndrome_metadata['x_checks']}, "
            f"Z checks={syndrome_metadata['z_checks']}"
        )
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
