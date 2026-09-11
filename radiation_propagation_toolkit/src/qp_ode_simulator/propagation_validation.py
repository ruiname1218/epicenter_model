"""Synthetic ground-truth validation for the propagation field.

This benchmark checks whether a simple reference fit can recover the parameters
used by :func:`qp_ode_simulator.simulator.event_probability`.  It is deliberately
field-level: it validates the configured wavefront and radial attenuation laws,
not their agreement with a particular radiation experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .simulator import event_probability


def validate_config(config: dict) -> None:
    """Reject malformed benchmark settings before running an expensive sweep."""
    required = (
        "seed",
        "trials_per_law",
        "layout",
        "time",
        "truth_ranges",
        "field",
        "nuisance",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("propagation validation config is missing: " + ", ".join(missing))
    if int(config["trials_per_law"]) < 1:
        raise ValueError("trials_per_law must be positive")
    if int(config["layout"]["points_per_axis"]) < 3:
        raise ValueError("layout.points_per_axis must be at least 3")
    layout_extent = float(config["layout"]["half_extent_mm"])
    if layout_extent <= 0:
        raise ValueError("layout.half_extent_mm must be positive")
    time = config["time"]
    if float(time["dt_ms"]) <= 0 or float(time["end_ms"]) <= float(time["start_ms"]):
        raise ValueError("time must have positive dt_ms and end_ms > start_ms")
    for name in (
        "ballistic_speed_m_per_s",
        "diffusion_coefficient_mm2_per_ms",
        "lambda_mm",
    ):
        lower, upper = map(float, config["truth_ranges"][name])
        if not 0 < lower < upper:
            raise ValueError(f"truth_ranges.{name} must be positive and increasing")
    center_extent = float(config["truth_ranges"]["epicenter_half_extent_mm"])
    if not 0 <= center_extent <= layout_extent:
        raise ValueError("epicenter_half_extent_mm must lie inside the layout")
    field = config["field"]
    if float(field["front_width_ms"]) <= 0 or float(field["beta"]) <= 0:
        raise ValueError("front_width_ms and beta must be positive")
    if not 0 < float(field["maximum_amplitude"]) < 1:
        raise ValueError("maximum_amplitude must lie in (0, 1)")
    if not 0 <= float(field["baseline_probability"]) < 1:
        raise ValueError("baseline_probability must lie in [0, 1)")
    if float(config["nuisance"]["arrival_jitter_sd_us"]) < 0:
        raise ValueError("arrival_jitter_sd_us must be non-negative")
    if float(config["nuisance"]["qubit_response_log_sd"]) < 0:
        raise ValueError("qubit_response_log_sd must be non-negative")


def _layout(config: dict) -> np.ndarray:
    extent = float(config["layout"]["half_extent_mm"])
    axis = np.linspace(-extent, extent, int(config["layout"]["points_per_axis"]))
    return np.asarray([(row, column) for row in axis for column in axis], dtype=float)


def _time_axis(config: dict) -> np.ndarray:
    settings = config["time"]
    start = float(settings["start_ms"])
    stop = float(settings["end_ms"])
    step = float(settings["dt_ms"])
    return np.arange(start, stop + step / 2, step)


def _parameters(
    config: dict,
    *,
    law: str,
    propagation_value: float,
    lambda_mm: float,
    epicenter: np.ndarray,
) -> dict:
    field = config["field"]
    return {
        "event_onset_ms": float(field["event_onset_ms"]),
        "peak_time_ms": 100.0,
        "rise_time_ms": 1.0,
        "prompt_fraction": 1.0,
        "maximum_amplitude": float(field["maximum_amplitude"]),
        "decay_time_ms": 1000.0,
        "spread_time_ms": 1.0,
        "initial_lambda_mm": lambda_mm,
        "maximum_lambda_mm": lambda_mm,
        "beta_transition_ms": 1.0,
        "initial_beta": float(field["beta"]),
        "late_beta": float(field["beta"]),
        "epicenter_row": float(epicenter[0]),
        "epicenter_col": float(epicenter[1]),
        "axis_ratio": 1.0,
        "angle_degrees": 0.0,
        "source_count": 1,
        "reflection_enabled": False,
        "propagation_law": law,
        "apparent_speed_m_per_s": (
            propagation_value if law == "ballistic" else np.nan
        ),
        "diffusion_coefficient_mm2_per_ms": (
            propagation_value if law == "diffusive" else np.nan
        ),
        "front_width_ms": float(field["front_width_ms"]),
        "maximum_distance_mm": 100.0,
        "cutoff_width_mm": 0.2,
        "halo_enabled": False,
        "global_spatial_mixing_fraction": 0.0,
        "baseline_drift_fraction": 0.0,
        "temporal_backend": "empirical",
        "is_control": False,
    }


def _half_height_arrivals(
    time_ms: np.ndarray, response: np.ndarray, plateau_samples: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    plateau = response[-plateau_samples:].mean(axis=0)
    arrivals = np.empty(response.shape[1], dtype=float)
    for qubit in range(response.shape[1]):
        centered = response[:, qubit] - 0.5 * plateau[qubit]
        crossings = np.flatnonzero(centered >= 0)
        if not len(crossings) or crossings[0] == 0:
            raise RuntimeError("benchmark time window does not bracket a wavefront")
        upper = int(crossings[0])
        lower = upper - 1
        y0, y1 = centered[lower], centered[upper]
        fraction = -y0 / (y1 - y0)
        arrivals[qubit] = time_ms[lower] + fraction * (
            time_ms[upper] - time_ms[lower]
        )
    return arrivals, plateau


def _fit_wavefront(
    coords: np.ndarray, arrivals_ms: np.ndarray, law: str
) -> tuple[np.ndarray, float, float, float, bool]:
    initial_scale = 25.0 if law == "ballistic" else 10.0
    center_bound = float(np.max(np.abs(coords)))

    def residual(parameters: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(coords - parameters[:2], axis=1)
        scale = np.exp(parameters[2])
        delay = distance / scale if law == "ballistic" else distance**2 / scale
        return parameters[3] + delay - arrivals_ms

    earliest = coords[int(np.argmin(arrivals_ms))]
    weights = np.exp(-(arrivals_ms - arrivals_ms.min()) / 0.08)
    weighted = np.sum(coords * weights[:, None], axis=0) / weights.sum()
    starts = (earliest, weighted, np.zeros(2))
    fits = []
    for center in starts:
        initial = np.asarray(
            [center[0], center[1], np.log(initial_scale), arrivals_ms.min() - 0.02]
        )
        fits.append(
            least_squares(
                residual,
                initial,
                bounds=(
                    [-center_bound, -center_bound, np.log(3.0), -0.2],
                    [center_bound, center_bound, np.log(100.0), 0.5],
                ),
                loss="soft_l1",
                f_scale=0.03,
            )
        )
    fit = min(fits, key=lambda item: float(np.sum(residual(item.x) ** 2)))
    arrival_rmse_ms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    return (
        fit.x[:2],
        float(np.exp(fit.x[2])),
        float(fit.x[3]),
        arrival_rmse_ms,
        bool(fit.success),
    )


def _fit_lambda(
    coords: np.ndarray, center: np.ndarray, plateau: np.ndarray, beta: float
) -> float:
    distance = np.linalg.norm(coords - center, axis=1)
    slope, _ = np.polyfit(distance**beta, np.log(np.maximum(plateau, 1e-12)), 1)
    if slope >= 0:
        return np.nan
    return float((-1.0 / slope) ** (1.0 / beta))


def _r_squared(truth: np.ndarray, estimate: np.ndarray) -> float:
    residual = float(np.sum((estimate - truth) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else np.nan


def _law_summary(frame: pd.DataFrame) -> dict:
    truth = frame["propagation_value_true"].to_numpy(dtype=float)
    estimate = frame["propagation_value_estimated"].to_numpy(dtype=float)
    lambda_truth = frame["lambda_true_mm"].to_numpy(dtype=float)
    lambda_estimate = frame["lambda_estimated_mm"].to_numpy(dtype=float)
    return {
        "trials": int(len(frame)),
        "propagation_median_absolute_percent_error": float(
            np.median(np.abs(estimate / truth - 1.0)) * 100
        ),
        "propagation_mean_absolute_percent_error": float(
            np.mean(np.abs(estimate / truth - 1.0)) * 100
        ),
        "propagation_r_squared": _r_squared(truth, estimate),
        "lambda_median_absolute_percent_error": float(
            np.median(np.abs(lambda_estimate / lambda_truth - 1.0)) * 100
        ),
        "lambda_mean_absolute_percent_error": float(
            np.mean(np.abs(lambda_estimate / lambda_truth - 1.0)) * 100
        ),
        "epicenter_median_error_mm": float(frame["epicenter_error_mm"].median()),
        "epicenter_p90_error_mm": float(frame["epicenter_error_mm"].quantile(0.9)),
        "arrival_rmse_median_us": float(frame["arrival_rmse_us"].median()),
        "successful_fit_fraction": float(frame["fit_success"].mean()),
    }


def run_validation(config: dict) -> tuple[pd.DataFrame, dict]:
    """Run a seeded ballistic/diffusive parameter-recovery benchmark."""
    validate_config(config)
    rng = np.random.default_rng(int(config["seed"]))
    coords = _layout(config)
    time_ms = _time_axis(config)
    baseline_level = float(config["field"]["baseline_probability"])
    baseline = np.full(len(coords), baseline_level)
    beta = float(config["field"]["beta"])
    center_limit = float(config["truth_ranges"]["epicenter_half_extent_mm"])
    jitter_sd_ms = float(config["nuisance"]["arrival_jitter_sd_us"]) / 1000.0
    sensitivity_sd = float(config["nuisance"]["qubit_response_log_sd"])
    rows = []

    settings = (
        ("ballistic", "ballistic_speed_m_per_s"),
        ("diffusive", "diffusion_coefficient_mm2_per_ms"),
    )
    for law, range_name in settings:
        lower, upper = map(float, config["truth_ranges"][range_name])
        lambda_lower, lambda_upper = map(float, config["truth_ranges"]["lambda_mm"])
        for trial in range(int(config["trials_per_law"])):
            propagation_truth = float(rng.uniform(lower, upper))
            lambda_truth = float(rng.uniform(lambda_lower, lambda_upper))
            epicenter_truth = rng.uniform(-center_limit, center_limit, size=2)
            sensitivity = rng.lognormal(
                mean=-0.5 * sensitivity_sd**2,
                sigma=sensitivity_sd,
                size=len(coords),
            )
            arrival_jitter_ms = rng.normal(0.0, jitter_sd_ms, size=len(coords))
            arrival_jitter_ms -= arrival_jitter_ms.mean()
            parameters = _parameters(
                config,
                law=law,
                propagation_value=propagation_truth,
                lambda_mm=lambda_truth,
                epicenter=epicenter_truth,
            )
            probability = event_probability(
                time_ms,
                parameters,
                baseline,
                sensitivity,
                arrival_jitter_ms=arrival_jitter_ms,
                coords=coords,
            )
            response = (probability - baseline[None, :]) / (1.0 - baseline[None, :])
            arrivals, plateau = _half_height_arrivals(time_ms, response)
            (
                center_estimate,
                propagation_estimate,
                onset_estimate,
                arrival_rmse,
                success,
            ) = _fit_wavefront(coords, arrivals, law)
            lambda_estimate = _fit_lambda(coords, center_estimate, plateau, beta)
            rows.append(
                {
                    "law": law,
                    "trial": trial,
                    "propagation_value_true": propagation_truth,
                    "propagation_value_estimated": propagation_estimate,
                    "lambda_true_mm": lambda_truth,
                    "lambda_estimated_mm": lambda_estimate,
                    "epicenter_row_true_mm": float(epicenter_truth[0]),
                    "epicenter_col_true_mm": float(epicenter_truth[1]),
                    "epicenter_row_estimated_mm": float(center_estimate[0]),
                    "epicenter_col_estimated_mm": float(center_estimate[1]),
                    "epicenter_error_mm": float(
                        np.linalg.norm(center_estimate - epicenter_truth)
                    ),
                    "onset_estimated_ms": onset_estimate,
                    "arrival_rmse_us": arrival_rmse * 1000.0,
                    "fit_success": success and np.isfinite(lambda_estimate),
                }
            )

    frame = pd.DataFrame(rows)
    summary = {
        "schema_version": 1,
        "benchmark": "synthetic propagation ground-truth recovery",
        "seed": int(config["seed"]),
        "deterministic_given_seed": True,
        "field_level": True,
        "binary_shot_noise_included": False,
        "layout_qubits": int(len(coords)),
        "arrival_jitter_sd_us": float(config["nuisance"]["arrival_jitter_sd_us"]),
        "qubit_response_log_sd": sensitivity_sd,
        "by_law": {
            law: _law_summary(group.reset_index(drop=True))
            for law, group in frame.groupby("law", sort=False)
        },
        "claim_boundary": (
            "This self-consistency benchmark validates implementation and parameter "
            "recoverability under controlled nuisance variation. It is not evidence "
            "that the phenomenological field matches radiation transport in hardware."
        ),
    }
    return frame, summary


def make_figure(
    frame: pd.DataFrame, output: str | Path, summary: dict | None = None
) -> Path:
    """Render the four-panel recovery summary used in the documentation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    colors = {"ballistic": "#0072B2", "diffusive": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))

    for axis, law, label in (
        (axes[0, 0], "ballistic", "Apparent speed (m/s)"),
        (axes[0, 1], "diffusive", "Diffusion coefficient (mm²/ms)"),
    ):
        group = frame[frame["law"] == law]
        truth = group["propagation_value_true"].to_numpy()
        estimate = group["propagation_value_estimated"].to_numpy()
        bounds = [min(truth.min(), estimate.min()), max(truth.max(), estimate.max())]
        axis.plot(
            bounds,
            bounds,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="ideal",
        )
        axis.scatter(truth, estimate, s=30, alpha=0.75, color=colors[law])
        mape = np.mean(np.abs(estimate / truth - 1.0)) * 100
        r_squared = _r_squared(truth, estimate)
        axis.text(
            0.04,
            0.94,
            f"MAPE = {mape:.1f}%\n$R^2$ = {r_squared:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axis.set_xlabel(f"Configured {label.lower()}")
        axis.set_ylabel(f"Recovered {label.lower()}")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, loc="lower right")

    axis = axes[1, 0]
    all_values = []
    for law in ("ballistic", "diffusive"):
        group = frame[frame["law"] == law]
        truth = group["lambda_true_mm"].to_numpy()
        estimate = group["lambda_estimated_mm"].to_numpy()
        all_values.extend(truth)
        all_values.extend(estimate)
        axis.scatter(
            truth,
            estimate,
            s=27,
            alpha=0.65,
            color=colors[law],
            label=law,
        )
    bounds = [min(all_values), max(all_values)]
    axis.plot(
        bounds,
        bounds,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="ideal",
    )
    lambda_mape = np.mean(
        np.abs(frame["lambda_estimated_mm"] / frame["lambda_true_mm"] - 1.0)
    ) * 100
    axis.text(
        0.04,
        0.94,
        f"Combined MAPE = {lambda_mape:.1f}%",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    axis.set_xlabel("Configured radial scale λ (mm)")
    axis.set_ylabel("Recovered radial scale λ (mm)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower right")

    axis = axes[1, 1]
    for law in ("ballistic", "diffusive"):
        errors = np.sort(
            frame.loc[frame["law"] == law, "epicenter_error_mm"].to_numpy()
        )
        cumulative = np.arange(1, len(errors) + 1) / len(errors)
        axis.step(
            errors,
            cumulative,
            where="post",
            linewidth=2.2,
            color=colors[law],
            label=law,
        )
        axis.axvline(np.median(errors), color=colors[law], alpha=0.45, linestyle=":")
    axis.set_xlabel("Epicenter localization error (mm)")
    axis.set_ylabel("Fraction of trials ≤ error")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "Propagation self-consistency: recovery from a noisy synthetic field",
        fontsize=14,
        fontweight="bold",
    )
    trials = int(frame.groupby("law").size().min())
    jitter = None if summary is None else summary.get("arrival_jitter_sd_us")
    scatter = None if summary is None else summary.get("qubit_response_log_sd")
    nuisance_text = (
        f"{jitter:g} µs RMS arrival jitter and {scatter:.0%} log-response scatter"
        if jitter is not None and scatter is not None
        else "controlled arrival jitter and qubit-response scatter"
    )
    fig.text(
        0.5,
        0.018,
        f"{trials} trials/law; {nuisance_text}. "
        "Field-level implementation test, not hardware validation.",
        ha="center",
        fontsize=9,
    )
    # Reserve enough room for both the lower x-axis labels and the scope note.
    # This matters for the high-resolution PNG embedded in the README.
    fig.tight_layout(rect=(0, 0.105, 1, 0.95))
    path = destination / "propagation_ground_truth_recovery.png"
    fig.savefig(path, dpi=220)
    fig.savefig(destination / "propagation_ground_truth_recovery.pdf")
    plt.close(fig)
    return path


def save_validation(frame: pd.DataFrame, summary: dict, output: str | Path) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "propagation_ground_truth_recovery.csv", index=False)
    (destination / "propagation_ground_truth_recovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    make_figure(frame, destination, summary)
    return destination
