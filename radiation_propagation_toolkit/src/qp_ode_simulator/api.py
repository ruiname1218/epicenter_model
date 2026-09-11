"""Small, stable public API over the simulator and QEC implementation modules."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import simulator as simulator_module
from . import stim_qec as stim_module
from .layouts import DEFAULT_COORDS, validate_coordinates
from .simulator import simulate
from .stim_qec import (
    StimLayout,
    build_stim_layout,
    collect_run_manifest,
    generate_circuit_syndromes,
    save_circuit_syndrome_output,
    uses_fixed_hardware,
    uses_peak_matched_qp,
)


@dataclass
class SimulationResult:
    """In-memory result of one QP-ODE event simulation."""

    observations: np.ndarray
    probabilities: np.ndarray | None
    time_ms: np.ndarray
    parameters: pd.DataFrame
    baselines: np.ndarray
    baseline_probabilities: np.ndarray | None
    physics: dict[str, np.ndarray]
    coords_mm: np.ndarray
    config: dict

    def save(self, output: str | Path) -> Path:
        """Write a portable NPZ archive, event labels, and resolved config."""
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "observations": self.observations,
            "time_ms": self.time_ms,
            "coords": self.coords_mm,
            "baselines": self.baselines,
            **self.physics,
        }
        if self.probabilities is not None:
            arrays["probabilities"] = self.probabilities
        if self.baseline_probabilities is not None:
            arrays["baseline_probabilities"] = self.baseline_probabilities
        np.savez_compressed(destination / "simulated_events.npz", **arrays)
        self.parameters.to_csv(destination / "true_parameters.csv", index=False)
        (destination / "resolved_config.json").write_text(
            json.dumps(self.config, indent=2) + "\n"
        )
        return destination


@dataclass
class StimPipelineResult:
    """QP field, circuit-level detector samples, and the generated Stim layout."""

    simulation: SimulationResult
    syndrome_arrays: dict[str, np.ndarray]
    syndrome_metadata: dict
    layout: StimLayout
    stim_config: dict

    def save(self, output: str | Path, *, argv: list[str] | None = None) -> Path:
        destination = self.simulation.save(output)
        (destination / "resolved_stim_config.json").write_text(
            json.dumps(self.stim_config, indent=2) + "\n"
        )
        manifest = collect_run_manifest(
            Path(__file__).resolve().parents[2],
            argv=list(sys.argv if argv is None else argv),
            source_paths=[Path(simulator_module.__file__), Path(stim_module.__file__), Path(__file__)],
            simulator_config=self.simulation.config,
            stim_config=self.stim_config,
            template_circuit=self.layout.circuit,
        )
        (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save_circuit_syndrome_output(
            destination,
            self.syndrome_arrays,
            self.syndrome_metadata,
            template_circuit=self.layout.circuit,
        )
        return destination


def run_simulation(
    config: dict,
    *,
    coords_mm: np.ndarray = DEFAULT_COORDS,
) -> SimulationResult:
    """Run a simulation without mutating the caller's configuration."""
    resolved = copy.deepcopy(config)
    coordinates = validate_coordinates(coords_mm)
    values = simulate(
        resolved,
        coords=coordinates,
        include_baseline_probabilities=True,
        include_physics_arrays=True,
    )
    observations, probabilities, time_ms, parameters, baselines, baseline, physics = values
    return SimulationResult(
        observations=observations,
        probabilities=probabilities,
        time_ms=time_ms,
        parameters=parameters,
        baselines=baselines,
        baseline_probabilities=baseline,
        physics=physics,
        coords_mm=coordinates,
        config=resolved,
    )


def run_stim_pipeline(
    simulator_config: dict,
    stim_config: dict,
    *,
    allow_peak_matched_qp: bool = False,
    allow_varying_hardware: bool = False,
) -> StimPipelineResult:
    """Run QP-ODE -> T1/T2 -> Stim -> detector/logical samples end to end."""
    resolved_simulator = copy.deepcopy(simulator_config)
    resolved_stim = copy.deepcopy(stim_config)
    if resolved_simulator.get("temporal_model", {}).get("backend") != "qp_ode":
        raise ValueError("Stim pipeline requires temporal_model.backend='qp_ode'")
    peak_matched = uses_peak_matched_qp(resolved_simulator)
    if peak_matched and not allow_peak_matched_qp:
        raise ValueError("peak-matched QP profiles require allow_peak_matched_qp=True")
    fixed_hardware = uses_fixed_hardware(resolved_simulator)
    if not fixed_hardware and not allow_varying_hardware:
        raise ValueError("QEC generation requires fixed hardware unless explicitly overridden")
    resolved_simulator.setdefault("output", {})["save_probabilities"] = True
    resolved_simulator["output"]["save_physics_diagnostics"] = True

    layout = build_stim_layout(resolved_stim)
    simulation = run_simulation(resolved_simulator, coords_mm=layout.physical_coords_mm)
    event_onset = simulation.parameters["event_onset_ms"].to_numpy(dtype=np.float64)
    arrays, metadata = generate_circuit_syndromes(
        simulation.physics["t1_us"],
        simulation.time_ms,
        layout,
        resolved_stim,
        t2_us=simulation.physics.get("t2_us"),
        frequency_shift_ghz=simulation.physics.get("frequency_shift_ghz"),
        event_onset_ms=event_onset,
        # The simulator initializes QP density to zero and requires onset within
        # its time interval, so the first field sample is the hardware baseline.
        baseline_t1_us=simulation.physics["t1_us"][:, 0, :],
        baseline_t2_us=simulation.physics["t2_us"][:, 0, :],
    )
    event_is_control = simulation.parameters["is_control"].to_numpy(dtype=np.uint8)
    arrays["event_onset_ms"] = event_onset
    arrays["event_is_control"] = event_is_control
    metadata.update(
        {
            "control_events": int(event_is_control.sum()),
            "qp_peak_matched": peak_matched,
            "hardware_fixed_across_events": fixed_hardware,
            "particle_energy_conservation_modeled": False,
            "qp_generation_semantics": (
                "local density-like source proxy at each circuit-qubit coordinate; "
                "the sum over qubits is not deposited particle energy"
            ),
        }
    )
    return StimPipelineResult(simulation, arrays, metadata, layout, resolved_stim)
