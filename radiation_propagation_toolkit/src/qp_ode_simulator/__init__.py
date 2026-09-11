"""QP-ODE radiation propagation and superconducting-QEC toolkit."""

from .api import SimulationResult, StimPipelineResult, run_simulation, run_stim_pipeline
from .configuration import config_path, load_default_simulator_config, load_json
from .layouts import DEFAULT_COORDS, load_coordinates, rectangular_layout
from .simulator import (
    qp_dephasing_rate_per_us,
    qp_relaxation_coefficient_per_us,
    solve_qp_dynamics,
)
from .syndrome import generate_syndromes, t1_to_pauli_probabilities

__all__ = [
    "DEFAULT_COORDS",
    "SimulationResult",
    "StimPipelineResult",
    "config_path",
    "generate_syndromes",
    "load_coordinates",
    "load_default_simulator_config",
    "load_json",
    "qp_dephasing_rate_per_us",
    "qp_relaxation_coefficient_per_us",
    "rectangular_layout",
    "run_simulation",
    "run_stim_pipeline",
    "solve_qp_dynamics",
    "t1_to_pauli_probabilities",
]

__version__ = "0.1.0"
