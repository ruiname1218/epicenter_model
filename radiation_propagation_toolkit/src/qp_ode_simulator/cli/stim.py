"""CLI for the end-to-end QP-ODE to Stim pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..api import run_stim_pipeline
from ..configuration import config_path, load_json
from ..simulator import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate circuit-level syndrome data from QP-ODE events"
    )
    parser.add_argument("--config", type=Path, default=config_path("simulator_base"))
    parser.add_argument("--profile", type=Path, default=config_path("qp_ode_generic"))
    parser.add_argument("--stim-config", type=Path, default=config_path("stim_surface_code_d3"))
    parser.add_argument("--n-events", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--syndrome-seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("qp_ode_stim_output"))
    parser.add_argument("--allow-peak-matched-qp", action="store_true")
    parser.add_argument("--allow-varying-hardware", action="store_true")
    args = parser.parse_args()

    simulator_config = load_config(args.config, args.profile)
    stim_config = load_json(args.stim_config)
    if args.n_events is not None:
        simulator_config["n_events"] = args.n_events
    if args.seed is not None:
        simulator_config["seed"] = args.seed
    if args.syndrome_seed is not None:
        stim_config["seed"] = args.syndrome_seed
    result = run_stim_pipeline(
        simulator_config,
        stim_config,
        allow_peak_matched_qp=args.allow_peak_matched_qp,
        allow_varying_hardware=args.allow_varying_hardware,
    )
    result.save(args.output)
    print(
        f"generated {len(result.simulation.parameters)} events on "
        f"{len(result.layout.qubit_ids)} active circuit qubits"
    )
    print(
        f"Stim output: {stim_config['rounds']} rounds, "
        f"{stim_config.get('shots_per_event', 1)} shots/event, "
        f"{result.layout.circuit.num_detectors} detectors"
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
