"""CLI for continuous QP-ODE field generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..api import run_simulation
from ..configuration import config_path
from ..layouts import DEFAULT_COORDS, load_coordinates
from ..simulator import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QP-ODE radiation events")
    parser.add_argument("--config", type=Path, default=config_path("simulator_base"))
    parser.add_argument("--profile", type=Path, default=config_path("qp_ode_generic"))
    parser.add_argument("--coords", type=Path, help="optional CSV/JSON/NPY coordinates in mm")
    parser.add_argument("--n-events", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("qp_ode_output"))
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    if args.n_events is not None:
        config["n_events"] = args.n_events
    if args.seed is not None:
        config["seed"] = args.seed
    coordinates = DEFAULT_COORDS if args.coords is None else load_coordinates(args.coords)
    result = run_simulation(config, coords_mm=coordinates)
    result.save(args.output)
    print(
        f"generated {len(result.parameters)} events x {len(result.time_ms)} samples x "
        f"{len(result.coords_mm)} qubits"
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
