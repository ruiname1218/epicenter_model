"""CLI for the lightweight algebraic syndrome proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..configuration import config_path, load_json
from ..syndrome import generate_syndromes, save_syndrome_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate algebraic syndrome proxy data from a saved simulation"
    )
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=config_path("syndrome_proxy_t1_pauli"))
    parser.add_argument("--output", type=Path, default=Path("qp_ode_proxy_syndrome"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_json(args.config)
    archive = np.load(args.simulation)
    required = {"probabilities", "coords", "time_ms", "t1_us"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"simulation archive is missing required arrays: {sorted(missing)}")
    baseline = (
        archive["baseline_probabilities"]
        if "baseline_probabilities" in archive.files
        else archive["baselines"]
    )
    result, metadata = generate_syndromes(
        archive["probabilities"],
        baseline,
        archive["coords"],
        config,
        seed=args.seed,
        time_ms=archive["time_ms"],
        t1_us=archive["t1_us"],
    )
    metadata["source_simulation"] = str(args.simulation)
    save_syndrome_output(args.output, result, metadata)
    print(
        f"generated {metadata['events']} events x {metadata['rounds_per_event']} rounds; "
        f"X checks={metadata['x_checks']}, Z checks={metadata['z_checks']}"
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
