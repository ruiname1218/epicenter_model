"""Check the local-field integration against a finer time grid, not hardware truth."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator import run_simulation
from qp_ode_simulator.configuration import load_json
from qp_ode_simulator.localization_study import paired_configuration
from qp_ode_simulator.stim_qec import average_t1_over_rounds, build_stim_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    manifest = load_json(args.study_dir / "study_manifest.json")
    base, stim, study = manifest["simulator"], manifest["stim"], manifest["study"]
    coords = build_stim_layout(stim).physical_coords_mm
    starts = stim["start_time_ms"] + np.arange(stim["rounds"]) * stim["round_duration_ms"]
    rows = []
    # First balanced cycle: 12 source families, all strength/range settings.
    for family in range(12):
        for strength in range(len(study["generation_per_us"])):
            for reach in range(len(study["range_scales"])):
                resolved = paired_configuration(base, study, family, strength, reach)
                rates = {}
                for step_us in (2.0, 1.0, 0.5):
                    config = copy.deepcopy(resolved)
                    config["time"]["dt_ms"] = step_us / 1000
                    result = run_simulation(config, coords_mm=coords)
                    effective = average_t1_over_rounds(result.physics["t1_us"], result.time_ms,
                                                       starts, stim["round_duration_ms"])
                    baseline = 1 / result.physics["t1_us"][:, :1, :]
                    rates[step_us] = 1 / effective - baseline
                ref = rates[0.5]
                norm = max(float(np.sqrt(np.mean(ref**2))), 1e-12)
                peak = max(float(np.max(np.abs(ref))), 1e-12)
                for step_us in (2.0, 1.0):
                    difference = rates[step_us] - ref
                    rows.append({"family": family, "strength": strength, "reach": reach,
                                 "step_us": step_us, "reference_step_us": 0.5,
                                 "relative_rms_rate_error": float(np.sqrt(np.mean(difference**2)) / norm),
                                 "peak_normalized_max_rate_error": float(np.max(np.abs(difference)) / peak)})
        print(f"resolution check {family+1}/12 families", flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.groupby("step_us")[["relative_rms_rate_error", "peak_normalized_max_rate_error"]].agg(["median", "max"]).to_string())


if __name__ == "__main__":
    main()
