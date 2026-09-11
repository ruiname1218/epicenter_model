import copy
import json

import numpy as np
import pandas as pd
import pytest

from qp_ode_simulator import load_default_simulator_config
from qp_ode_simulator.configuration import load_json
from qp_ode_simulator.localization_study import (
    family_splits, paired_configuration, prefix_syndromes, summarize_predictions, validate_study,
)
from qp_ode_simulator.stim_qec import ROLE_DATA, build_stim_layout, generate_circuit_syndromes


def test_study_summary_serializes_numpy_group_keys():
    frame = pd.DataFrame({
        "family": [0, 1], "strength": [0, 0], "reach": [1, 1],
        "window_ms": [1.024, 1.024], "bin_width_us": [64., 64.],
        "protocol": ["all_conditions"] * 2, "x_mm": [0., 1.], "y_mm": [0., 0.],
        "predicted_x_mm": [1., 1.], "predicted_y_mm": [0., 0.],
        "error_mm": [1., 0.], "center_error_mm": [0.5, 0.5],
        "centroid_error_mm": [1., 1.],
    })
    result = json.loads(json.dumps(summarize_predictions(frame, 2)))
    assert result["groups"][0]["mean_error_mm"] == 0.5
    assert result["groups"][0]["independent_families"] == 2


@pytest.mark.parametrize("timing", ["gate_slices", "round_boundary"])
def test_data_target_matches_explicit_baseline_ancillas_including_decoder(timing):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    config = load_json("examples/localization_stim.json")
    config.update(rounds=3, shots_per_event=8)
    config["decoder"]["enabled"] = True
    config["radiation_channel"].update(targets="data", timing=timing)
    layout = build_stim_layout(config)
    times = np.arange(5) * 0.001
    field = np.full((1, 5, len(layout.qubit_ids)), 10.0, dtype=np.float32)
    field[:, 0] = 100
    original = field.copy()
    baseline = np.full((1, len(layout.qubit_ids)), 100.0)
    actual, _ = generate_circuit_syndromes(
        field, times, layout, config, t2_us=2 * field,
        baseline_t1_us=baseline, baseline_t2_us=2 * baseline,
    )
    reference = field.copy()
    reference[:, :, layout.roles != ROLE_DATA] = 100
    config["radiation_channel"]["targets"] = "all"
    expected, _ = generate_circuit_syndromes(reference, times, layout, config, t2_us=2 * reference)
    for key in ("detector_events", "measurement_records", "decoder_predictions", "pauli_x_probability", "round_t1_us"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert np.all(actual["pauli_x_probability"][:, :, layout.roles != ROLE_DATA] > 0)
    np.testing.assert_array_equal(field, original)


def test_data_target_requires_a_baseline_without_pre_event_samples():
    pytest.importorskip("stim")
    config = load_json("examples/localization_stim.json")
    config.update(rounds=3)
    config["radiation_channel"]["targets"] = "data"
    layout = build_stim_layout(config)
    field = np.full((1, 5, len(layout.qubit_ids)), 10.0)
    with pytest.raises(ValueError, match="requires baseline"):
        generate_circuit_syndromes(field, np.arange(5) * .001, layout, config)


def test_paired_study_preserves_source_seed_and_group_splits():
    config = load_json("examples/localization_study.json")
    base = load_default_simulator_config()
    first = paired_configuration(base, config, 0, 0, 0)
    second = paired_configuration(base, config, 0, 2, 1)
    assert first["seed"] == second["seed"]
    assert first["device_seed"] == second["device_seed"]
    for key in ("epicenter", "shape", "propagation"):
        assert first[key] == second[key]
    splits = family_splits(config)
    assert len(set(splits["train"]) & set(splits["test"])) == 0
    for ids in splits.values():
        cells = set()
        for family in ids:
            event = paired_configuration(base, config, family, 0, 0)
            cells.add((event["shape"]["geometry_choices"][0], event["propagation"]["law_choices"][0],
                       next(iter(event["epicenter"]["region_probabilities"]))))
        assert len(cells) == 12
    bad = copy.deepcopy(config)
    bad["families"] = 35
    with pytest.raises(ValueError):
        validate_study(bad)


def test_prefix_uses_existing_detectors_without_future_or_final_boundary():
    pytest.importorskip("stim")
    from qp_ode_simulator import run_stim_pipeline
    base = load_default_simulator_config()
    base.update(n_events=1)
    base["time"] = {"start_ms": 0.0, "end_ms": .012, "dt_ms": .001}
    base["event_timing"] = {"onset_time_ms": .002, "control_fraction": 0.0}
    config = load_json("examples/localization_stim.json")
    config.update(rounds=8, shots_per_event=1)
    data = run_stim_pipeline(base, config).syndrome_arrays
    short = prefix_syndromes(data, .004)
    assert len(short["round_time_ms"]) == 4
    assert short["detector_coords"][:, 2].max() == 3
    selection = data["detector_coords"][:, 2] < 4
    np.testing.assert_array_equal(short["detector_events"], data["detector_events"][:, :, selection])
    with pytest.raises(ValueError):
        prefix_syndromes(data, .009)
