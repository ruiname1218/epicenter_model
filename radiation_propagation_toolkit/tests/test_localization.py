import copy
import json

import numpy as np
import pandas as pd
import pytest

from qp_ode_simulator.localization import (
    GEOMETRY_KEYS,
    extract_features,
    feature_specification,
    load_syndromes,
    predict_epicenters,
    split_events,
    train_localizer,
)


def synthetic_data(events=96, shots=2, rounds=128):
    """Observable toy problem, independent of the radiation generator."""
    rng = np.random.default_rng(13)
    grid = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    coords = np.asarray([[*site, t] for t in range(rounds + 1) for site in grid])
    truth = rng.uniform(-0.8, 0.8, (events, 2))
    physical = 2 * grid - 1
    distance = np.linalg.norm(truth[:, None, :] - physical[None, :, :], axis=2)
    rates = 0.05 + 0.6 * np.exp(-distance**2 / 2)
    probabilities = np.tile(rates, (1, rounds + 1))
    bits = (rng.random((events, shots, len(coords))) < probabilities[:, None, :]).astype(np.uint8)
    data = {
        "detector_events": bits,
        "detector_coords": coords,
        "circuit_grid_coords": grid,
        "circuit_physical_coords_mm": physical,
        "circuit_qubit_ids": np.arange(4),
        "circuit_qubit_roles": np.ones(4, dtype=np.uint8),
        "round_start_time_ms": np.arange(rounds) * 0.001,
        "round_time_ms": (np.arange(rounds) + 0.5) * 0.001,
        "gate_slice_duration_ms": np.full((rounds, 1), 0.001),
        "event_is_control": np.zeros(events, dtype=np.uint8),
    }
    return data, truth


def save_training_data(path, data, truth):
    path.mkdir()
    np.savez_compressed(path / "stim_syndrome_events.npz", **data)
    pd.DataFrame({
        "event": np.arange(len(truth)),
        "epicenter_row": truth[:, 0], "epicenter_col": truth[:, 1],
        "is_control": data["event_is_control"].astype(bool),
        "source_count": np.ones(len(truth), dtype=int),
        "geometry": "circular",
    }).to_csv(path / "true_parameters.csv", index=False)


def test_event_split_never_separates_shots():
    ids = np.arange(50)
    first, second = split_events(ids, 3), split_events(ids, 3)
    sample_events = np.repeat(ids, 4)
    membership = np.zeros(len(sample_events), dtype=int)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        mask = np.isin(sample_events, first[name])
        membership += mask
        assert np.all(mask.reshape(-1, 4).sum(axis=1) % 4 == 0)
    np.testing.assert_array_equal(membership, 1)


def test_pooling_counts_and_excludes_boundaries():
    data, _ = synthetic_data(events=1, shots=1, rounds=5)
    data["detector_events"][:] = 0
    coords = data["detector_coords"]
    active = ((coords[:, 0] == 0) & (coords[:, 1] == 0) & (coords[:, 2] == 1))
    boundary = (coords[:, 2] == 0) | (coords[:, 2] == 5)
    data["detector_events"][0, 0, active | boundary] = 1
    spec = feature_specification(data, bins=2)
    _, rates = extract_features(data, spec)
    expected = np.zeros((1, 4, 2))
    expected[0, 0, 0] = 0.5
    np.testing.assert_allclose(rates, expected)
    np.testing.assert_allclose(spec["sites_mm"], data["circuit_physical_coords_mm"])


@pytest.mark.parametrize("key", ["detector_coords", "circuit_physical_coords_mm", "round_time_ms"])
def test_changed_geometry_or_schedule_is_rejected(key):
    data, _ = synthetic_data(events=1)
    spec = feature_specification(data)
    changed = copy.deepcopy(data)
    changed[key].flat[0] += 0.1
    with pytest.raises(ValueError, match=key):
        extract_features(changed, spec)


def test_features_ignore_simulator_truth_and_reject_probabilities(tmp_path):
    data, _ = synthetic_data(events=1)
    spec = feature_specification(data)
    expected, _ = extract_features(data, spec)
    data["t1_us"] = np.full((1, 2, 3), np.nan)
    data["event_onset_ms"] = np.asarray([-999.])
    path = tmp_path / "input.npz"
    np.savez(path, **data)
    loaded = load_syndromes(path)
    assert "t1_us" not in loaded and "event_onset_ms" not in loaded
    np.testing.assert_array_equal(extract_features(loaded, spec)[0], expected)
    data["detector_events"] = data["detector_events"].astype(float) + 0.01
    np.savez(path, **data)
    with pytest.raises(ValueError, match="binary"):
        load_syndromes(path)


def test_training_predicts_unseen_events_and_reloads_without_labels(tmp_path):
    pytest.importorskip("sklearn")
    data, truth = synthetic_data()
    data["event_is_control"][-2:] = 1
    truth[-2:] = np.nan  # Controls do not have an epicenter label.
    dataset, output = tmp_path / "dataset", tmp_path / "model"
    save_training_data(dataset, data, truth)
    report = train_localizer(dataset, output, bins=4, trees=32, jobs=1)
    assert report["excluded_controls"] == 2
    test = report["evaluation"]["test"]
    assert test["model"]["mean_error_mm"] < 0.8 * test["training_mean_baseline"]["mean_error_mm"]
    splits = json.loads((output / "splits.json").read_text())
    assert all(94 not in ids and 95 not in ids for ids in splits.values())
    held_out = np.asarray(splits["test"])
    inference = {key: data[key] for key in GEOMETRY_KEYS}
    inference["detector_events"] = data["detector_events"][held_out]
    np.savez(tmp_path / "unlabelled.npz", **inference)
    predictions = predict_epicenters(output / "model.joblib", tmp_path / "unlabelled.npz")
    saved = pd.read_csv(output / "test_predictions.csv")
    np.testing.assert_allclose(predictions[["x_mm", "y_mm"]], saved[["predicted_x_mm", "predicted_y_mm"]])
    assert list(predictions.columns) == ["event", "shot", "x_mm", "y_mm"]
    with pytest.raises(ValueError, match="new or empty"):
        train_localizer(dataset, output, bins=4, trees=1)


def test_label_alignment_and_multiple_sources_are_rejected(tmp_path):
    pytest.importorskip("sklearn")
    data, truth = synthetic_data(events=20)
    dataset = tmp_path / "dataset"
    save_training_data(dataset, data, truth)
    path = dataset / "true_parameters.csv"
    labels = pd.read_csv(path)
    labels.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="label rows"):
        train_localizer(dataset, tmp_path / "model")
    labels.loc[0, "source_count"] = 2
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="single-source"):
        train_localizer(dataset, tmp_path / "model")
