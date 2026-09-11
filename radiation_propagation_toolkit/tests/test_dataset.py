import copy
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from qp_ode_simulator import config_path, load_default_simulator_config
from qp_ode_simulator.configuration import load_json
from qp_ode_simulator import dataset as module
from qp_ode_simulator.localization import extract_features, predict_epicenters


pytest.importorskip("stim")


def configuration():
    simulator = load_default_simulator_config()
    simulator["time"] = {"start_ms": 0.0, "end_ms": 0.016, "dt_ms": 0.001}
    simulator["event_timing"] = {"onset_time_ms": 0.003, "control_fraction": 0.0}
    circuit = load_json(config_path("stim_surface_code_d3"))
    circuit.update(rounds=8, shots_per_event=2)
    plan = {
        "geometry": ["circular", "elliptical"],
        "propagation_law": ["ballistic", "diffusive"],
        "epicenter_region": ["inside", "edge", "outside"],
        "generation_bands_per_us": [[1e-10, 1e-9], [1e-9, 1e-8], [1e-8, 3e-7]],
    }
    return simulator, circuit, plan


def collect(root):
    shards = list(module.iter_shards(root))
    return np.concatenate([d["detector_events"] for d, _ in shards]), pd.concat([p for _, p in shards], ignore_index=True)


def test_balanced_sampling_and_controls():
    simulator, _, plan = configuration()
    simulator["event_timing"]["control_fraction"] = 0.25
    cells, controls = [], []
    for event in range(72):
        config, label = module.event_configuration(simulator, plan, event, 42, 123)
        cells.append((config["shape"]["geometry_choices"][0], config["propagation"]["law_choices"][0],
                      next(iter(config["epicenter"]["region_probabilities"])), label["strength_band"]))
        low, high = plan["generation_bands_per_us"][label["strength_band"]]
        assert low <= config["temporal_model"]["qp_ode"]["generation_scale_per_us"] <= high
        controls.append(config["event_timing"]["control_fraction"])
    assert set(Counter(cells).values()) == {2}
    assert 0 < sum(controls) < 72
    assert simulator["event_timing"]["control_fraction"] == 0.25


def test_resume_and_batch_size_preserve_events_and_fixed_hardware(tmp_path, monkeypatch):
    simulator, circuit, plan = configuration()
    first, second = tmp_path / "interrupted", tmp_path / "other_batches"
    original = module.run_stim_pipeline
    calls = 0

    def interrupt(config, stim):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original(config, stim)

    monkeypatch.setattr(module, "run_stim_pipeline", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.generate_dataset(simulator, circuit, plan, first, n_events=4, batch_size=2, progress=None)
    assert module.dataset_manifest(first, require_complete=False)["completed_events"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        list(module.iter_shards(first))
    monkeypatch.setattr(module, "run_stim_pipeline", original)
    module.generate_dataset(simulator, circuit, plan, first, n_events=4, batch_size=3, resume=True, progress=None)
    module.generate_dataset(simulator, circuit, plan, second, n_events=4, batch_size=3, progress=None)
    a, pa = collect(first)
    b, pb = collect(second)
    np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(pa, pb)
    assert pa.event_uid.nunique() == 4
    assert pa.qp_baseline_t1_by_qubit_us.nunique() == 1
    assert pa.qp_baseline_t2_by_qubit_us.nunique() == 1
    assert pa.qp_qubit_frequency_by_qubit_ghz.nunique() == 1
    before = module._hash_file(first / "dataset_manifest.json")
    module.generate_dataset(simulator, circuit, plan, first, n_events=4, resume=True, progress=None)
    assert before == module._hash_file(first / "dataset_manifest.json")
    with pytest.raises(ValueError, match="cannot resume"):
        module.generate_dataset(simulator, circuit, plan, first, n_events=4, device_seed=999, resume=True, progress=None)


def test_corrupt_shard_is_not_silently_regenerated(tmp_path):
    simulator, circuit, plan = configuration()
    manifest = module.generate_dataset(simulator, circuit, plan, tmp_path, n_events=2, progress=None)
    path = tmp_path / manifest["shards"][0]["csv"]
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        module.generate_dataset(simulator, circuit, plan, tmp_path, n_events=2, resume=True, progress=None)


def test_runtime_changes_reject_resume(tmp_path, monkeypatch):
    simulator, circuit, plan = configuration()
    manifest = module.generate_dataset(simulator, circuit, plan, tmp_path, n_events=2, progress=None)
    generation = manifest["generation"]
    assert {"api.py", "layouts.py", "syndrome.py"} <= set(generation["source_hashes"])
    assert {"python", "numpy", "scipy", "stim"} == set(generation["runtime_versions"])
    original = module.importlib.metadata.version
    monkeypatch.setattr(module.importlib.metadata, "version",
                        lambda name: "changed" if name == "stim" else original(name))
    with pytest.raises(ValueError, match="cannot resume"):
        module.generate_dataset(simulator, circuit, plan, tmp_path, n_events=2, resume=True, progress=None)


def test_parallel_generation_matches_serial(tmp_path):
    simulator, circuit, plan = configuration()
    module.generate_dataset(simulator, circuit, plan, tmp_path / "serial", n_events=4, progress=None)
    module.generate_dataset(simulator, circuit, plan, tmp_path / "parallel", n_events=4,
                            batch_size=2, workers=2, progress=None)
    a, pa = collect(tmp_path / "serial")
    b, pb = collect(tmp_path / "parallel")
    np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(pa, pb)


def test_sharded_training_reload_and_prefix_audit(tmp_path):
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    from qp_ode_simulator.streaming_localization import train_sharded_localizer
    simulator, circuit, plan = configuration()
    root = tmp_path / "data"
    manifest = module.generate_dataset(simulator, circuit, plan, root, n_events=24, batch_size=6, progress=None)
    report = train_sharded_localizer(root, tmp_path / "model", bins=2, epochs=2, batch_size=8)
    assert sum(report["independent_events"].values()) == 24
    assert report["evaluation"]["test"]["model"]["samples"] == 8
    import json
    splits = json.loads((tmp_path / "model/splits.json").read_text())
    model = joblib.load(tmp_path / "model/model.joblib")
    training_features = []
    for data, labels in module.iter_shards(root):
        features, _ = extract_features(data, model["features"])
        mask = np.repeat(labels.event.isin(splits["train"]).to_numpy(), 2)
        training_features.append(features[mask])
    np.testing.assert_allclose(model["estimator"].named_steps["scaler"].mean_, np.concatenate(training_features).mean(axis=0), atol=1e-6)
    predictions = predict_epicenters(tmp_path / "model/model.joblib", root / manifest["shards"][0]["npz"])
    assert predictions.shape == (12, 4)
    assert np.isfinite(predictions[["x_mm", "y_mm"]]).all().all()
    data, _ = next(module.iter_shards(root))
    altered = copy.deepcopy(data)
    altered["circuit_id"] = np.asarray("another-code")
    with pytest.raises(ValueError, match="circuit_id"):
        extract_features(altered, model["features"])
    summary = module.audit_dataset(root, tmp_path / "audit", [0.004, 0.008])
    assert summary["events"] == 24
    frame = pd.read_csv(tmp_path / "audit/event_window_diagnostics.csv")
    assert len(frame) == 48
    assert set(frame[frame.window_ms == 0.004].event_uid) == set(frame[frame.window_ms == 0.008].event_uid)
    with pytest.raises(ValueError, match="observation duration"):
        module.audit_dataset(root, tmp_path / "invalid_audit", [1.0])
