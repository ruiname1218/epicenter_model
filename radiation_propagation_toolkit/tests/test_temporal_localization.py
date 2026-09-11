import copy
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stim")
pytest.importorskip("sklearn")

from qp_ode_simulator import config_path, load_default_simulator_config
from qp_ode_simulator.configuration import load_json
from qp_ode_simulator.dataset import generate_dataset, iter_shards, dataset_manifest
from qp_ode_simulator.learning_plan import create_plan, extend_plan, validate_plan
from qp_ode_simulator.localization import feature_specification
from qp_ode_simulator.temporal_localization import (
    TemporalCNN, detector_series, predict_cnn, run_experiment,
)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("temporal")
    sim = load_default_simulator_config()
    sim["time"] = {"start_ms": 0., "end_ms": .016, "dt_ms": .001}
    sim["event_timing"] = {"onset_time_ms": .003, "control_fraction": 0.}
    stim = load_json(config_path("stim_surface_code_d3"))
    stim.update(rounds=8, shots_per_event=2)
    sampling = {"geometry": ["circular", "elliptical"], "propagation_law": ["ballistic"],
                "epicenter_region": ["inside"],
                "generation_bands_per_us": [[1e-10, 1e-9], [1e-9, 1e-8], [1e-8, 3e-7]]}
    generate_dataset(sim, stim, sampling, root / "data", n_events=36, batch_size=12, progress=None)
    create_plan(root / "data", root / "plan.json", sizes=[5, 10], seed=1)
    return root


def test_frozen_nested_plan_and_ood(tiny_dataset):
    root = tiny_dataset
    plan = json.loads((root / "plan.json").read_text())
    assert plan["training_sets"]["5"] == plan["training_sets"]["10"][:5]
    labels = pd.concat([p for _, p in iter_shards(root / "data")], ignore_index=True)
    manifest = dataset_manifest(root / "data")
    validate_plan(plan, labels, manifest, root / "data")
    other = create_plan(root / "data", root / "same_seed.json", sizes=[5, 10], seed=1)
    assert other == plan
    with pytest.raises(ValueError, match="already exists"):
        create_plan(root / "data", root / "plan.json", sizes=[5, 10])
    bad = copy.deepcopy(plan)
    bad["validation"].append(bad["train_pool"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_plan(bad, labels, manifest, root / "data")
    bad = copy.deepcopy(plan)
    bad["training_sets"]["5"] = list(reversed(bad["training_sets"]["5"]))
    with pytest.raises(ValueError, match="nested"):
        validate_plan(bad, labels, manifest, root / "data")
    bad = copy.deepcopy(plan)
    bad["source_id"] = "changed"
    with pytest.raises(ValueError, match="another dataset"):
        validate_plan(bad, labels, manifest, root / "data")
    bad = copy.deepcopy(plan)
    bad["train_pool"][0], bad["test_ood"][0] = bad["test_ood"][0], bad["train_pool"][0]
    with pytest.raises(ValueError, match="leaked"):
        validate_plan(bad, labels, manifest, root / "data")
    families = labels.copy()
    families["family_id"] = np.arange(len(labels)) // 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_plan(plan, families, manifest, root / "data")


def test_raw_detector_order_no_boundary_or_truth_leak(tiny_dataset):
    data, _ = next(iter_shards(tiny_dataset / "data"))
    spec = feature_specification(data, bins=2)
    series = detector_series(data, spec)
    bits = data["detector_events"].reshape(-1, len(data["detector_coords"]))
    for site_index, site in enumerate(spec["sites_grid"]):
        for tick in range(1, len(data["round_time_ms"])):
            index = np.flatnonzero(np.all(data["detector_coords"] == [*site, tick], axis=1))
            np.testing.assert_array_equal(series[:, site_index, tick-1], bits[:, index[0]])
    changed = copy.deepcopy(data)
    boundary = (data["detector_coords"][:, 2] == 0) | (data["detector_coords"][:, 2] == len(data["round_time_ms"]))
    changed["detector_events"][:, :, boundary] ^= 1
    changed["true_epicenter"] = np.full((12, 2), 99999.)
    np.testing.assert_array_equal(series, detector_series(changed, spec))
    changed["round_time_ms"] += .1
    with pytest.raises(ValueError, match="differs"):
        detector_series(changed, spec)


def test_extension_preserves_evaluation_and_shots(tiny_dataset):
    root = tiny_dataset
    manifest = dataset_manifest(root / "data")
    expanded = root / "expanded"
    generate_dataset(manifest["simulator_config"], manifest["stim_config"], manifest["sampling"],
                     expanded, n_events=72, batch_size=12, progress=None)
    before = json.loads((root / "plan.json").read_text())
    after = extend_plan(root / "plan.json", expanded, root / "extended_plan.json", sizes=[5, 10, 30])
    for key in ("validation", "test_id", "test_ood"):
        assert before[key] == after[key]
    assert before["training_sets"]["10"] == after["training_sets"]["30"][:10]
    original_bits = np.concatenate([data["detector_events"] for data, _ in iter_shards(root / "data")])
    new_bits = np.concatenate([data["detector_events"] for data, _ in iter_shards(expanded)])
    np.testing.assert_array_equal(original_bits, new_bits[:36])
    with pytest.raises(ValueError, match="retain every previous"):
        extend_plan(root / "plan.json", expanded, root / "invalid_plan.json", sizes=[30])


def test_cnn_shapes_and_gradients():
    torch.set_num_threads(1)
    model = TemporalCNN(8)
    prediction = model(torch.zeros(3, 8, 1023))
    assert prediction.shape == (3, 2)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_cnn_learns_controlled_spatial_signal():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = TemporalCNN(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    raw = torch.zeros(32, 4, 128)
    target = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])[torch.arange(32) % 4]
    raw[torch.arange(32), torch.arange(32) % 4, 32:96] = 1.
    model.eval()
    initial = torch.nn.functional.mse_loss(model(raw), target).item()
    for _ in range(60):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(raw), target)
        loss.backward()
        optimizer.step()
    model.eval()
    assert torch.nn.functional.mse_loss(model(raw), target).item() < initial * .1


def test_training_save_reload_inference_without_labels(tiny_dataset):
    root = tiny_dataset
    report = run_experiment(root / "data", root / "plan.json", root / "experiment",
                            seeds=[7], epochs=2, patience=1, bins=2, batch_size=8, threads=1)
    assert len(report) == 8  # two models, two sizes, ID and OOD
    predictions = pd.read_csv(root / "experiment/test_predictions.csv")
    run = root / "experiment/cnn_n10_seed7"
    meta = json.loads((run / "model.json").read_text())
    assert 1 <= meta["selected_epoch"] <= 2
    from qp_ode_simulator.localization import GEOMETRY_KEYS
    for shard, labels in iter_shards(root / "data"):
        path = root / "unlabelled.npz"
        np.savez_compressed(path, detector_events=shard["detector_events"],
                            circuit_id=shard["circuit_id"], **{k: shard[k] for k in GEOMETRY_KEYS})
        pred = predict_cnn(run, path, batch_size=8)
        assert pred.shape == (len(labels) * 2, 2)
        for i, event in enumerate(labels.event):
            expected = predictions[(predictions.model == "cnn") & (predictions.train_events == 10)
                                   & (predictions.event == event)].sort_values("shot")
            if len(expected):
                np.testing.assert_allclose(pred[2*i:2*i+2], expected[["predicted_x_mm", "predicted_y_mm"]], atol=1e-6)
    with pytest.raises(ValueError, match="new or empty"):
        run_experiment(root / "data", root / "plan.json", root / "experiment")
