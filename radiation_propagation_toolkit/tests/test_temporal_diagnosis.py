import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from qp_ode_simulator.temporal_diagnosis import (
    ARMS, DiagnosisCNN, evaluate_frozen, oracle_values, predict_model, rei_center, run,
)
from qp_ode_simulator.temporal_localization import TemporalCNN
from test_temporal_localization import tiny_dataset


@pytest.mark.parametrize("arm", ARMS)
def test_shapes_gradients_and_privilege(arm):
    torch.set_num_threads(1)
    model = DiagnosisCNN(8, arm)
    x = torch.zeros(3, 8, 127)
    extra = torch.ones(3, model.oracle_size) if model.oracle_size else None
    pred = model(x, extra)
    assert pred.shape == (3, 3 if model.joint else 2)
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if model.oracle_size:
        with pytest.raises(ValueError, match="privileged"):
            model(x)
    else:
        with pytest.raises(ValueError, match="privileged"):
            model(x, torch.ones(3, 1))


def test_original_exact_equivalence():
    torch.manual_seed(41)
    old = TemporalCNN(8).eval()
    torch.manual_seed(41)
    new = DiagnosisCNN(8, "original").eval()
    x = torch.rand(2, 8, 127)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)


def test_rei_minmax_squared_and_abstention():
    sites = np.array([[0., 0.], [1., 0.], [2., 0.]])
    bits = np.zeros((3, 3, 10))
    bits[0, 0, :2], bits[0, 1, :4], bits[0, 2, :6] = 1, 1, 1
    bits[2, :2, :] = 1
    result = rei_center(bits, sites, sites)
    np.testing.assert_allclose(result[0], [1.8, 0.])
    assert np.isnan(result[1:]).all()


def test_train_reload_and_oracle_rejection(tiny_dataset):
    import pandas as pd
    from qp_ode_simulator.dataset import dataset_manifest, generate_dataset
    root = tiny_dataset
    destination = root / "diagnosis"
    report = run(root / "data", root / "plan.json", destination, seeds=(7,), epochs=1, patience=1)
    assert len(report) > 0
    predictions = pd.read_csv(destination / "test_predictions.csv")
    plan = json.loads((root / "plan.json").read_text())
    assert not set(predictions.event) & set(plan["training_sets"]["10"])
    shard = next((root / "data").glob("*.npz"))
    pred = predict_model(destination / "multiscale_joint_seed7", shard)
    assert pred.shape[1] == 3 and np.isfinite(pred).all()
    labels = pd.read_csv(shard.with_suffix(".csv"))
    expected = predictions[(predictions.arm == "multiscale_joint") & predictions.event.isin(labels.event)]
    for row in expected.itertuples():
        index = int(np.flatnonzero(labels.event.to_numpy() == row.event)[0])*2 + row.shot
        np.testing.assert_allclose(pred[index], [row.predicted_x_mm, row.predicted_y_mm, row.predicted_onset_ms], atol=1e-5)
    with pytest.raises(ValueError, match="privileged"):
        predict_model(destination / "oracle_onset_seed7", shard)
    # Ordinary inputs do not even inspect onset or any source parameter.
    assert oracle_values(pd.DataFrame(index=range(3)), "original").shape == (3, 0)
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_frozen(destination, root / "data", root / "data", root / "invalid_fresh")
    manifest = dataset_manifest(root / "data")
    fresh = root / "fresh"
    generate_dataset(manifest["simulator_config"], manifest["stim_config"], manifest["sampling"],
                     fresh, n_events=36, batch_size=12, seed=789, syndrome_seed=790, progress=None)
    result = evaluate_frozen(destination, fresh, root / "data", root / "confirmation")
    assert any(r["arm"] == "multiscale_joint" and "onset_mae_rounds" in r for r in result)
