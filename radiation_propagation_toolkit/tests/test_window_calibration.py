import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("stim")
from test_temporal_localization import tiny_dataset
from qp_ode_simulator.dataset import dataset_manifest, iter_shards
from qp_ode_simulator.localization import feature_specification, load_syndromes
from qp_ode_simulator.window_calibration import (
    WindowView, generate_calibration, validate_calibration, observation_prefix,
    predict_window_model, run_ablation,
)


def test_window_normalization_and_future_isolation():
    raw = np.zeros((2, 3, 15), dtype=np.uint8)
    raw[:, :, 3:6] = 1
    probabilities = np.array([.01, .03, .1])
    view = WindowView(raw, 8, probabilities)
    expected = (raw[:, :, :7]-probabilities[:, None]) / (10*np.sqrt(probabilities*(1-probabilities)))[:, None]
    np.testing.assert_allclose(view[:], expected, rtol=1e-6)
    before = view[:]
    raw[:, :, 7:] = 1
    np.testing.assert_array_equal(before, view[:])
    np.testing.assert_array_equal(WindowView(raw, 8)[:], raw[:, :, :7])
    with pytest.raises(ValueError, match="quiet probabilities"):
        WindowView(raw, 8, [0., .03, .1])
    with pytest.raises(ValueError, match="window"):
        WindowView(raw, 17)


def test_calibration_and_full_ablation(tiny_dataset):
    root = tiny_dataset
    calibration = root / "quiet"
    record = generate_calibration(root / "data", calibration, shots=128)
    data, _ = next(iter_shards(root / "data"))
    spec = feature_specification(data, 2)
    manifest = dataset_manifest(root / "data")
    assert validate_calibration(calibration, manifest, spec) == record
    assert record["observations_per_site"] == 128 * 7
    bad = json.loads(json.dumps(manifest))
    bad["generation"]["device_seed"] += 1
    with pytest.raises(ValueError, match="device"):
        validate_calibration(calibration, bad, spec)
    result = run_ablation(root / "data", root / "plan.json", calibration, root / "ablation",
                          seeds=[7], windows=(4, 8), epochs=2, patience=1, batch_size=8)
    assert len(result["groups"]) == 8
    assert len(result["paired_contrasts"]) == 8
    predictions = pd.read_csv(root / "ablation/test_predictions.csv")
    for calibrated in (False, True):
        model = root / f"ablation/cnn_r4_{'calibrated' if calibrated else 'raw'}_seed7"
        npz = root / f"observable_{calibrated}.npz"
        np.savez_compressed(npz, **data)
        full = predict_window_model(model, npz)
        prefix = observation_prefix(data, 4)
        prefix["unknown_truth"] = np.array([999.])
        short = root / f"prefix_{calibrated}.npz"
        np.savez_compressed(short, **prefix)
        np.testing.assert_allclose(full, predict_window_model(model, short), atol=1e-6)
        for i, event in enumerate(data["event_ids"]):
            expected = predictions[(predictions["rounds"] == 4) & (predictions.calibrated == calibrated)
                                   & (predictions.event == event)].sort_values("shot")
            if len(expected):
                np.testing.assert_allclose(full[2*i:2*i+2], expected[["predicted_x_mm", "predicted_y_mm"]], atol=1e-6)
    with pytest.raises(ValueError, match="insufficient"):
        observation_prefix(prefix, 8)
