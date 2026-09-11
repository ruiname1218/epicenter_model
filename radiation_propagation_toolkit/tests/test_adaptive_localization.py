import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
from qp_ode_simulator.adaptive_localization import (
    candidate_prediction, estimated_cuts, export_selected, load_model, representations, segmented_features, shrink,
)


def test_segmentation_same_cuts_same_features():
    raw = np.zeros((2, 3, 120), dtype=np.uint8)
    raw[0, 0, 30:] = 1
    info = {"fixed_cut": 30, "quiet": [.03]*3,
            "specification": {"geometry": {"round_time_ms": np.arange(121)*.001}}}
    result = representations(raw, info, [.031, .031])
    np.testing.assert_array_equal(result["fixed"], result["oracle"])
    assert result["fixed"].shape == (2, 3*7+1)
    assert np.isfinite(result["fixed"]).all()
    assert "oracle" not in representations(raw, info)


def test_change_point_uses_observed_upward_step():
    raw = np.zeros((2, 4, 128), dtype=np.uint8)
    raw[0, :, 32:] = 1
    raw[1, :, 48:] = 1
    np.testing.assert_array_equal(estimated_cuts(raw), [32, 48])


def test_shrink_identity_prior_and_negative_signal():
    predictions = np.array([[2., 3.], [4., 5.], [6., 7.]])
    np.testing.assert_array_equal(shrink(predictions, [0., 0.], [0., 0., 0.], 0), predictions)
    got = shrink(predictions, [0., 0.], [-1., 0., .01], .01)
    np.testing.assert_allclose(got, [[0, 0], [0, 0], [3, 3.5]])


def test_invalid_segmentation_and_oracle_rejected():
    with pytest.raises(ValueError, match="cuts"):
        segmented_features(np.zeros((1, 2, 20)), [0], [.03, .03])
    with pytest.raises(ValueError, match="quiet"):
        segmented_features(np.zeros((1, 2, 20)), [4], [0., .03])
    with pytest.raises(ValueError, match="oracle"):
        candidate_prediction({"kind": "regressor", "representation": "oracle"}, {}, None, None, None, None)


def test_compact_export_preserves_selected_regressor(tmp_path):
    import joblib
    from sklearn.linear_model import Ridge
    from qp_ode_simulator.dataset import _write_json
    from qp_ode_simulator.localization import _hash_file
    original = tmp_path / "full"
    original.mkdir()
    x = np.arange(12).reshape(6, 2)
    model = Ridge().fit(x, x)
    candidate = {"name": "fixed_ridge", "kind": "regressor", "representation": "fixed", "models": [model]}
    joblib.dump({"candidates": [candidate], "states": ["unused CNN"]}, original / "models.joblib")
    _write_json(original / "model.json", {"format": "adaptive_localization_v1", "selected": "fixed_ridge",
                                         "artifact_sha256": _hash_file(original / "models.joblib")})
    export_selected(original, tmp_path / "compact")
    info, artifact = load_model(tmp_path / "compact")
    assert info["inference_only"] and not artifact["states"]
    np.testing.assert_array_equal(artifact["candidates"][0]["models"][0].predict(x), model.predict(x))
