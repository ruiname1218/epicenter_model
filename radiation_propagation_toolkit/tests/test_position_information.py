import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
from qp_ode_simulator.position_information import timing_features


def test_known_onset_features_use_interior_detector_times():
    raw = np.array([[[1, 0, 1], [0, 1, 1]]], dtype=np.uint8)
    spec = {"geometry": {"round_time_ms": [.001, .002, .003, .004]}}
    result = timing_features(raw, [.003], spec)
    np.testing.assert_allclose(result, [[1, 0, .5, 1, -.5, 1]])


@pytest.mark.parametrize("onset", [.001, .005])
def test_known_onset_rejects_missing_pre_or_post(onset):
    raw = np.zeros((1, 2, 3), dtype=np.uint8)
    spec = {"geometry": {"round_time_ms": [.001, .002, .003, .004]}}
    with pytest.raises(ValueError, match="pre- and post"):
        timing_features(raw, [onset], spec)
