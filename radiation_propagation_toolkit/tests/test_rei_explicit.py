import numpy as np
import pytest

from qp_ode_simulator.temporal_diagnosis import rei_center


SITES = np.array([[0., 0.], [1., 0.], [0., 1.]])


def test_manual_squared_centroid_and_tail():
    bits = np.array([[[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]])
    # Frequencies 1, .5, .25 normalize to 1, 1/3, 0.
    actual = rei_center(bits, SITES, SITES, circuit_repetitions=1, history_length=4)
    np.testing.assert_allclose(actual, [[.1, 0.]])
    padded = np.concatenate([np.zeros_like(bits), bits], axis=2)
    np.testing.assert_allclose(actual, rei_center(padded, SITES, SITES,
                              circuit_repetitions=1, history_length=4))


def test_repetitions_are_independent_of_history():
    bits = np.array([[[1, 1], [1, 0], [0, 1]]])
    assert np.isnan(rei_center(bits, SITES, SITES,
                             circuit_repetitions=1, history_length=1)).all()
    np.testing.assert_allclose(rei_center(bits, SITES, SITES),
                              rei_center(bits, SITES, SITES, circuit_repetitions=3))


@pytest.mark.parametrize('kw', [{'history_length': 0}, {'circuit_repetitions': 0},
                               {'history_length': 1.5}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        rei_center(np.ones((1, 3, 4)), SITES, SITES, **kw)
