import numpy as np

from qp_ode_simulator.syndrome_features_study import counts, coincidences, fit_calibration, feature_sets, pairs_for
from qp_ode_simulator.distance_growth import features
from qp_ode_simulator.feature_refinements import WeakSignalGate


def test_counts_and_coincidences_are_direct_binary_products():
    raw = np.zeros((1, 2, 32), dtype=np.uint8)
    raw[0, 0, ::2] = 1
    raw[0, 1, 1::2] = 1
    np.testing.assert_array_equal(counts(raw, 4), .5*np.ones((1, 2, 4)))
    joint = coincidences(raw, np.array([[0, 1]]))
    np.testing.assert_array_equal(joint[:, :4], 0)
    # lag-one cross products exist, self products do not.
    assert joint[:, 4:].max() > .4
    np.testing.assert_array_equal(joint[:, 4:].reshape(1, 4, 4)[:, :, 2:], 0)


def test_features_batch_independent_and_baseline_exact():
    rng = np.random.default_rng(4)
    sites = np.array([[0, 0], [0, 2], [2, 0]])
    quiet = (rng.random((128, 3, 64)) < .05).astype(np.uint8)
    calibration = fit_calibration(quiet, sites)
    raw = (rng.random((67, 3, 64)) < .1).astype(np.uint8)
    prep = {'cut': 16, 'quiet': [.05]*3}
    all_features = feature_sets(raw, prep, calibration)
    single = feature_sets(raw[3:4], prep, calibration)
    np.testing.assert_array_equal(all_features['baseline'], features(raw, prep))
    for key, value in all_features.items():
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value[3:4], single[key], atol=1e-6)
    assert all_features['combined'].shape[1] > all_features['multiscale'].shape[1]
    assert all_features['multiscale'].shape == all_features['whitened'].shape
    assert np.linalg.eigvalsh(calibration['white'][4]).min() > 0


def test_pair_geometry_excludes_distant_and_duplicate_links():
    pairs = pairs_for([[0, 0], [2, 0], [0, 2], [20, 20]])
    assert set(map(tuple, pairs)) == {(0, 1), (0, 2), (1, 2)}


def test_learned_gate_uses_observed_probability_and_keeps_strong_endpoint():
    class Classifier:
        classes_ = np.array([0, 1])
        def predict_proba(self, x): return np.c_[1-x[:, 0], x[:, 0]]
    class Regressor:
        def predict(self, x): return np.full((len(x), 2), 2.)
    model = WeakSignalGate(Classifier(), Regressor(), Regressor(), [0, 0], fraction=.5, power=2)
    actual = model.predict(np.array([[0.], [.5], [1.]]))
    np.testing.assert_allclose(actual, [[2.,2.], [1.75,1.75], [1.,1.]])
