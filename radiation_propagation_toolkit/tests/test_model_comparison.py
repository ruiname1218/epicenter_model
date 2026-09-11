import joblib
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
from qp_ode_simulator.model_comparison import configurations, fit_mlp


def test_grid_and_families():
    grid = configurations()
    assert len(grid) == 20
    assert len({item["family"] for item in grid}) == 7


def test_mlp_external_validation_and_serialization(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    y = x[:, :2]*2
    train, val = np.arange(40), np.arange(40, 50)
    model, history = fit_mlp(x, y, train, val, seed=41, hidden=(16,), alpha=1., epochs=20, patience=5)
    assert 1 <= history["selected_epoch"] <= 20
    np.testing.assert_allclose(model.scaler.mean_, x[train].mean(0))
    np.testing.assert_allclose(model.center, y[train].mean(0))
    assert model.estimator.early_stopping is False
    path = tmp_path / "mlp.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict(x[50:]), joblib.load(path).predict(x[50:]))
    # Changing unused test features/labels cannot affect fitting or epoch selection.
    x[50:], y[50:] = 999, -999
    other, other_history = fit_mlp(x, y, train, val, seed=41, hidden=(16,), alpha=1., epochs=20, patience=5)
    assert other_history == history
    np.testing.assert_array_equal(model.predict(x[:40]), other.predict(x[:40]))
