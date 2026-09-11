import numpy as np
from qp_ode_simulator.uncertainty_experiments import weighted_median,grid_labels


def test_median_stays_near_majority_not_outlier():
    pts=np.array([[0.,0.],[0.,0.],[100.,0.]])
    p=weighted_median(pts,np.ones((1,3)))
    assert np.linalg.norm(p)<1e-4


def test_grid_edges_and_range():
    labels=grid_labels(np.array([[-1.,-1.],[1.,1.],[2.,2.]]),np.array([[-1.,-1.],[1.,1.]]),5)
    assert labels.tolist()==[0,24,24]


def test_median_batch_independence():
    rng=np.random.default_rng(41)
    pts=rng.normal(size=(30,2));weights=rng.uniform(size=(12,30))
    np.testing.assert_allclose(weighted_median(pts,weights)[:2],
                               weighted_median(pts,weights[:2]),atol=1e-12)
