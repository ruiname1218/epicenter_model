import numpy as np
import pytest
from qp_ode_simulator.feature_selection_study import raw_features,fit_reduction,reduce_features,GAMMAS
from qp_ode_simulator.long_observation import features


def test_feature_bank_baseline_equivalence_and_dimensions():
    rng=np.random.default_rng(3);raw=(rng.random((2,24,8191))<.03).astype(np.uint8);q=[.02]*24
    a=raw_features(raw[...,:4095],q,339);b=features(raw,q)
    np.testing.assert_allclose(a['full'],b['4_full'],atol=1e-7)
    np.testing.assert_allclose(a['relative'],b['4_relative'],atol=1e-7)
    assert {k:v.shape[1] for k,v in a.items()}==dict(full=265,relative=120,relative_level=125,early=72,late=48,delta=96,signed=120,sqrt=120,log=120)
    assert all(np.isfinite(v).all() for v in a.values()) and len(GAMMAS)==3
    with pytest.raises(ValueError):raw_features(raw,q,339)


def test_frozen_reduction_inference_needs_no_targets():
    rng=np.random.default_rng(4);x=rng.normal(size=(80,120));y=x[:,:2]
    state=fit_reduction(x,y);held=rng.normal(size=(4,120))
    result=reduce_features({'relative':held},state)
    assert result['top60'].shape==(4,60) and result['pca32'].shape==(4,32)
    assert 0 in state['top'] and 1 in state['top']
