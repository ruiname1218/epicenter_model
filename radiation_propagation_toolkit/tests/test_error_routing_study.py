import numpy as np
import pandas as pd
import pytest
from qp_ode_simulator.error_routing_study import event_folds,mix,train_gate,gate_predict


def test_folds_keep_shots_together_and_cover_all_events():
    rows=pd.DataFrame({'event_uid':np.repeat(np.arange(36),2),'strength_band':np.repeat(np.arange(36)%3,2)})
    folds=event_folds(rows)
    assert set(folds)=={0,1,2}
    assert (folds[::2]==folds[1::2]).all()
    assert np.array_equal(folds,event_folds(rows))


def test_mix_is_convex_and_validates_weights():
    p=np.array([[[0,0],[2,4]]],dtype=float)
    np.testing.assert_allclose(mix(p,np.array([[.25,.75]])),[[1.5,3]])
    with pytest.raises(ValueError):mix(p,np.array([[.5,.8]]))
    with pytest.raises(ValueError):mix(p,np.array([[-.1,1.1]]))


def test_gate_inference_needs_no_truth_and_stays_in_bank_bounds():
    rng=np.random.default_rng(42);x=rng.normal(size=(30,5));y=rng.normal(size=(30,2))
    p=np.stack([y,y+1,y-1,y+2],axis=1);bands=np.arange(30)%3
    state=train_gate(x,p,y,bands,.01,.2,41)
    answer=gate_predict(state,x,p)
    assert np.isfinite(answer).all()
    assert (answer>=p.min(1)-1e-6).all() and (answer<=p.max(1)+1e-6).all()
