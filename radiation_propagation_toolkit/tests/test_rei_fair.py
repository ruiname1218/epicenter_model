import numpy as np
import torch
from qp_ode_simulator.rei_fair_data import features,feature_edges,configuration
from qp_ode_simulator.rei_fair_models import rei_rates,CNN,cnn_predict
from qp_ode_simulator.temporal_diagnosis import rei_center


def test_rei_sufficient_statistics_exact_equivalence():
    rng=np.random.default_rng(9);sites=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    raw=rng.binomial(1,np.array([.02,.04,.1,.07])[None,:,None],(8,4,2047))
    raw[0]=0;raw[1]=1
    for k in [256,512,1024,2047]:
        expected=rei_center(raw,sites,sites,history_length=k,circuit_repetitions=1)
        actual=rei_rates(raw[...,-k:].mean(-1),sites,sites,k)
        np.testing.assert_allclose(actual,expected,equal_nan=True,atol=1e-12)


def test_equal_windows_and_no_future_features():
    rng=np.random.default_rng(2);raw=rng.binomial(1,.03,(2,24,8191));quiet=np.full(24,.03)
    for h,d in [(2,72),(4,120),(8,168)]:
        a=features(raw,h,quiet);assert a.shape==(2,d)
        changed=raw.copy();changed[...,h*1024-1:]=1-changed[...,h*1024-1:]
        np.testing.assert_array_equal(a,features(changed,h,quiet))
        assert feature_edges(h)[-1]==h*1024-1


def test_noquiet_model_does_not_require_calibration():
    raw=np.ones((2,24,8191),dtype=np.uint8)
    assert np.isfinite(features(raw,4)).all()
    np.testing.assert_allclose(features(raw,4),1/1.003)


def test_cnn_uses_observed_input_only():
    torch.set_num_threads(1);torch.manual_seed(1)
    x=np.zeros((2,24,128),dtype=np.float32);m=CNN()
    b=dict(state=m.state_dict(),mean=np.zeros((1,24,1)),std=np.ones((1,24,1)),center=np.array([0.,0.]))
    a=cnn_predict(b,x);assert a.shape==(2,2) and np.isfinite(a).all()
    b['unused_truth']=np.array([999.,-999.]);np.testing.assert_array_equal(a,cnn_predict(b,x))
