import numpy as np
import pytest

from qp_ode_simulator.weak_information_study import role,bin_means,observed_features,template_scores


def test_split_complete_event_boundaries():
    assert role(0)==role(1727)=='train'
    assert role(1728)==role(2015)=='validation'
    assert role(2016)==role(2591)=='test'
    with pytest.raises(ValueError):role(2592)


def test_binning_and_observable_batch_independence():
    raw=np.random.default_rng(3).integers(0,2,(3,2,2047),dtype=np.uint8)
    q=np.full(2,.03)
    for kind in ['fixed','estimated','multi']:
        a=observed_features(raw,q,345,kind)
        b=observed_features(raw[1:2],q,345,kind)
        np.testing.assert_allclose(a[1:2],b,atol=1e-6)
        assert np.isfinite(a).all()
    np.testing.assert_allclose(bin_means(np.ones((2,31))),1)


def test_composite_template_scores_match_explicit_binomial_terms():
    rng=np.random.default_rng(2)
    bank=rng.uniform(.01,.2,(3,2,16)); observed=rng.uniform(0,.2,(4,2,16))
    lengths=np.diff(np.linspace(0,2047,17,dtype=int)); counts=observed*lengths
    expected=np.array([[np.sum(counts[i]*np.log(p)+(lengths-counts[i])*np.log1p(-p)) for p in bank] for i in range(4)])
    expected-=expected.max(1,keepdims=True)
    np.testing.assert_allclose(template_scores(observed,bank,{},2047,'composite'),expected,atol=1e-10)


def test_gls_template_prefers_matching_mean():
    bank=np.stack([np.zeros((2,16)),np.ones((2,16))])
    cal={'white':np.eye(2),'mean':np.zeros((2,16))}
    scores=template_scores(bank,bank,cal,2047,'gls')
    np.testing.assert_array_equal(scores.argmax(1),[0,1])
