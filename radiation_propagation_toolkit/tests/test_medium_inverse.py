import numpy as np
from qp_ode_simulator.medium_inverse import edges,rates,relative,score_matrix,template_predict,fit_template
from qp_ode_simulator.radical_data import relative_features


def test_paired_time_boundaries():
    for h,n in [(4,16),(8,32),(16,64)]:
        e=edges(h);assert len(e)==n+1 and e[0]==0 and e[-1]==h*1024-1
        assert np.all(np.diff(e)>0)
    np.testing.assert_array_equal(edges(16)[:33],edges(8))
    np.testing.assert_array_equal(edges(8)[:17],edges(4))


def test_features_no_future_leak_and_old_baseline_equivalence():
    raw=np.random.default_rng(1).binomial(1,.03,(2,24,16383));quiet=np.full(24,.03)
    a=relative(raw,4,quiet);counts=rates(raw,4)
    np.testing.assert_allclose(a,relative_features(raw[...,:4095],quiet))
    raw[...,4095:]=1-raw[...,4095:]
    np.testing.assert_array_equal(a,relative(raw,4,quiet));np.testing.assert_array_equal(counts,rates(raw,4))
    assert relative(raw,16,quiet).shape==(2,216)


def test_binomial_template_bank_one_per_physical_event():
    profiles=np.stack([np.full((24,16),.02),np.full((24,16),.1)])
    latent=np.repeat(profiles,2,axis=0);centers=np.repeat([[-1.,2.],[3.,-2.]],2,axis=0)
    b=fit_template(latent,latent,centers,4,'binomial')
    assert len(b['centers'])==2
    np.testing.assert_allclose(template_predict(profiles,b),centers[::2],atol=1e-5)


def test_gls_score_agrees_with_mahalanobis_distance():
    cov=np.array([[2.,.7],[.7,1.]])
    chol=np.linalg.cholesky(cov);profiles=np.array([[.1,.2],[.6,.7]])
    b=dict(kind='GLS',chol=chol,noise_mean=np.array([.01,-.02]),whitened=np.linalg.solve(chol,profiles.T).T,
        centers=np.array([[0.,0.],[1.,1.]]),temperature=1.)
    x=np.array([[[.13,.24]],[[.65,.75]]]);s=score_matrix(x,b)
    residual=x.reshape(2,-1)[:,None]-b['noise_mean']-profiles[None]
    direct=-.5*np.einsum('nki,ij,nkj->nk',residual,np.linalg.inv(cov),residual)
    np.testing.assert_allclose(s[:,1]-s[:,0],direct[:,1]-direct[:,0])


def test_no_information_returns_template_center_average():
    profiles=np.full((4,24,16),.03);xy=np.array([[-1.,0.],[-1.,0.],[3.,2.],[3.,2.]])
    b=fit_template(profiles,profiles,xy,4,'binomial')
    np.testing.assert_allclose(template_predict(profiles[:1],b),[[1.,1.]])
