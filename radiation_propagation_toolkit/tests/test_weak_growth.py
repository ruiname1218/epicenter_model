import itertools
import numpy as np
import pandas as pd
from qp_ode_simulator.weak_growth import new_split,correlations,fit_svr
from qp_ode_simulator.data_growth import STRATA


def test_new_roles_disjoint_and_balanced():
    rows=[]
    for keys in itertools.product(['circular','elliptical'],['ballistic','diffusive'],['inside','edge','outside'],[0,1,2]):
        for i in range(50):rows.append(dict(zip(STRATA,keys),event_uid=f'{keys}_{i}'))
    split=new_split(pd.DataFrame(rows))
    assert {k:len(v) for k,v in split.items()}=={'train':1080,'validation':360,'test':360}
    assert not set(split['train'])&set(split['validation'])
    assert not set(split['train'])&set(split['test'])
    assert not set(split['validation'])&set(split['test'])


def test_connected_correlations_constant_and_shape():
    x=np.ones((3,8,64),dtype=np.uint8)
    c=correlations(x,16)
    assert c.shape==(3,108)
    np.testing.assert_allclose(c,0,atol=1e-6)


def test_connected_correlations_match_direct_calculation():
    x=np.random.default_rng(1).integers(0,2,(2,8,64),dtype=np.uint8)
    c=correlations(x,16)
    v=x[0,:,16:32].astype(float)
    expected=np.mean(v[0]*v[1])-v[0].mean()*v[1].mean()
    np.testing.assert_allclose(c[0,0],expected,atol=1e-6)
    lag=np.mean(v[0,:-1]*v[0,1:])-v[0,:-1].mean()*v[0,1:].mean()
    np.testing.assert_allclose(c[0,28],lag,atol=1e-6)


def test_weighted_svr_prediction_roundtrip(tmp_path):
    import joblib
    rng=np.random.default_rng(3);x=rng.normal(size=(24,4));y=rng.normal(size=(24,2))
    model=fit_svr(x,y,np.linspace(.5,1.5,len(x)))
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(model.predict(x[:3]),joblib.load(path).predict(x[:3]))
