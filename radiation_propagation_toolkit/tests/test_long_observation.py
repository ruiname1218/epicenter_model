import numpy as np
import pytest
from qp_ode_simulator.long_observation import features,edges_for,role


def test_long_windows_preserve_early_bins_and_ignore_future():
    rng=np.random.default_rng(2);raw=(rng.random((2,24,8191))<.03).astype(np.uint8)
    other=raw.copy();other[...,4095:]=1-other[...,4095:]
    a=features(raw,[.02]*24);b=features(other,[.02]*24)
    for h in [2,4]:
        for kind in ['full','relative']:np.testing.assert_array_equal(a[f'{h}_{kind}'],b[f'{h}_{kind}'])
    for h in [4,8]:
        np.testing.assert_array_equal(edges_for(h)[:5],edges_for(2))
        n=5 if h==4 else 7
        np.testing.assert_array_equal(a[f'{h}_relative'].reshape(2,24,n)[:,:,:3],a['2_relative'].reshape(2,24,3))
    assert not np.array_equal(a['8_full'],b['8_full'])


def test_long_observation_roles():
    assert role(0)=='train' and role(1296)=='validation' and role(1512)=='test'
    with pytest.raises(ValueError):role(2052)
