import numpy as np
from qp_ode_simulator.information_ablation import representations,VARIANTS


def test_representation_dimensions_and_time_average():
    rng=np.random.default_rng(42);raw=(rng.random((3,24,2047))<.04).astype(np.uint8)
    q=np.linspace(.02,.04,24);pre={'cut':339,'quiet':q.tolist()}
    values=representations(raw,pre)
    assert set(values)==set(VARIANTS)
    assert {k:v.shape[1] for k,v in values.items()}==dict(full=169,excess_only=96,relative_only=72,time_average=24,no_spatial=4,global_quiet=169,window512=169,window1024=169)
    np.testing.assert_allclose(values['time_average'],raw.mean(2)-q,atol=1e-8)


def test_short_windows_do_not_use_future_bits():
    raw=np.zeros((1,24,2047),dtype=np.uint8);other=raw.copy();other[...,1023:]=1
    pre={'cut':339,'quiet':[.01]*24}
    a=representations(raw,pre);b=representations(other,pre)
    for name in ['window512','window1024']:np.testing.assert_array_equal(a[name],b[name])
    assert not np.array_equal(a['full'],b['full'])
