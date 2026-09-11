import numpy as np
import pandas as pd
import pytest

from qp_ode_simulator.distance_growth import role,training_indices,features
from qp_ode_simulator.adaptive_localization import segmented_features


def test_roles_and_nested_event_selection():
    rows=[]
    cells=[(geometry,strength,other) for geometry in ['circular','elliptical'] for strength in range(3) for other in range(6)]
    for origin,n in [('old',864),('new',3456)]:
        for i in range(n):
            g,s,_=cells[i%36]
            rows.append(dict(origin=origin,event=i,geometry=g,strength_band=s,role='train' if origin=='old' else role(i)))
    frame=pd.DataFrame(rows);subsets=[]
    for n in [720,1440,2880]:
        ids=training_indices(frame,n);assert len(ids)==n
        assert (frame.iloc[ids].role=='train').all()
        assert not ((frame.iloc[ids].geometry=='elliptical')&(frame.iloc[ids].strength_band==2)).any()
        assert (frame.iloc[ids].strength_band==0).sum()==n*2//5
        subsets.append(set(ids))
    assert subsets[0]<subsets[1]<subsets[2]
    with pytest.raises(ValueError):role(3456)


def test_batched_features_equal_full_and_do_not_use_future_labels():
    raw=np.random.default_rng(4).integers(0,2,(135,24,32),dtype=np.uint8)
    prep=dict(cut=8,quiet=np.full(24,.03))
    expected=segmented_features(raw,np.full(len(raw),8),prep['quiet'])
    np.testing.assert_array_equal(features(raw,prep),expected)
    assert expected.shape==(135,169)
