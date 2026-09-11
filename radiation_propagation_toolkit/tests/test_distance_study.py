import copy
import json

import numpy as np
import pytest

from qp_ode_simulator.simulator import sample_epicenter
from qp_ode_simulator.distance_study import union_geometry, geometry, role, ordinary
from qp_ode_simulator.adaptive_localization import segmented_features
from qp_ode_simulator.dataset import event_configuration
from qp_ode_simulator import load_default_simulator_config


def test_fixed_reference_region_independent_of_layout():
    for region in ['inside','edge','outside']:
        c=dict(reference_bounds_mm=[[-3,3],[-3,3]],region_probabilities={region:1.},outside_margin_mm=[.25,1.])
        a=sample_epicenter(np.random.default_rng(4),c,np.array([[-3,-3],[3,3]]))
        b=sample_epicenter(np.random.default_rng(4),c,np.array([[-7,-7],[7,7]]))
        assert a==b
    for bounds in [[[3,-3],[-3,3]], [[0,1]], [[0,np.nan],[0,1]]]:
        with pytest.raises(ValueError):sample_epicenter(np.random.default_rng(1),dict(c,reference_bounds_mm=bounds))


def test_union_geometry_and_variable_features():
    circuits={str(d):dict(task='surface_code:rotated_memory_z',distance=d,rounds=8,
                         round_duration_ms=.001,start_time_ms=0,layout={'qubit_pitch_mm':1.}) for d in [3,5,7]}
    union,indices=union_geometry(circuits)
    for d in [3,5,7]:
        layout,sites,_=geometry(circuits[str(d)])
        np.testing.assert_allclose(union[indices[d]],layout.physical_coords_mm,atol=1e-9)
        assert len(sites)==d*d-1 and len(layout.qubit_ids)==2*d*d-1
        raw=np.zeros((4,d*d-1,20),dtype=np.uint8)
        x=segmented_features(raw,np.full(4,5),np.full(d*d-1,.01))
        assert x.shape==(4,7*(d*d-1)+1)


def test_balanced_event_splits():
    base=load_default_simulator_config()
    plan=dict(geometry=['circular','elliptical'],propagation_law=['ballistic','diffusive'],
              epicenter_region=['inside','edge','outside'],generation_bands_per_us=[[1e-10,1e-9],[1e-9,1e-8],[1e-8,3e-7]])
    from collections import Counter
    groups={r:Counter() for r in ['train','validation','test']}
    for i in range(1440):
        c,l=event_configuration(base,plan,i,2026090901,123)
        key=(c['shape']['geometry_choices'][0],c['propagation']['law_choices'][0],next(iter(c['epicenter']['region_probabilities'])),l['strength_band'])
        groups[role(i)][key]+=1
    for r,n in [('train',24),('validation',8),('test',8)]:
        assert len(groups[r])==36 and set(groups[r].values())=={n}
