import copy

import numpy as np
import pytest

from qp_ode_simulator.fault_response import build_response,fault_bits,sample,interior
from qp_ode_simulator.stim_qec import build_stim_layout,round_tick_counts,inject_scheduled_gate_slice_pauli_channels


@pytest.fixture(scope='module', params=[3,5,7])
def config(request):
    return dict(task='surface_code:rotated_memory_z',distance=request.param,rounds=8,round_duration_ms=.001,start_time_ms=0,
        circuit_noise={'after_clifford_depolarization':.0005,'before_measure_flip_probability':.001},
        noise_accounting={'depolarizing_channels_exclude_t1_t2':True})


@pytest.fixture(scope='module')
def response(config):return build_response(config)


def test_random_forced_multifault_patterns_match_stim_exactly(config,response):
    c=copy.deepcopy(config);c['circuit_noise']={k:0. for k in c['circuit_noise']}
    layout=build_stim_layout(c);counts=round_tick_counts(layout.circuit,c['rounds']);_,order=interior(layout,c['rounds'])
    rng=np.random.default_rng(4);shape=(c['rounds'],counts.max(),len(layout.qubit_ids))
    for _ in range(64):
        labels=rng.integers(1,4,shape)*(rng.random(shape)<.05)
        xyz=[(labels==k).astype(float) for k in [1,2,3]]
        circuit=inject_scheduled_gate_slice_pauli_channels(layout.circuit,layout.qubit_ids,*xyz,counts)
        expected=circuit.compile_detector_sampler(seed=123).sample(shots=1)[:,order].reshape(1,c['distance']**2-1,c['rounds']-1)
        actual=fault_bits(response,((labels==1)|(labels==2))[None],((labels==2)|(labels==3))[None])
        np.testing.assert_array_equal(actual,expected)


def test_categorical_sampling_matches_stim_marginals_and_parities(config,response):
    layout=build_stim_layout(config);counts=round_tick_counts(layout.circuit,config['rounds']);_,order=interior(layout,config['rounds'])
    rng=np.random.default_rng(8);shape=(config['rounds'],counts.max(),len(layout.qubit_ids))
    xyz=[rng.uniform(.0001,.002,shape).astype(np.float32) for _ in range(3)]
    circuit=inject_scheduled_gate_slice_pauli_channels(layout.circuit,layout.qubit_ids,*xyz,counts)
    shots=20000
    reference=circuit.compile_detector_sampler(seed=124).sample(shots=shots)[:,order].reshape(shots,config['distance']**2-1,config['rounds']-1)
    actual,expected=sample(response,config,*xyz,seed=125,shots=shots)
    for observed in [reference,actual]:
        tolerance=6*np.sqrt(expected*(1-expected)/shots)+2/shots
        assert np.all(np.abs(observed.mean(0)-expected)<tolerance)
    a=actual.reshape(shots,-1);b=reference.reshape(shots,-1)
    # Joint checks, not only single-detector means: random parity observables.
    for _ in range(30):
        indices=rng.choice(a.shape[1],5,replace=False)
        pa=np.bitwise_xor.reduce(a[:,indices],axis=1).mean()
        pb=np.bitwise_xor.reduce(b[:,indices],axis=1).mean()
        assert abs(pa-pb)<6*np.sqrt((pa*(1-pa)+pb*(1-pb))/shots)+2/shots
    x,_=sample(response,config,*xyz,seed=17,shots=4)
    y,_=sample(response,config,*xyz,seed=17,shots=4)
    np.testing.assert_array_equal(x,y)
