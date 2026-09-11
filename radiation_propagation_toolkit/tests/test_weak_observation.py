import numpy as np
import pytest
import stim

from qp_ode_simulator.weak_observation import detector_marginals,reduce_background
from qp_ode_simulator.stim_qec import scheduled_gate_slice_pauli_probabilities,_compose_independent_z_channel
from qp_ode_simulator.syndrome import t1_to_pauli_probabilities


@pytest.mark.parametrize('phase',[False,True])
def test_batched_gate_channels_equal_scalar_loop(phase):
    rng=np.random.default_rng(1);t1=rng.uniform(10,300,(2,7,5)).astype(np.float32);t2=t1*1.2
    counts=np.array([2,3,2,3,1,3,2]);durations=np.tile([3e-5,4e-5,7.8e-4],(7,1))
    shift=rng.normal(0,1e-4,t1.shape)
    actual=scheduled_gate_slice_pauli_probabilities(t1,durations,counts,round_t2_us=t2,
        round_frequency_shift_ghz=shift,include_coherent_phase_approximation=phase)
    for r,count in enumerate(counts):
        for s in range(count):
            expected=t1_to_pauli_probabilities(t1[:,r:r+1],float(durations[r,s]),t2_us=t2[:,r:r+1])
            if phase:
                p=np.sin(np.pi*shift[:,r:r+1]*(float(durations[r,s])*1e6))**2
                expected=_compose_independent_z_channel(expected,p)
            for key in ['x','y','z','i','t2_us']:
                np.testing.assert_array_equal(actual[key][:,r,s],expected[key][:,0])
        np.testing.assert_array_equal(actual['x'][:,r,count:],0)


def test_background_intervention_preserves_excess():
    t=np.array([[[100.,200.],[50.,100.]]])
    changed=reduce_background(t,.1)
    np.testing.assert_allclose(1/changed-1/changed[:,:1],1/t-1/t[:,:1],rtol=1e-6)
    np.testing.assert_allclose(changed[:,:1],10*t[:,:1])
    np.testing.assert_array_equal(reduce_background(t,1),t.astype(np.float32))


def test_dem_marginal_parity():
    c=stim.Circuit('R 0\nX_ERROR(0.1) 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]')
    np.testing.assert_allclose(detector_marginals(c),[.26])


def test_dem_optimized_accumulation_exactly_matches_original():
    c=stim.Circuit.generated('surface_code:rotated_memory_z',distance=3,rounds=16,
        after_clifford_depolarization=.0012345678912345,before_measure_flip_probability=.00234567891)
    ref=np.zeros(c.num_detectors)
    for instruction in c.detector_error_model(approximate_disjoint_errors=True).flattened():
        if instruction.type!='error':continue
        targets=[t.val for t in instruction.targets_copy() if t.is_relative_detector_id()]
        ids,counts=np.unique(targets,return_counts=True)
        ref[ids[counts%2==1].astype(int)]+=np.log1p(-2*instruction.args_copy()[0])
    np.testing.assert_array_equal(detector_marginals(c),-.5*np.expm1(ref))


def test_batched_circuit_is_exact_original_append_order(monkeypatch):
    from qp_ode_simulator import circuit_building
    from qp_ode_simulator.stim_qec import build_stim_layout,round_tick_counts,inject_scheduled_gate_slice_pauli_channels
    config=dict(task='surface_code:rotated_memory_z',distance=3,rounds=16,round_duration_ms=.001,start_time_ms=0,
        circuit_noise={'before_measure_flip_probability':.0012345678912345})
    layout=build_stim_layout(config);counts=round_tick_counts(layout.circuit,16)
    rng=np.random.default_rng(2);p=rng.uniform(1e-9,.001,(16,counts.max(),len(layout.qubit_ids),3))
    args=(layout.circuit,layout.qubit_ids,p[:,:,:,0],p[:,:,:,1],p[:,:,:,2],counts)
    actual=inject_scheduled_gate_slice_pauli_channels(*args)
    class Reference:
        def __init__(self):self.c=stim.Circuit()
        def append(self,*args):self.c.append(*args)
        def finish(self):return self.c
    monkeypatch.setattr(circuit_building,'CircuitAccumulator',Reference)
    expected=inject_scheduled_gate_slice_pauli_channels(*args)
    assert actual==expected
    np.testing.assert_array_equal(actual.compile_sampler(seed=123).sample(10),expected.compile_sampler(seed=123).sample(10))


def test_bins_keep_short_final_bin():
    from qp_ode_simulator.weak_observation_fit import binned
    sums,lengths=binned(np.ones((2,8,65),dtype=np.uint8))
    np.testing.assert_array_equal(lengths,[32,32,1])
    np.testing.assert_array_equal(sums.sum(-1),65)


def test_prediction_has_no_future_or_query_truth_inputs():
    from qp_ode_simulator.weak_observation_fit import predict_case
    class Constant:
        def predict(self,x):return np.zeros((len(x),2))
    rng=np.random.default_rng(3);raw=rng.binomial(1,.03,(2,8,4095)).astype(np.uint8)
    a=dict(rounds=2048,quiet=np.full(8,.02),fixed={'fixed_ridge':Constant(),'fixed_svr':Constant()},
        forests=[Constant()],aligned={'aligned_ridge':Constant(),'aligned_svr':Constant()},
        template_probabilities=np.full((2,8,64),.02),template_coordinates=np.array([[-1.,0],[1.,0]]),prior=np.zeros(2),noise_whitener=np.eye(8))
    before=predict_case(a,raw)
    raw[:,:,2047:]=1-raw[:,:,2047:]
    after=predict_case(a,raw)
    short=predict_case(a,raw[:,:,:2047])
    assert len(before)==33
    for name in before:
        np.testing.assert_allclose(before[name],after[name],atol=1e-10)
        np.testing.assert_allclose(before[name],short[name],atol=1e-10)


def test_heldout_latent_template_access_rejected(tmp_path):
    from qp_ode_simulator.dataset import _write_json
    from qp_ode_simulator.weak_observation_fit import read_events
    (tmp_path/'events').mkdir()
    _write_json(tmp_path/'events/event_0000.json',dict(role='test'))
    with pytest.raises(ValueError,match='held-out latent'):
        read_events(tmp_path,[0],'nominal',2048,templates=True)


def test_weak_split_balanced_and_disjoint():
    from qp_ode_simulator.weak_observation import split_events
    base=dict(shape={},propagation={},epicenter={},temporal_model={'qp_ode':{}})
    plan=dict(geometry=['circular','elliptical'],propagation_law=['ballistic','diffusive'],
        epicenter_region=['inside','edge','outside'],generation_bands_per_us=[[1e-10,1e-9]])
    s=split_events(base,plan)
    assert {k:len(v) for k,v in s.items()}==dict(train=432,validation=144,test=144)
    assert len(set(s['train']+s['validation']+s['test']))==720
