import numpy as np
import pytest
from qp_ode_simulator.specialist_study import mix_experts, POOLS


def test_expert_routing_endpoints_and_middle():
    experts={'weak':np.zeros((3,2)),'medium':np.full((3,2),4.),'strong':np.full((3,2),10.),
             'low_overlap':np.full((3,2),2.),'high_overlap':np.full((3,2),8.)}
    np.testing.assert_array_equal(mix_experts(experts,np.eye(3),'strict_two'),[[0,0],[5,5],[10,10]])
    np.testing.assert_array_equal(mix_experts(experts,np.eye(3),'three'),[[0,0],[4,4],[10,10]])
    np.testing.assert_array_equal(mix_experts(experts,np.eye(3),'overlap_two'),[[2,2],[5,5],[8,8]])
    probs=np.tile([.2,.3,.5],(3,1))
    np.testing.assert_allclose(mix_experts(experts,probs,'three'),6.2)


def test_disjoint_and_overlapping_training_pools():
    assert not set(POOLS['weak']) & set(POOLS['strong'])
    assert set(POOLS['low_overlap']) & set(POOLS['high_overlap']) == {1}


def test_invalid_gate_probabilities_rejected():
    with pytest.raises(ValueError): mix_experts({},[[.3,.4,.4]],'three')
    with pytest.raises(ValueError): mix_experts({},[[-.1,.6,.5]],'three')
