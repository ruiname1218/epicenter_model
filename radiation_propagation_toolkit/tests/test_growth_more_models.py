import pandas as pd
import pytest
from qp_ode_simulator.growth_more_models import paired_delta, tree_model


def test_no_implicit_shot_validation():
    model=tree_model('hist_gradient_boosting',8,41)
    assert model.estimator.early_stopping is False
    assert tree_model('extra_trees',12,41).min_samples_leaf==12


def test_pairing_rejects_different_shots():
    a=pd.DataFrame(dict(event_uid=['a'],shot=[0],error_mm=[1.]))
    b=a.assign(shot=1)
    with pytest.raises(ValueError,match='identity'):
        paired_delta(a,b)


def test_pairing_aggregates_seed_and_shot_within_event():
    a=pd.DataFrame(dict(event_uid=['a','a','a','a'],shot=[0,1,0,1],error_mm=[2.,4.,2.,4.]))
    b=pd.DataFrame(dict(event_uid=['a','a'],shot=[0,1],error_mm=[1.,3.]))
    result=paired_delta(a,b)
    assert result['difference_mm']==1.
    assert result['ci95_mm']==[1.,1.]
