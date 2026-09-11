import numpy as np
import torch
import pandas as pd
import pytest
import json
from qp_ode_simulator.radical_data import spec,bin_rates,local_coincidences,relative_features,nuisance,TEST_SEEDS
from qp_ode_simulator.radical_models import Temporal,temporal_input,temporal_predict,inverse_predict,regression_inputs


def test_split_roots_and_roles():
    assert spec(0)[2]=='train' and spec(863)[2]=='train'
    assert spec(864)[2]=='validation' and spec(1007)[2]=='validation'
    assert {spec(i)[0] for i in range(1008,1548)}==set(TEST_SEEDS)
    assert len({spec(i)[:2] for i in range(1548)})==1548


def test_binning_and_coincidences():
    raw=np.ones((2,4,4095),dtype=np.uint8)
    assert bin_rates(raw,128).shape==(2,4,128)
    sites=np.array([[0,0],[1,0],[0,1],[1,1]])
    np.testing.assert_array_equal(local_coincidences(raw,sites),1)
    raw[:,0]=0
    assert np.all(local_coincidences(raw,sites)[:,0]==0)
    assert relative_features(raw,[0]*4).shape==(2,20)


def test_temporal_operational_inputs_and_matched_capacity():
    torch.set_num_threads(1);torch.manual_seed(1)
    rates=np.random.default_rng(1).random((2,24,128)).astype(np.float32)
    coinc=np.zeros_like(rates)
    x,mean,std=temporal_input(rates,coinc,'coinc_aux')
    model=Temporal();bundle=dict(state=model.state_dict(),arm='coinc_aux',mean=mean,std=std)
    pred=temporal_predict(bundle,rates,coinc)
    assert pred.shape==(2,2) and np.isfinite(pred).all()
    # Privileged targets can be shuffled without ever being passed to inference.
    bundle['unused_true_nuisance']=np.ones((2,17))*999
    np.testing.assert_array_equal(pred,temporal_predict(bundle,rates,coinc))
    plain,_,_=temporal_input(rates,np.ones_like(coinc),'rates_plain')
    np.testing.assert_array_equal(plain[:,24:],0)
    assert model(torch.from_numpy(x))[2].shape==(2,384)


def test_inverse_score_prefers_matching_profile_and_zero_information_prior():
    profiles=np.stack([np.full((24,16),.02),np.full((24,16),.2)])
    centers=np.array([[-2.,1.],[3.,-1.]])
    np.testing.assert_allclose(inverse_predict(profiles,profiles,centers,1),centers,atol=1e-6)
    same=np.repeat(profiles[:1],2,axis=0)
    np.testing.assert_allclose(inverse_predict(profiles[:1],same,centers,1),centers.mean(0)[None])


def test_privileged_features_are_separate_from_observed_regression_inputs():
    data={k:np.ones((2,120)) for k in ['base','dense','pair']}
    data.update(nuisance=np.ones((2,17)),privileged_mean=np.ones((2,24,16)))
    before=regression_inputs(data,[0]*24)
    data['nuisance']*=7;data['privileged_mean']*=.03
    after=regression_inputs(data,[0]*24)
    for key in ['base','dense','pair']:np.testing.assert_array_equal(before[key],after[key])
    assert not np.array_equal(before['oracle_nuisance'],after['oracle_nuisance'])


def test_nuisance_excludes_coordinates_and_uses_axis_orientation():
    row={k:1. for k in ['qp_generation_scale_per_us','initial_lambda_mm','maximum_lambda_mm','maximum_distance_mm',
        'apparent_speed_m_per_s','diffusion_coefficient_mm2_per_ms','qp_source_lifetime_ms','qp_trapping_rate_per_us',
        'qp_recombination_rate_per_us','axis_ratio','event_onset_ms','front_width_ms','spread_time_ms']}
    row.update(angle_degrees=30,geometry='elliptical',propagation_law='diffusive',epicenter_row=0,epicenter_col=0)
    a=nuisance(pd.DataFrame([row]));row.update(epicenter_row=123,epicenter_col=-999,angle_degrees=210)
    np.testing.assert_allclose(a,nuisance(pd.DataFrame([row])),atol=1e-6)
    assert a.shape==(1,17) and a[0,10]==row['event_onset_ms']
    row.update(propagation_law='ballistic',diffusion_coefficient_mm2_per_ms=np.nan)
    assert np.isfinite(nuisance(pd.DataFrame([row]))).all()


def test_public_inference_rejects_oracle_without_reading_truth(tmp_path):
    from qp_ode_simulator.radical_predict import predict
    (tmp_path/'selection.json').write_text(json.dumps(dict(selected='oracle_mean_SVR')))
    with pytest.raises(ValueError,match='operational'):predict(tmp_path,{})
