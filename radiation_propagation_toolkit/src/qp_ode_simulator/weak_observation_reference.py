"""Supplementary audit against previously frozen general-distribution models."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .adaptive_localization import segmented_features
from .attention_mdn_gp import MixturePosition,nn_predict
from .weak_observation_fit import read_events
from .growth_more_models import paired_delta
from .dataset import _write_json
from .localization import _hash_file


def run(root):
    root=Path(root).resolve();out=root/'reference_audit';out.mkdir(exist_ok=False)
    previous=root.parent/'weak_growth_20260908/experiment'
    selected=json.loads((previous/'selected_before_test.json').read_text())
    assert selected['overall']=='mdn_n3600_w1'
    paths=[previous/'models.joblib',*sorted((previous/'mdn_n3600_w1').glob('*'))]
    _write_json(out/'protocol.json',dict(status='supplementary comparison after primary experiment; old baselines chosen by their pre-existing frozen protocol, not current test rank',
        source_sha256=_hash_file(Path(__file__)),references=['previous_mdn3600','previous_blend3600','previous_prior'],
        old_artifact_hashes={str(p):_hash_file(p) for p in paths},input='nominal 2048-round prefix only; no query physical truth'))
    ids=json.loads((root/'protocol.json').read_text())['split']['test']
    raw,rows,_,sites=read_events(root,ids,'nominal',2048)
    artifact=joblib.load(previous/'models.joblib');coords=np.asarray(artifact['specification']['geometry']['detector_coords'])
    np.testing.assert_array_equal(sites,np.unique(coords[(coords[:,2]>=1)&(coords[:,2]<2048),:2],axis=0))
    x=segmented_features(raw,np.full(len(raw),artifact['fixed_features']['cut']),artifact['fixed_features']['quiet'])
    svr=artifact['regressors']['svr_n3600'].predict(x)
    et=np.mean([m.predict(x) for m in artifact['forests']['et_n3600']],axis=0)
    preds=dict(previous_blend3600=.5*(svr+et),previous_prior=np.broadcast_to(artifact['prior'],(len(raw),2)).copy())
    torch.set_num_threads(2);folder=previous/'mdn_n3600_w1';pre=joblib.load(folder/'preprocessing.joblib');data=pre['scaler'].transform(x)
    ps=[]
    for path in sorted(folder.glob('seed*.pt')):
        model=MixturePosition(4,64);model.load_state_dict(torch.load(path,weights_only=True))
        p,_=nn_predict(model,data,pre['center'],pre['scale']);ps.append(p)
    preds['previous_mdn3600']=np.mean(ps,axis=0)
    truth=rows[['epicenter_row','epicenter_col']].to_numpy();frames=[]
    for name,p in preds.items():
        f=rows[['event_uid','shot']].copy();f['candidate']=name
        f['predicted_x_mm'],f['predicted_y_mm']=p.T;f['error_mm']=np.linalg.norm(p-truth,axis=1);frames.append(f)
    refs=pd.concat(frames,ignore_index=True);refs.to_csv(out/'predictions.csv',index=False)
    primary=pd.read_csv(root/'experiment/predictions.csv');selections=json.loads((root/'experiment/selected_before_test.json').read_text())
    contrasts=[]
    for case in ['nominal_r2048',selections['best_nominal_case'],'quiet10_r4096','quiet10_r8192']:
        name=selections['cases'][case]['selected']['overall'];a=primary[(primary.case==case)&(primary.candidate==name)]
        for reference in preds:
            b=refs[refs.candidate==reference]
            contrasts.append(dict(case=case,candidate=name,reference=reference,**paired_delta(a,b)))
    _write_json(out/'paired.json',contrasts)
    metrics=refs.groupby('candidate').error_mm.agg(['mean','median',lambda v:v.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    print(metrics.to_string(),flush=True);print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
