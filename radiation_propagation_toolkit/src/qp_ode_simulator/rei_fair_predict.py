"""Common raw-syndrome interface for the frozen fair-comparison models."""
from pathlib import Path
import numpy as np
import joblib
from .rei_fair_data import read,HORIZONS,features,bin_rates
from .rei_fair_models import verify,cnn_predict,rei_rates,SEEDS


def predict(root,raw,horizon=4,family='SVR'):
    root=Path(root);s=verify(root);p=read(root/'protocol.json')
    if horizon not in HORIZONS:raise ValueError('Unsupported observation time')
    if family not in ['SVR','SVR_noquiet','Ridge','CNN','REI_full','REI_1024','REI_valK']:raise ValueError('Unknown model')
    ticks=horizon*1024-1;raw=np.asarray(raw)
    if raw.ndim!=3 or raw.shape[1]!=24 or raw.shape[-1]<ticks or len(raw)==0:raise ValueError('Expected [shot,24check,time] with enough rounds')
    raw=raw[...,:ticks]
    if not np.isin(raw,[0,1]).all():raise ValueError('Detector events must be binary')
    if family.startswith('REI'):
        k=ticks if family=='REI_full' else 1024 if family=='REI_1024' else int(s['rei_history'][str(horizon)]['selected'])
        xy=rei_rates(raw[...,-k:].mean(-1),p['sites'],p['physical'],k);valid=np.isfinite(xy).all(1)
        return np.where(valid[:,None],xy,s['center']),valid
    if family=='CNN':
        x=bin_rates(raw,128);xy=np.mean([cnn_predict(joblib.load(root/f'h{horizon}_CNN_{seed}.joblib'),x) for seed in SEEDS],axis=0)
    else:
        x=features(raw,horizon,None if family=='SVR_noquiet' else read(root/'quiet.json')['rates'])
        xy=joblib.load(root/f'h{horizon}_{family}.joblib').predict(x)
    return xy,np.ones(len(raw),bool)
