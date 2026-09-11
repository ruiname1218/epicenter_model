"""Predict with frozen pilot artifacts, without reading any event truth labels."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from . import radical_models
from .radical_data import read,_hash_file,bin_rates,relative_features,local_coincidences
from .radical_models import ARMS,SEEDS,temporal_predict,inverse_predict


def predict(root,observed,model=None):
    """observed maps layout names to binary [shot, check, 4095] arrays.

    Does not read fit/test rows, nuisance labels, or query conditional means.
    Frozen training templates are allowed; query-specific truth is not.
    """
    root=Path(root);s=read(root/'selection.json');name=model or s['selected']
    allowed={'base_SVR','base_ET','dense_SVR','dense_ET','pair_SVR','pair_ET','forward_inverse','template_inverse',*ARMS}
    if name not in allowed:raise ValueError('Only operational models are supported, never oracle inputs')
    assert _hash_file(Path(radical_models.__file__))==s['source_hash']
    assert _hash_file(root/'protocol.json')==s['protocol_hash']
    assert _hash_file(root/'quiet.json')==s['quiet_hash']
    needed=['dense'] if name.startswith('dense_') else ['base','right'] if name.startswith('pair_') else ['base']
    for layout in needed:
        a=np.asarray(observed[layout]);checks=48 if layout=='dense' else 24
        if a.ndim!=3 or a.shape[1:]!=(checks,4095) or len(a)==0 or not np.isin(a,[0,1]).all():
            raise ValueError('Expected nonempty binary [shot,check,4095] detector array')
    if len({len(observed[k]) for k in needed})!=1:raise ValueError('Shot counts differ between patches')
    def artifact(key):
        file=f'{key}.joblib';assert _hash_file(root/file)==s['hashes'][file];return joblib.load(root/file)
    if name in ['forward_inverse','template_inverse']:
        return inverse_predict(bin_rates(observed['base'],16),**artifact(name))
    if name in ARMS:
        torch.set_num_threads(2);p=read(root/'protocol.json');raw=observed['base']
        rates=bin_rates(raw,128);coinc=local_coincidences(raw,p['sites']['base'])
        return np.mean([temporal_predict(artifact(f'{name}_{seed}'),rates,coinc) for seed in SEEDS],axis=0)
    quiet=read(root/'quiet.json');raw=np.concatenate([observed[k] for k in needed],axis=1)
    baseline=sum([quiet[k] for k in needed],[])
    return artifact(name).predict(relative_features(raw,baseline))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('observed_npz',type=Path);p.add_argument('output_csv',type=Path);p.add_argument('--model');a=p.parse_args()
    with np.load(a.observed_npz) as z:pred=predict(a.root,z,a.model)
    pd.DataFrame(pred,columns=['x_mm','y_mm']).to_csv(a.output_csv,index=False,mode='x')
