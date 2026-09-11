"""Syndrome-only inference for a frozen medium-event experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from . import medium_inverse as study


def predict(root,raw,name=None):
    root=Path(root);p=study.check(root);s=study.read(root/'selection.json');name=name or s['primary']
    assert s['protocol_hash']==study._hash_file(root/'protocol.json')
    assert s['source_hash']==study._hash_file(Path(study.__file__))
    filename=f'{name}.joblib'
    if filename not in s['hashes']:raise ValueError('Unknown frozen model')
    assert study._hash_file(root/filename)==s['hashes'][filename]
    h=4 if name=='reference' else int(name.split('_')[1][1:])
    raw=np.asarray(raw)
    if raw.ndim!=3 or raw.shape[1]!=24 or raw.shape[-1]<h*1024-1 or len(raw)==0:
        raise ValueError('Expected nonempty [shot,24check,time] with enough rounds')
    raw=raw[...,:h*1024-1]
    if not np.isin(raw,[0,1]).all():raise ValueError('Detector records must be binary')
    model=joblib.load(root/filename)
    if name.endswith('SVR'):return model.predict(study.relative(raw,h,p['quiet']))
    return study.template_predict(study.rates(raw,h),model)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('observed_npz',type=Path);p.add_argument('output_csv',type=Path);p.add_argument('--model');a=p.parse_args()
    with np.load(a.observed_npz) as z:xy=predict(a.root,z['raw'],a.model)
    pd.DataFrame(xy,columns=['x_mm','y_mm']).to_csv(a.output_csv,index=False,mode='x')
