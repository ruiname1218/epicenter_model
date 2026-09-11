"""Before test freezing, add the old-selected settings to every input ablation.

Recorded as a validation-stage amendment, not a preregistered confirmatory trial.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file
from qp_ode_simulator.syndrome_features_study import VARIANTS


def fit(args):
    root, variant = args; out = root/variant
    deadline = time.monotonic()+3600
    while not (out/'frozen.json').exists():
        if time.monotonic() > deadline: raise TimeoutError('Initial training did not finish: '+variant)
        time.sleep(1)
    info = json.loads((out/'frozen.json').read_text())
    if 'SVR_parent' in info['logs']: return variant
    rows = pd.read_csv(root/'rows.csv'); x = np.load(root/f'features_{variant}.npy')
    y = rows[['epicenter_row', 'epicenter_col']].to_numpy()
    train = np.flatnonzero(rows.role == 'train'); val = np.flatnonzero(rows.role == 'validation')
    weak = rows.iloc[val].strength_band.to_numpy() == 0
    with np.load(out/'validation.npz') as z: preds = {k:z[k] for k in z.files}
    score = lambda pred: {'ordinary':float(np.linalg.norm(pred-y[val],axis=1).mean()), 'weak':float(np.linalg.norm(pred[weak]-y[val][weak],axis=1).mean())}
    for name, config in [('SVR_parent', {'C':1., 'gamma':.57/x.shape[1]}), ('ET_parent', {'min_samples_leaf':3})]:
        start = time.monotonic()
        if name.startswith('SVR'): model = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(**config, epsilon=.1)))
        else: model = ExtraTreesRegressor(n_estimators=256, random_state=41, n_jobs=2, **config)
        model.fit(x[train], y[train]); preds[name] = model.predict(x[val]); joblib.dump(model, out/f'{name}.joblib')
        info['logs'][name] = dict(**score(preds[name]), config=config, seconds=time.monotonic()-start)
        print(variant, name, info['logs'][name], flush=True)
    svr = min([k for k in preds if k.startswith('SVR')], key=lambda k:info['logs'][k]['ordinary'])
    et = min(['ET8', 'ET16', 'ET_parent'], key=lambda k:info['logs'][k]['ordinary'])
    info['blend'] = [svr, et]; preds['blend'] = .5*(preds[svr]+preds[et]); info['logs']['blend'] = score(preds['blend'])
    np.savez_compressed(out/'validation.npz', **preds)
    info['hashes'] = {f.name:_hash_file(f) for f in out.glob('*.joblib')}
    _write_json(out/'frozen.json', info)
    return variant


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    assert not (root/'selection.json').exists() and not (root/'events').exists()
    amendment = {'stage':'exploratory validation only; no new test generated or read',
        'reason':'Initial search omitted old-selected C1/high-gamma SVR and leaf3 ET. Add equal-budget matched controls to all five input variants before final selection.',
        'added_settings':{'SVR':{'C':1, 'gamma':'.57 / feature_dimension'}, 'ExtraTrees':{'min_samples_leaf':3}},
        'total_fits':45, 'source_hash':_hash_file(Path(__file__))}
    _write_json(root/'validation_amendment.json', amendment)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for name in pool.map(fit, [(root, v) for v in VARIANTS]): print('Extended',name,flush=True)
