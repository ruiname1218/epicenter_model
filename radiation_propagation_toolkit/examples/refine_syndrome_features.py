"""Exploratory validation refinements, before fresh test generation."""
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.localization import _hash_file
from qp_ode_simulator.feature_refinements import WeakSignalGate


def refine(args):
    root, variant = args; out = root/variant
    deadline = time.monotonic()+3600
    while not (out/'frozen.json').exists() or 'SVR_parent' not in json.loads((out/'frozen.json').read_text())['logs']:
        if time.monotonic() > deadline: raise TimeoutError(variant)
        time.sleep(1)
    info = json.loads((out/'frozen.json').read_text())
    rows = pd.read_csv(root/'rows.csv'); x = np.load(root/f'features_{variant}.npy')
    y = rows[['epicenter_row','epicenter_col']].to_numpy()
    train = np.flatnonzero(rows.role == 'train'); val = np.flatnonzero(rows.role == 'validation')
    weak = rows.iloc[val].strength_band.to_numpy() == 0
    with np.load(out/'validation.npz') as z: preds = {k:z[k] for k in z.files}
    def record(name, model):
        pred = model.predict(x[val]); preds[name] = pred
        e = np.linalg.norm(pred-y[val], axis=1)
        info['logs'][name] = {'ordinary':float(e.mean()), 'weak':float(e[weak].mean())}
        joblib.dump(model, out/f'{name}.joblib')
        print(variant, name, info['logs'][name], flush=True)
    if variant != 'baseline':
        for components in (32,128):
            for c, factor in ((10,.057),(1,.57)):
                name = f'PCA{components}_SVR{c}'
                model = make_pipeline(StandardScaler(), PCA(n_components=components, svd_solver='randomized', random_state=41),
                    MultiOutputRegressor(SVR(C=c,gamma=factor/components,epsilon=.1)))
                model.fit(x[train],y[train]); record(name,model)
    else:
        parent = Path(json.loads((root/'protocol.json').read_text())['parent'])/'n2880'
        svr = joblib.load(parent/'SVR.joblib'); et = joblib.load(parent/'ExtraTrees.joblib')
        center = json.loads((parent/'frozen.json').read_text())['center']
        expert = joblib.load(out/'ET_weak.joblib')
        labels = (rows.iloc[train].strength_band.to_numpy() == 0).astype(int)
        for name, classifier in [('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))),
                                  ('ET16',ExtraTreesClassifier(n_estimators=256,min_samples_leaf=16,random_state=41,n_jobs=2)),
                                  ('ET64',ExtraTreesClassifier(n_estimators=256,min_samples_leaf=64,random_state=41,n_jobs=2))]:
            classifier.fit(x[train],labels)
            for target, model in [('center',None),('expert',expert)]:
                for fraction in (.5,1.):
                    for power in (1,2):
                        key=f'learned_{name}_{target}_{fraction}_{power}'
                        record(key,WeakSignalGate(classifier,svr,et,center,model,fraction,power))
    np.savez_compressed(out/'validation.npz', **preds)
    info['hashes'] = {f.name:_hash_file(f) for f in out.glob('*.joblib')}; _write_json(out/'frozen.json',info)
    return variant


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    assert not (root/'selection.json').exists() and not (root/'events').exists()
    _write_json(root/'refinement_amendment.json',dict(stage='Validation only; no fresh test generated or read',
        reason='Initial feature additions did not improve ordinary validation. Try training-only PCA denoising and observable learned weak-signal gates.',
        PCA='multiscale/combined; 32/128 components; C10 gamma .057/components or C1 gamma .57/components; 8 fits',
        gate='baseline features; LogisticRegression C1 or ExtraTrees leaf16/64; 3 classifier fits; 24 mixtures of parent with prior/weak expert; weights 0.5/1 * P(weak)^1/2',
        total_fits=56, source_hash=_hash_file(Path(__file__))))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for variant in pool.map(refine,[(root,v) for v in ('baseline','multiscale','combined')]): print('Refined',variant,flush=True)
