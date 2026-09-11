"""Cached, CPU single-thread, one-shot end-to-end inference microbenchmark.

Includes feature extraction; excludes loading, fitting and data acquisition.
Measures this Python implementation, not an optimized online system.
"""
import json
import os
from pathlib import Path
import platform
import sys
import time
import joblib
import numpy as np
from qp_ode_simulator.error_routing_study import SEEDS,BANK,gate_predict,verify
from qp_ode_simulator.distance_growth import features
from qp_ode_simulator.temporal_diagnosis import rei_center
from qp_ode_simulator.dataset import _write_json

root=Path(sys.argv[1]).resolve();p,s=verify(root);parent=Path(p['benchmark_parent'])
det={n:joblib.load(parent/f'{n}.joblib') for n in ['ridge','svr','medium','strong']}
middle=joblib.load(root/'medium_svr.joblib')
et={seed:{n:joblib.load(parent/f'seed{seed}'/f'{n}.joblib') for n in ['global','weak','strong','gate']} for seed in SEEDS}
for models in et.values():
    for model in models.values():model.n_jobs=1
states=joblib.load(root/f'{s["unconstrained"]}.joblib') if s['unconstrained'].startswith('ErrorGate') else None
geo=p['geometry']['distances']['5']

def predict(raw,name):
    if name=='REI':
        z=rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=s['history'],circuit_repetitions=1)
        return np.where(np.isfinite(z).all(1)[:,None],z,s['center'])
    x=features(raw,p['preprocessing']);svr=det['svr'].predict(x)
    if name=='SVR':return svr
    mid=middle.predict(x) if name=='MiddleOnlySVR' else det['medium'].predict(x)
    strsvr=det['strong'].predict(x);hard=[];globals_=[]
    for seed,models in et.items():
        global_=models['global'].predict(x);weak=models['weak'].predict(x)
        strong=.5*(models['strong'].predict(x)+strsvr)
        gate=np.eye(3)[models['gate'].predict_proba(x).argmax(1)]
        hard.append(.5*(.5*(svr+global_)+gate[:,0,None]*weak+gate[:,1,None]*mid+gate[:,2,None]*strong))
        globals_.append(global_)
    hard=np.mean(hard,axis=0)
    if name.startswith('ErrorGate'):
        bank=np.stack([det['ridge'].predict(x),svr,np.mean(globals_,axis=0),hard],axis=1)
        return np.mean([gate_predict(state,x,bank) for state in states],axis=0)
    return hard

names=['REI','SVR','ThreeHard','MiddleOnlySVR']
if states is not None:names.append(s['unconstrained'])
raw=[]
for i in np.linspace(0,1079,16,dtype=int):
    with np.load(root/'events'/f'{i:04d}.npz') as z:raw.append(z['d5'][:1])
times={n:[] for n in names}
# Warm caches and verify against saved predictions, not just execution success.
import pandas as pd
saved=pd.read_csv(root/'predictions.csv')
for i,z in zip(np.linspace(0,1079,16,dtype=int),raw):
    uid=json.loads((root/'events'/f'{i:04d}.json').read_text())['event_uid']
    for name in names:
        ref=saved[(saved.event_uid==uid)&(saved.shot==0)&(saved.model==name)][['pred_x','pred_y']].to_numpy()
        np.testing.assert_allclose(predict(z,name),ref,rtol=1e-5,atol=1e-6)
for repetition in range(3):
    for z in raw:
        for name in np.random.default_rng(20260910+repetition).permutation(names):
            start=time.perf_counter();predict(z,name);times[name].append(1000*(time.perf_counter()-start))
result=dict(environment=dict(python=platform.python_version(),machine=platform.machine(),cpu_count=os.cpu_count(),tree_jobs=1),
    scope='48 timed calls/model, 16 physical events x3 repeats, one shot each, cached models, includes feature extraction, excludes loading/acquisition/training. Other system workloads may affect wall time; not real-time certification.',
    selected=s['selected'],prediction_equivalence='passed',
    models={n:dict(median_ms=float(np.median(v)),p90_ms=float(np.quantile(v,.9)),mean_ms=float(np.mean(v))) for n,v in times.items()})
_write_json(root/'inference_timing.json',result);print(json.dumps(result,indent=2))
