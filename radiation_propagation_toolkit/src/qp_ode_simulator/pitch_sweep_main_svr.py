"""Main-protocol SVR-only pitch/density sweep at fixed d=5."""
from __future__ import annotations
import argparse, copy, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .api import run_simulation
from .dataset import event_configuration, _seed, _write_json, _generation_lock
from .fault_response import build_response, sample
from .distance_study import geometry, fields_to_pauli
from .localization import _hash_file
from .radical_data import relative_features
from .specialist_study import read

LEVELS={'ultra_low':1.50,'low':1.25,'standard':1.00,'high':5/6,'ultra_high':5/7}
# Compact density-sweep sample size: keep the main protocol and SVR settings,
# while reducing the number of expensive ODE simulations for a timely pilot.
TRAIN_SEED=20260925101;TEST_SEEDS=(20260925201,20260925301,20260925401);NTRAIN,NVAL,NTEST=180,60,60
def split(i):
    if i<NTRAIN:return TRAIN_SEED,i,'train'
    if i<NTRAIN+NVAL:return TRAIN_SEED,i,'validation'
    j=i-NTRAIN-NVAL;return TEST_SEEDS[j//NTEST],j%NTEST,'test'
def prepare(root):
    root=Path(root);root.mkdir(exist_ok=False);(root/'events').mkdir();inh=read(root.parent/'radical_pilot_20260911/protocol.json');base=copy.deepcopy(inh['base']);base['time']['end_ms']=8.196;circuits={}
    for n,pitch in LEVELS.items():
        c=copy.deepcopy(inh['circuits']['base']);c['distance']=5;c['rounds']=8192;c['layout'].update(qubit_pitch_mm=pitch,center_mm=(0.,0.));c.pop('seed',None);circuits[n]=c
    layouts={n:geometry(c)[0] for n,c in circuits.items()};union=np.unique(np.concatenate([np.round(v.physical_coords_mm,9) for v in layouts.values()]),axis=0);indices={n:np.array([np.flatnonzero(np.all(union==np.round(q,9),axis=1))[0] for q in l.physical_coords_mm]) for n,l in layouts.items()};responses={n:build_response(c) for n,c in circuits.items()}
    p=dict(base=base,sampling=inh['sampling'],circuits=circuits,union=union.tolist(),indices={n:v.tolist() for n,v in indices.items()},responses=responses,events=NTRAIN+NVAL+3*NTEST,levels=LEVELS,model='StandardScaler + independent x/y RBF SVR, C=1, epsilon=0.1, gamma=0.57/120, same main-protocol split and features.',source_hash=_hash_file(Path(__file__)));_write_json(root/'protocol.json',p)
    quiet={};cfg,_=event_configuration(base,inh['sampling'],0,TRAIN_SEED+1000,123);sim=run_simulation(cfg,coords_mm=union)
    for j,n in enumerate(LEVELS):
        xyz=fields_to_pauli(sim,circuits[n],indices[n],quiet=True);rr=[]
        for k in range(8):
            raw,_=sample(responses[n],circuits[n],*xyz,seed=_seed(TRAIN_SEED+2000,j,k)%(2**63-1),shots=16);rr.append(raw.mean((0,2)))
        quiet[n]=np.mean(rr,axis=0).tolist()
    _write_json(root/'quiet.json',quiet)
_W=None
def initialize(root):
    global _W;root=Path(root);_W=(root,read(root/'protocol.json'))
def event_job(i):
    root,p=_W;seed,local,role=split(i);cfg,label=event_configuration(p['base'],p['sampling'],local,seed,123);sim=run_simulation(cfg,coords_mm=np.asarray(p['union']));payload={}
    for j,n in enumerate(LEVELS):
        raw,_=sample(p['responses'][n],p['circuits'][n],*fields_to_pauli(sim,p['circuits'][n],np.asarray(p['indices'][n])),seed=_seed(seed+1,local,j)%(2**63-1),shots=2);payload[n]=raw
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()};row.update(label,event=i,event_uid=f'pitch-main-{seed}:{local}',role=role,test_seed=seed);path=root/'events'/f'{i:04d}.npz';tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**payload)
    tmp.replace(path);_write_json(path.with_suffix('.json'),row);return i
def generate(root,workers=8):
    root=Path(root);total=read(root/'protocol.json')['events']
    with _generation_lock(root):
        done=[i for i in range(total) if (root/'events'/f'{i:04d}.npz').exists()]
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),)) as pool:
            fs=[pool.submit(event_job,i) for i in range(total) if i not in done]
            for f in as_completed(fs):
                done.append(f.result());
                if len(done)%36==0:print('Generated',len(done),'/',total,flush=True)
        _write_json(root/'manifest.json',dict(hashes={f'{i:04d}':_hash_file(root/'events'/f'{i:04d}.npz') for i in done}))
def cache(root):
    root=Path(root);p=read(root/'protocol.json');q=read(root/'quiet.json');rows=[];data={n:[] for n in LEVELS}
    for i in range(p['events']):
        path=root/'events'/f'{i:04d}.npz';row=read(path.with_suffix('.json'))
        with np.load(path) as z:
            for n in LEVELS:data[n].append(relative_features(z[n],q[n]))
        rows.extend([dict(row,shot=j) for j in range(2)])
    pd.DataFrame(rows).to_csv(root/'rows.csv',index=False)
    for n,v in data.items():np.save(root/f'x_{n}.npy',np.concatenate(v))
    print('Cached',len(rows),'shots')
def fit(root):
    root=Path(root);rows=pd.read_csv(root/'rows.csv');y=rows[['epicenter_row','epicenter_col']].to_numpy();tr=np.flatnonzero(rows.role=='train');va=np.flatnonzero(rows.role=='validation');sel={}
    for n in LEVELS:
        x=np.load(root/f'x_{n}.npy');m=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=1,epsilon=.1,gamma=.57/x.shape[1])));m.fit(x[tr],y[tr]);e=np.linalg.norm(m.predict(x[va])-y[va],axis=1);joblib.dump(m,root/f'{n}_SVR.joblib');sel[n]=dict(validation_mean_mm=float(e.mean()),features=x.shape[1],hash=_hash_file(root/f'{n}_SVR.joblib'));print(n,float(e.mean()))
    _write_json(root/'selection.json',sel)
def evaluate(root):
    root=Path(root);rows=pd.read_csv(root/'rows.csv');y=rows[['epicenter_row','epicenter_col']].to_numpy();mask=rows.role=='test';frames=[]
    for n in LEVELS:
        pred=joblib.load(root/f'{n}_SVR.joblib').predict(np.load(root/f'x_{n}.npy')[mask]);r=rows.loc[mask].copy();r['pitch_level']=n;r['pitch_mm']=LEVELS[n];r['error_mm']=np.linalg.norm(pred-y[mask],axis=1);frames.append(r)
    f=pd.concat(frames,ignore_index=True);f.to_csv(root/'predictions.csv',index=False);out=[]
    for n,g in f.groupby('pitch_level'):
        for name,m in {'all':np.ones(len(g),bool),'weak':g.strength_band==0,'medium':g.strength_band==1,'strong':g.strength_band==2}.items():
            e=g.loc[m,'error_mm'];out.append(dict(pitch_level=n,pitch_mm=LEVELS[n],group=name,events=g.loc[m,'event'].nunique(),mean_mm=e.mean(),median_mm=e.median(),p90_mm=e.quantile(.9),within_1mm=(e<=1).mean()))
    pd.DataFrame(out).to_csv(root/'metrics.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','generate','cache','fit','evaluate']);ap.add_argument('root',type=Path);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();{'prepare':prepare,'generate':lambda r:generate(r,a.workers),'cache':cache,'fit':fit,'evaluate':evaluate}[a.stage](a.root)
