"""Frozen multi-seed d5 benchmark, three independent test-generation roots."""
import argparse
import copy
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .api import run_simulation
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .distance_growth import features
from .distance_study import fields_to_pauli,predict_nn
from .fault_response import sample
from .localization import _hash_file
from .specialist_study import load_fit
from .temporal_localization import TemporalCNN
from .temporal_diagnosis import rei_center

SEEDS=(41,42,43)
TEST_SEEDS=(2026090941,2026091041,2026091141)
EVENTS=1080

def read(p):return json.loads(Path(p).read_text())

def masks(rows):
    b=rows.strength_band
    return {'weak':b==0,'medium':b==1,'strong':b==2,'strong_circle':(b==2)&(rows.geometry=='circular'),
        'strong_ellipse':(b==2)&(rows.geometry=='elliptical'),'all':np.ones(len(rows),bool)}

def summary(error):
    e=np.asarray(error)
    return dict(mean_mm=float(e.mean()),median_mm=float(np.median(e)),p90_mm=float(np.quantile(e,.9)),p95_mm=float(np.quantile(e,.95)),
        within_1mm=float((e<=1).mean()),over_3mm=float((e>3).mean()))

def prepare(root):
    root.mkdir(exist_ok=False)
    p=read(root.parent/'specialists_20260909/protocol.json')
    p.update(seed=TEST_SEEDS[0],test_seeds=list(TEST_SEEDS),training_seeds=list(SEEDS),events=EVENTS,
        test='Three generation roots, 360 independent events/root, 36 strata x10 cycles/root. Same 1080 events for all methods, 2 shots/event.',
        primary='Overlap ensemble minus adapted REI on medium and strong (both shapes), two co-primary means; event bootstrap97.5% intervals for Bonferroni two-comparison coverage. All other contrasts descriptive95%.',
        families=['Ridge','SVR','ExtraTrees','Blend','Overlap','ThreeSoft','ThreeHard','CNN','REI','Prior'],
        training='Same 2880 independent training events; hyperparameters fixed from prior validation. ET seeds41/42/43. Deterministic Ridge/SVR/logistic fit once. CNN uses previously frozen distance-growth 3 seed checkpoints, no new tuning.',
        reproducibility='Training-seed variability at fixed training data; three fresh test-generation roots. Not training-dataset resampling or hardware variation.',
        previous_protocol_hash=_hash_file(root.parent/'specialists_20260909/protocol.json'),source_hash=_hash_file(Path(__file__)))
    _write_json(root/'protocol.json',p)

def fit_seed(args):
    root,seed=args;rows,x,y=load_fit(root);tr=rows.role=='train';out=root/f'seed{seed}';out.mkdir(exist_ok=True)
    for name,bands,leaf in [('global',[0,1,2],3),('weak',[0],16),('strong',[2],3)]:
        idx=np.flatnonzero(tr&rows.strength_band.isin(bands))
        model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=leaf,random_state=seed,n_jobs=2)
        model.fit(x[idx],y[idx]);joblib.dump(model,out/f'{name}.joblib')
    gate=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=16,random_state=seed,n_jobs=2)
    gate.fit(x[tr],rows.loc[tr,'strength_band']);joblib.dump(gate,out/'gate.joblib')
    print('Finished training seed',seed,flush=True)

def train(root):
    rows,x,y=load_fit(root);tr=rows.role=='train'
    settings=[('ridge',[0,1,2],'r'),('svr',[0,1,2],'s'),('low',[0,1],'r'),('high',[1,2],'s'),('medium',[1],'r'),('strong',[2],'s')]
    for name,bands,kind in settings:
        idx=np.flatnonzero(tr&rows.strength_band.isin(bands))
        m=make_pipeline(StandardScaler(),Ridge(alpha=1000) if kind=='r' else MultiOutputRegressor(SVR(C=1,gamma=.57/169,epsilon=.1)))
        m.fit(x[idx],y[idx]);joblib.dump(m,root/f'{name}.joblib')
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1500));m.fit(x[tr],rows.loc[tr,'strength_band']);joblib.dump(m,root/'logistic.joblib')
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(fit_seed,[(root,seed) for seed in SEEDS]))

def freeze(root):
    assert not (root/'selection.json').exists()
    p=read(root/'protocol.json');parent=Path(p['parent']);cnn=read(parent/'n2880/frozen.json')
    old=read(parent/'selected_before_test.json')
    s=dict(primary='Overlap ensemble',history=old['rei']['history'],cnn=cnn,
        status='Frozen before any new test event generation',source_hash=_hash_file(Path(__file__)),protocol_hash=_hash_file(root/'protocol.json'),
        hashes={str(f.relative_to(root)):_hash_file(f) for f in root.rglob('*.joblib')},
        cnn_hashes={f'CNN_{seed}.pt':_hash_file(parent/'n2880'/f'CNN_{seed}.pt') for seed in SEEDS})
    _write_json(root/'selection.json',s)

_WORKER=None
def initialize(root):
    global _WORKER
    _WORKER=Path(root),read(Path(root)/'protocol.json')
def event_job(i):
    root,p=_WORKER;block,local=divmod(i,360);seed=TEST_SEEDS[block]
    c,labels=event_configuration(p['base'],p['sampling'],local,seed,123)
    sim=run_simulation(c,coords_mm=np.asarray(p['geometry']['union_mm']))
    xyz=fields_to_pauli(sim,p['circuit'],np.asarray(p['geometry']['distances']['5']['union_indices']))
    bits,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(seed+1,local,5)%(2**63-1),shots=2)
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(labels,event=i,event_uid=f'strength-{seed}:{local}',test_seed=seed,role='test')
    path=root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,d5=bits)
    path.with_suffix('.npz.tmp').replace(path);_write_json(path.with_suffix('.json'),row);return i

def generate(root):
    s=read(root/'selection.json');assert s['source_hash']==_hash_file(Path(__file__))
    (root/'events').mkdir(exist_ok=True)
    with _generation_lock(root):
        done=[i for i in range(EVENTS) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            futures=[pool.submit(event_job,i) for i in range(EVENTS) if i not in done]
            for f in as_completed(futures):
                done.append(f.result())
                if len(done)%108==0:print('Generated',len(done),'/',EVENTS,flush=True)
        _write_json(root/'manifest.json',dict(selection_hash=_hash_file(root/'selection.json'),hashes={f.name:_hash_file(f) for f in (root/'events').iterdir()}))

def evaluate(root):
    torch.set_num_threads(2);p=read(root/'protocol.json');s=read(root/'selection.json');manifest=read(root/'manifest.json')
    assert s['source_hash']==_hash_file(Path(__file__)) and s['protocol_hash']==_hash_file(root/'protocol.json')
    assert manifest['selection_hash']==_hash_file(root/'selection.json')
    for name,digest in s['hashes'].items():assert _hash_file(root/name)==digest
    raw,rows=[],[]
    for i in range(EVENTS):
        path=root/'events'/f'{i:04d}.npz'
        for ext in ('.npz','.json'):assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:bits=z['d5']
        assert bits.shape==(2,24,2047) and np.isin(bits,[0,1]).all();raw.append(bits)
        r=read(path.with_suffix('.json'));rows.extend([dict(r,shot=j) for j in range(2)])
    raw=np.concatenate(raw);rows=pd.DataFrame(rows);x=features(raw,p['preprocessing']);y=rows[['epicenter_row','epicenter_col']].to_numpy()
    deterministic={n:joblib.load(root/f'{n}.joblib').predict(x) for n in ['ridge','svr','low','high','medium','strong']}
    prob=joblib.load(root/'logistic.joblib').predict_proba(x)
    predictions={('Ridge',0):deterministic['ridge'],('SVR',0):deterministic['svr'],('Prior',0):np.broadcast_to(s['cnn']['center'],y.shape)}
    parent=Path(p['parent'])/'n2880'
    for seed in SEEDS:
        models={n:joblib.load(root/f'seed{seed}'/f'{n}.joblib') for n in ['global','weak','strong','gate']}
        et=models['global'].predict(x);base=.5*(et+deterministic['svr']);weak=models['weak'].predict(x);strong=.5*(models['strong'].predict(x)+deterministic['strong'])
        gp=models['gate'].predict_proba(x);hard=np.eye(3)[gp.argmax(1)];w=hard[:,0]+.5*hard[:,1]
        overlap=w[:,None]*deterministic['low']+(1-w[:,None])*deterministic['high']
        three=prob[:,0,None]*weak+prob[:,1,None]*deterministic['medium']+prob[:,2,None]*strong
        three_hard=hard[:,0,None]*weak+hard[:,1,None]*deterministic['medium']+hard[:,2,None]*strong
        for name,pred in [('ExtraTrees',et),('Blend',base),('Overlap',.5*(base+overlap)),('ThreeSoft',three),('ThreeHard',.5*(base+three_hard))]:predictions[name,seed]=pred
        assert _hash_file(parent/f'CNN_{seed}.pt')==s['cnn_hashes'][f'CNN_{seed}.pt']
        cnn=TemporalCNN(24);cnn.load_state_dict(torch.load(parent/f'CNN_{seed}.pt',weights_only=True))
        predictions['CNN',seed]=predict_nn(cnn,raw,np.asarray(s['cnn']['center']),s['cnn']['scale'])
    for name in ['ExtraTrees','Blend','Overlap','ThreeSoft','ThreeHard','CNN']:
        predictions[name,-1]=np.mean([predictions[name,seed] for seed in SEEDS],axis=0)
    geo=p['geometry']['distances']['5'];rei=rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=s['history'],circuit_repetitions=1)
    answered=np.isfinite(rei).all(1);predictions['REI',0]=np.where(answered[:,None],rei,s['cnn']['center'])
    frames=[]
    for (name,seed),pred in predictions.items():
        f=rows[['event_uid','event','shot','test_seed','strength_band','geometry','propagation_law','epicenter_region']].copy()
        f['model']=name;f['training_seed']=seed;f['pred_x'],f['pred_y']=pred.T;f['error_mm']=np.linalg.norm(pred-y,axis=1);frames.append(f)
    f=pd.concat(frames,ignore_index=True);f.to_csv(root/'predictions.csv',index=False)
    metrics=[];pairs=[]
    for group,mask in masks(f).items():
        sub=f[mask]
        for (name,seed),r in sub.groupby(['model','training_seed']):
            for block in [0,*TEST_SEEDS]:
                a=r if block==0 else r[r.test_seed==block]
                metrics.append(dict(group=group,model=name,training_seed=int(seed),test_seed=block,events=a.event_uid.nunique(),**summary(a.error_mm)))
        for name in ['Ridge','SVR','ExtraTrees','Blend','Overlap','ThreeSoft','ThreeHard','CNN']:
            seed=0 if name in ['Ridge','SVR'] else -1
            for reference in ['REI','Blend','Prior']:
                refseed=-1 if reference=='Blend' else 0
                a=sub[(sub.model==name)&(sub.training_seed==seed)].groupby('event_uid').error_mm.mean()
                b=sub[(sub.model==reference)&(sub.training_seed==refseed)].groupby('event_uid').error_mm.mean()
                assert a.index.equals(b.index);delta=(a-b).to_numpy();rng=np.random.default_rng(2026090947)
                boot=rng.choice(delta,(10000,len(delta))).mean(1)
                pairs.append(dict(group=group,model=name,reference=reference,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ci975=np.quantile(boot,[.0125,.9875]).tolist()))
    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False);_write_json(root/'paired.json',pairs)
    _write_json(root/'evaluation.json',dict(rei_answer_rate=float(answered.mean()),selection_hash=_hash_file(root/'selection.json')))
    print(pd.DataFrame(metrics).query("test_seed==0 and training_seed<=0 and group in ['weak','medium','strong']").to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','freeze','generate','evaluate']);parser.add_argument('root',type=Path);a=parser.parse_args();globals()[a.stage](a.root)
