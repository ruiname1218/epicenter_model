"""Medium-only paired duration, sample-size and covariance-aware template pilot."""
import argparse
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib
from scipy.linalg import solve_triangular
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from .radical_data import read,bin_rates
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .api import run_simulation
from .distance_study import fields_to_pauli
from .fault_response import sample
from .localization import _hash_file
from .strength_benchmark import summary

TRAIN_SEED=20260911211
TEST_SEEDS=(20260911221,20260911231,20260911241)
SIZES=(288,576,864)
HORIZONS=(4,8,16)
TEMPERATURES=(.1,1.,10.,100.)


def edges(h):
    if h not in HORIZONS:raise ValueError('duration')
    # Retain every earlier boundary; the first4ms has16 intervals.
    return np.r_[np.linspace(0,4095,17,dtype=int),np.arange(4351,h*1024,256)]


def rates(raw,h):
    e=edges(h);return np.stack([raw[...,a:b].mean(-1) for a,b in zip(e[:-1],e[1:])],-1).astype(np.float32)


def relative(raw,h,quiet):
    e=[0,339,908,1477,2047,3071,4095]
    if h>=8:e.extend([6143,8191])
    if h>=16:e.extend([12287,16383])
    r=np.stack([raw[...,a:b].mean(-1) for a,b in zip(e[1:-1],e[2:])],-1)
    p=np.maximum(r-np.asarray(quiet)[None,:,None],0)
    return (p/(p.mean(1,keepdims=True)+.003)).reshape(len(raw),-1).astype(np.float32)


def prepare(root):
    root.mkdir(exist_ok=False);(root/'events').mkdir()
    old=root.parent/'radical_pilot_20260911';p=read(old/'protocol.json')
    base=copy.deepcopy(p['base']);base['time']['end_ms']=16.388
    circuit=copy.deepcopy(p['circuits']['base']);circuit['rounds']=16384
    protocol=dict(base=base,circuit=circuit,sampling=p['sampling'],coords=p['union'],indices=p['indices']['base'],response=p['response']['base'],
        quiet=read(old/'quiet.json')['base'],parent=str(old.resolve()),
        design='Medium only: nested train288/576/864,validation96,test180(3 roots x60),2 shots/event. Paired prefixes4/8/16ms. No true strength input at inference; evaluation conditional on medium event.',
        selection='SVR and binomial/GLS template temperature chosen on validation mean separately per duration/size. Primary: validation-selected16ms864 vs fixed binomial4ms288. Other comparisons descriptive95%.',
        source_hash=_hash_file(Path(__file__)),dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['api.py','dataset.py','distance_study.py','fault_response.py','simulator.py','stim_qec.py','radical_data.py']})
    specs=[]
    for seed,total in [(TRAIN_SEED,2880),*[(s,180) for s in TEST_SEEDS]]:
        for local in range(total):
            _,label=event_configuration(base,p['sampling'],local,seed,123)
            if label['strength_band']!=1:continue
            role=('train' if local<2592 else 'validation') if seed==TRAIN_SEED else 'test'
            specs.append(dict(event=len(specs),local=local,seed=seed,role=role))
    assert [sum(s['role']==r for s in specs) for r in ['train','validation','test']]==[864,96,180]
    _write_json(root/'protocol.json',protocol);_write_json(root/'specs.json',specs)


def check(root):
    p=read(root/'protocol.json');assert p['source_hash']==_hash_file(Path(__file__))
    for name,h in p['dependencies'].items():assert _hash_file(Path(__file__).with_name(name))==h
    return p


_WORKER=None
def initialize(root):
    global _WORKER
    root=Path(root);_WORKER=root,read(root/'protocol.json')


def event_job(spec):
    root,p=_WORKER;c,label=event_configuration(p['base'],p['sampling'],spec['local'],spec['seed'],123)
    sim=run_simulation(c,coords_mm=np.asarray(p['coords']))
    xyz=fields_to_pauli(sim,p['circuit'],np.asarray(p['indices']))
    raw,latent=sample(p['response'],p['circuit'],*xyz,seed=_seed(spec['seed']+1,spec['local'],5)%(2**63-1),shots=2)
    assert raw.shape==(2,24,16383)
    payload={'raw':raw}
    for h in HORIZONS:payload[f'latent{h}']=rates(latent,h)
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(label,**spec,event_uid=f'medium-{spec["seed"]}:{spec["local"]}')
    path=root/'events'/f'{spec["event"]:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**payload)
    path.with_suffix('.npz.tmp').replace(path);_write_json(path.with_suffix('.json'),row)
    return spec['event']


def generate(root,test=False):
    check(root);specs=[s for s in read(root/'specs.json') if (s['role']=='test')==test]
    selection_hash=_hash_file(root/'selection.json') if test else None
    with _generation_lock(root):
        done=[s['event'] for s in specs if all((root/'events'/f'{s["event"]:04d}{ext}').exists() for ext in ['.npz','.json'])]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            fs=[pool.submit(event_job,s) for s in specs if s['event'] not in done]
            for f in as_completed(fs):
                done.append(f.result())
                if len(done)%60==0:print('Generated',len(done),'/',len(specs),'test',test,flush=True)
        if test:assert selection_hash==_hash_file(root/'selection.json')
        _write_json(root/('test_manifest.json' if test else 'fit_manifest.json'),dict(selection_hash=selection_hash,
            hashes={f'{s["event"]:04d}{ext}':_hash_file(root/'events'/f'{s["event"]:04d}{ext}') for s in specs for ext in ['.npz','.json']}))


def load(root,test=False):
    p=check(root);m=read(root/('test_manifest.json' if test else 'fit_manifest.json'));rows=[];data={}
    if test:assert m['selection_hash']==_hash_file(root/'selection.json')
    for spec in read(root/'specs.json'):
        if (spec['role']=='test')!=test:continue
        path=root/'events'/f'{spec["event"]:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==m['hashes'][f'{spec["event"]:04d}{ext}']
        with np.load(path) as z:
            for h in HORIZONS:
                data.setdefault(f'counts{h}',[]).append(rates(z['raw'],h))
                data.setdefault(f'latent{h}',[]).append(np.repeat(z[f'latent{h}'][None],2,axis=0))
                data.setdefault(f'relative{h}',[]).append(relative(z['raw'],h,p['quiet']))
        row=read(path.with_suffix('.json'));rows.extend([dict(row,shot=j) for j in range(2)])
    return {k:np.concatenate(v) for k,v in data.items()},pd.DataFrame(rows)


def score_matrix(observed,bundle):
    x=observed.reshape(len(observed),-1)
    if bundle['kind']=='binomial':
        return (x*bundle['widths'])@bundle['logodds'].T+bundle['offset']
    # All nuisance noise parameters below are estimated on training events.
    w=solve_triangular(bundle['chol'],(x-bundle['noise_mean']).T,lower=True).T
    return w@bundle['whitened'].T-.5*np.sum(bundle['whitened']**2,axis=1)


def template_predict(observed,bundle,temperature=None):
    t=bundle['temperature'] if temperature is None else temperature
    return softmax(score_matrix(observed,bundle)/t,axis=1)@bundle['centers']


def fit_template(counts,latent,centers,h,kind):
    # Two shots/event, only one copy of each true training profile in the bank.
    profiles=latent[::2].reshape(len(latent)//2,-1).astype(np.float64)
    b=dict(kind=kind,centers=centers[::2],temperature=1.)
    if kind=='binomial':
        p=np.clip(profiles,1e-5,1-1e-5);widths=np.tile(np.diff(edges(h)),24)
        b.update(widths=widths,logodds=np.log(p)-np.log1p(-p),offset=np.log1p(-p)@widths)
    else:
        residual=(counts-latent).reshape(len(counts),-1).astype(np.float64)
        covariance=LedoitWolf().fit(residual)
        chol=np.linalg.cholesky(covariance.covariance_+np.eye(profiles.shape[1])*1e-8)
        b.update(chol=chol,noise_mean=covariance.location_,whitened=solve_triangular(chol,profiles.T,lower=True).T,shrinkage=float(covariance.shrinkage_))
    return b


def train(root):
    if (root/'selection.json').exists():raise ValueError('frozen run')
    data,rows=load(root);rows.to_csv(root/'fit_rows.csv',index=False)
    va=(rows.role=='validation').to_numpy();y=rows[['epicenter_row','epicenter_col']].to_numpy();pred={};hashes={};selected={}
    for n in SIZES:
        tr=(rows.role=='train').to_numpy()&(rows.event.to_numpy()<n)
        assert tr.sum()==n*2
        for h in HORIZONS:
            x=data[f'relative{h}'];name=f'n{n}_h{h}_SVR'
            model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=1,epsilon=.1,gamma=.57/x.shape[1])))
            model.fit(x[tr],y[tr]);pred[name]=model.predict(x[va]);joblib.dump(model,root/f'{name}.joblib');hashes[f'{name}.joblib']=_hash_file(root/f'{name}.joblib')
            for kind in ['binomial','GLS']:
                name=f'n{n}_h{h}_{kind}';b=fit_template(data[f'counts{h}'][tr],data[f'latent{h}'][tr],y[tr],h,kind)
                scores=score_matrix(data[f'counts{h}'][va],b)
                candidates={t:softmax(scores/t,axis=1)@b['centers'] for t in TEMPERATURES}
                t=min(candidates,key=lambda t:np.linalg.norm(candidates[t]-y[va],axis=1).mean());b['temperature']=t;pred[name]=candidates[t]
                joblib.dump(b,root/f'{name}.joblib');hashes[f'{name}.joblib']=_hash_file(root/f'{name}.joblib')
            keys=[f'n{n}_h{h}_{k}' for k in ['SVR','binomial','GLS']]
            selected[f'n{n}_h{h}']=min(keys,key=lambda k:np.linalg.norm(pred[k]-y[va],axis=1).mean())
            print('Cell',n,h,{k:float(np.linalg.norm(pred[k]-y[va],axis=1).mean()) for k in keys},flush=True)
    # Explicit old-rule control: 4ms288 medium templates at temperature1.
    b=joblib.load(root/'n288_h4_binomial.joblib');b['temperature']=1.;joblib.dump(b,root/'reference.joblib');hashes['reference.joblib']=_hash_file(root/'reference.joblib')
    pred['reference']=template_predict(data['counts4'][va],b)
    records=[dict(model=k,**summary(np.linalg.norm(v-y[va],axis=1))) for k,v in pred.items()]
    pd.DataFrame(records).to_csv(root/'validation_metrics.csv',index=False)
    _write_json(root/'selection.json',dict(selected=selected,primary=selected['n864_h16'],reference='reference',hashes=hashes,
        source_hash=_hash_file(Path(__file__)),protocol_hash=_hash_file(root/'protocol.json'),manifest_hash=_hash_file(root/'fit_manifest.json')))
    print('FROZEN',selected,flush=True)


def evaluate(root):
    s=read(root/'selection.json');assert s['source_hash']==_hash_file(Path(__file__)) and s['protocol_hash']==_hash_file(root/'protocol.json')
    assert s['manifest_hash']==_hash_file(root/'fit_manifest.json')
    for name,h in s['hashes'].items():assert _hash_file(root/name)==h
    data,rows=load(root,True);rows.to_csv(root/'test_rows.csv',index=False);y=rows[['epicenter_row','epicenter_col']].to_numpy();pred={}
    for file in s['hashes']:
        name=Path(file).stem;b=joblib.load(root/file);h=4 if name=='reference' else int(name.split('_')[1][1:])
        pred[name]=b.predict(data[f'relative{h}']) if name.endswith('SVR') else template_predict(data[f'counts{h}'],b)
    p=read(root/'protocol.json');old=Path(p['parent']);parent_selection=read(old/'selection.json')
    for name in ['base_SVR','template_inverse']:
        assert _hash_file(old/f'{name}.joblib')==parent_selection['hashes'][f'{name}.joblib']
        b=joblib.load(old/f'{name}.joblib')
        if name=='base_SVR':pred['old_allband_SVR4']=b.predict(data['relative4'])
        else:
            from .radical_models import inverse_predict
            pred['old_allband_template4']=inverse_predict(data['counts4'],**b)
    records=[];frames=[];blocks=[]
    for name,v in pred.items():
        error=np.linalg.norm(v-y,axis=1);records.append(dict(model=name,**summary(error)))
        f=rows[['event_uid','shot','seed','geometry']].copy();f['model']=name;f['error_mm']=error;f['pred_x'],f['pred_y']=v.T;frames.append(f)
        for seed in TEST_SEEDS:blocks.append(dict(model=name,seed=seed,**summary(error[rows.seed==seed])))
    pd.DataFrame(records).to_csv(root/'metrics.csv',index=False);pd.concat(frames).to_csv(root/'predictions.csv',index=False);pd.DataFrame(blocks).to_csv(root/'seed_metrics.csv',index=False)
    contrasts=[(s['primary'],s['reference'])]
    for h in HORIZONS:
        for n in SIZES:
            contrasts.append((f'n{n}_h{h}_GLS',f'n{n}_h{h}_binomial'))
            if n>288:
                for k in ['SVR','binomial','GLS']:contrasts.append((f'n{n}_h{h}_{k}',f'n288_h{h}_{k}'))
            if h>4:
                for k in ['SVR','binomial','GLS']:contrasts.append((f'n{n}_h{h}_{k}',f'n{n}_h4_{k}'))
    contrasts.extend([(s['primary'],'old_allband_SVR4'),(s['primary'],'old_allband_template4')]);paired=[]
    for name,ref in sorted(set(contrasts)):
        delta=np.linalg.norm(pred[name]-y,axis=1)-np.linalg.norm(pred[ref]-y,axis=1)
        event_delta=delta.reshape(-1,2).mean(1);boot=np.random.default_rng(20260911317).choice(event_delta,(10000,len(event_delta))).mean(1)
        paired.append(dict(model=name,reference=ref,difference_mm=float(event_delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),primary=name==s['primary'] and ref==s['reference']))
    _write_json(root/'paired.json',paired)
    fitrows=pd.read_csv(root/'fit_rows.csv');assert set(rows.event_uid).isdisjoint(fitrows.event_uid)
    oldseeds=set(fitrows.generation_seed)
    for folder in root.parent.iterdir():
        if folder==root:continue
        for f in (folder/'events').glob('*.json'):
            g=read(f).get('generation_seed')
            if g is not None:oldseeds.add(g)
    assert not oldseeds&set(rows.generation_seed)
    _write_json(root/'audit.json',dict(train_events=864,validation_events=96,test_events=int(rows.event_uid.nunique()),test_shots=len(rows),
        hash_checks=True,unseen_test_generation_seeds=True,medium_only=True,primary=s['primary'],
        note='Paired shot prefixes, frozen validation selection. Ancillary contrasts exploratory, training sample common not independent repeats. Physical latent targets training only.'))
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate_fit','train','generate_test','evaluate']);p.add_argument('root',type=Path);a=p.parse_args()
    if a.stage.startswith('generate'):generate(a.root,a.stage=='generate_test')
    else:globals()[a.stage](a.root)
