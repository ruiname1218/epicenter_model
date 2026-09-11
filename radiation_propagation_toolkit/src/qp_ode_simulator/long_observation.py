"""Paired 2/4/8 ms prefixes of the same physical and syndrome realization."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .api import run_simulation
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .distance_study import fields_to_pauli
from .fault_response import sample
from .localization import _hash_file
from .specialist_study import read
from .strength_benchmark import masks,summary
from .temporal_diagnosis import rei_center

SEED=2026091101
HORIZONS={2:2047,4:4095,8:8191}


def role(i):
    if not 0<=i<2052:raise ValueError('event index out of range')
    return 'train' if i<1296 else 'validation' if i<1512 else 'test'


def eligible(label):
    return not (label['geometry']=='elliptical' and label['strength_band']==2)


def edges_for(horizon,cut=339):
    edges=[0,cut,cut+(2047-cut)//3,cut+2*(2047-cut)//3,2047]
    if horizon>=4:edges.extend([3071,4095])
    if horizon>=8:edges.extend([6143,8191])
    if horizon not in HORIZONS:raise ValueError('unknown horizon')
    return np.asarray(edges)


def features(raw,quiet,cut=339):
    out={};quiet=np.asarray(quiet)
    for h in HORIZONS:
        e=edges_for(h,cut);rates=np.stack([raw[...,a:b].mean(-1) for a,b in zip(e[:-1],e[1:])],axis=-1)
        excess=rates-quiet[None,:,None];positive=np.maximum(excess[:,:,1:],0)
        relative=positive/(positive.mean(1,keepdims=True)+.003)
        out[f'{h}_full']=np.column_stack([excess.reshape(len(raw),-1),relative.reshape(len(raw),-1),np.full(len(raw),cut/2047)]).astype(np.float32)
        out[f'{h}_relative']=relative.reshape(len(raw),-1).astype(np.float32)
    return out


def prepare(root):
    root.mkdir(exist_ok=True);(root/'events').mkdir(exist_ok=True)
    assert not (root/'ids.json').exists() and not any((root/'events').iterdir()), 'already prepared'
    inherited=read(root.parent/'strength_benchmark_20260909/protocol.json')
    base=copy.deepcopy(inherited['base']);base['time']['end_ms']=8.196
    circuit=copy.deepcopy(inherited['circuit']);circuit['rounds']=8192
    p=dict(seed=SEED,base=base,sampling=inherited['sampling'],circuit=circuit,geometry=inherited['geometry'],response=inherited['response'],preprocessing=inherited['preprocessing'],
        fitting='Generate eligible events only from 36 train cycles and 6 validation cycles; exclude strong ellipses:1080 train,180 validation. Fresh test15 cycles includes both shapes:540 physical events.2 shots/event.',
        comparison='Same 8.192ms realization, prefix2.048/4.096/8.192ms; preserve all first2ms time boundaries and append wider late bins. Physical noise/device unchanged.',
        models='Each duration x full/relative x Ridge1000,SVR C1 gamma .57/d,SVR C10 gamma .057/d,ET256 leaf3 seed41.24 fits.',
        selection='Freeze equal medium/strong validation mean minimum separately per duration; all candidates and tails reported. Fixed relative/SVR1 is primary information-duration comparison.',
        primary='Relative/SVR1 at4minus2 and8minus2 in medium and strong:4 co-primary means,98.75% event-bootstrap intervals. Others descriptive95%.',
        limitation='Pilot uses1080 training events not previous2880. Same-device simulator; not streaming detection/decoding. No true timing/strength input. Fixed quiet calibration inherited independent of fresh events.',
        source_hash=_hash_file(Path(__file__)),dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['api.py','dataset.py','distance_study.py','fault_response.py','simulator.py','stim_qec.py']})
    _write_json(root/'protocol.json',p)
    ids={k:[] for k in ['train','validation','test']}
    for i in range(2052):
        config,label=event_configuration(base,p['sampling'],i,SEED,123)
        label=dict(label,geometry=config['shape']['geometry_choices'][0])
        if role(i)=='test' or eligible(label):ids[role(i)].append(i)
    assert [len(ids[k]) for k in ids]==[1080,180,540]
    _write_json(root/'ids.json',ids)


_WORKER=None
def initialize(root):
    global _WORKER
    root=Path(root);_WORKER=root,read(root/'protocol.json')


def event_job(i):
    root,p=_WORKER;c,label=event_configuration(p['base'],p['sampling'],i,SEED,123)
    sim=run_simulation(c,coords_mm=np.asarray(p['geometry']['union_mm']))
    xyz=fields_to_pauli(sim,p['circuit'],np.asarray(p['geometry']['distances']['5']['union_indices']))
    raw,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(SEED+1,i,5)%(2**63-1),shots=2)
    assert raw.shape==(2,24,8191)
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(label,event=i,event_uid=f'long-{SEED}:{i}',role=role(i),test_block=(i-1512)//180 if i>=1512 else -1)
    path=root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,d5=raw)
    path.with_suffix('.npz.tmp').replace(path);_write_json(path.with_suffix('.json'),row)
    return i


def verify(root):
    p=read(root/'protocol.json');assert p['source_hash']==_hash_file(Path(__file__))
    for name,digest in p['dependencies'].items():assert _hash_file(Path(__file__).with_name(name))==digest
    if (root/'selection.json').exists():
        s=read(root/'selection.json');assert s['protocol_hash']==_hash_file(root/'protocol.json')
        for name,digest in s['hashes'].items():assert _hash_file(root/name)==digest
    return p


def generate(root,test=False):
    verify(root);ids=read(root/'ids.json');ids=ids['test'] if test else ids['train']+ids['validation']
    if test:assert (root/'selection.json').exists()
    with _generation_lock(root):
        done=[i for i in ids if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            fs=[pool.submit(event_job,i) for i in ids if i not in done]
            for f in as_completed(fs):
                done.append(f.result())
                if len(done)%90==0:print('Generated',len(done),'/',len(ids),'test',test,flush=True)
        _write_json(root/('test_manifest.json' if test else 'fit_manifest.json'),dict(selection_hash=_hash_file(root/'selection.json') if test else None,
            hashes={f'{i:04d}{ext}':_hash_file(root/'events'/f'{i:04d}{ext}') for i in ids for ext in ['.npz','.json']}))


def load(root,test=False):
    p=verify(root);ids=read(root/'ids.json');ids=ids['test'] if test else ids['train']+ids['validation']
    manifest=read(root/('test_manifest.json' if test else 'fit_manifest.json'))
    if test:assert manifest['selection_hash']==_hash_file(root/'selection.json')
    parts={};rows=[]
    for i in ids:
        path=root/'events'/f'{i:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:raw=z['d5']
        for name,x in features(raw,p['preprocessing']['quiet'],p['preprocessing']['cut']).items():parts.setdefault(name,[]).append(x)
        row=read(path.with_suffix('.json'));rows.extend([dict(row,shot=j) for j in range(2)])
    return {k:np.concatenate(v) for k,v in parts.items()},pd.DataFrame(rows)


def summarize(pred,rows):
    y=rows[['epicenter_row','epicenter_col']].to_numpy();records=[]
    for name,v in pred.items():
        error=np.linalg.norm(v-y,axis=1)
        for group,mask in masks(rows).items():
            if np.any(mask):records.append(dict(model=name,group=group,events=rows.loc[mask,'event_uid'].nunique(),**summary(error[mask])))
    return pd.DataFrame(records)


def train(root):
    xs,rows=load(root);rows.to_csv(root/'fit_rows.csv',index=False);y=rows[['epicenter_row','epicenter_col']].to_numpy();tr=rows.role=='train';v=rows.role=='validation';pred={}
    for case,x in xs.items():
        for kind in ['Ridge','SVR1','SVR10','ET']:
            if kind=='Ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=1000))
            elif kind=='ET':model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=3,random_state=41,n_jobs=2)
            else:model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=1 if kind=='SVR1' else 10,gamma=(.57 if kind=='SVR1' else .057)/x.shape[1],epsilon=.1)))
            model.fit(x[tr],y[tr]);name=f'{case}_{kind}';joblib.dump(model,root/f'{name}.joblib');pred[name]=model.predict(x[v]);print('Fit',name,flush=True)
    m=summarize(pred,rows[v]);m.to_csv(root/'validation_metrics.csv',index=False)
    chosen={}
    for h in HORIZONS:
        scores=m[m.model.str.startswith(str(h)+'_')&m.group.isin(['medium','strong'])].groupby('model').mean_mm.mean();chosen[str(h)]=scores.idxmin()
    _write_json(root/'selection.json',dict(selected=chosen,protocol_hash=_hash_file(root/'protocol.json'),hashes={f.name:_hash_file(f) for f in root.glob('*.joblib')},status='Frozen before fresh test generation'))
    print('Selected',chosen,flush=True)


def evaluate(root):
    p=verify(root);s=read(root/'selection.json');xs,rows=load(root,True);rows.to_csv(root/'test_rows.csv',index=False)
    y=rows[['epicenter_row','epicenter_col']].to_numpy();pred={};timings={}
    for name in s['hashes']:
        key=Path(name).stem;case='_'.join(key.split('_')[:2]);model=joblib.load(root/name)
        start=time.perf_counter();pred[key]=model.predict(xs[case]);timings[key]=(time.perf_counter()-start)/len(rows)*1000
    geo=p['geometry']['distances']['5'];rei={h:[] for h in HORIZONS}
    for i in read(root/'ids.json')['test']:
        with np.load(root/'events'/f'{i:04d}.npz') as z:raw=z['d5']
        for h,ticks in HORIZONS.items():
            value=rei_center(raw[...,:ticks],geo['sites_mm'],geo['physical_mm'],history_length=1024,circuit_repetitions=1)
            rei[h].append(value)
    trainrows=pd.read_csv(root/'fit_rows.csv');center=trainrows[trainrows.role=='train'][['epicenter_row','epicenter_col']].mean().to_numpy()
    answers={}
    for h,values in rei.items():
        a=np.concatenate(values);valid=np.isfinite(a).all(1);answers[str(h)]=float(valid.mean());pred[f'{h}_REI']=np.where(valid[:,None],a,center)
    m=summarize(pred,rows);m.to_csv(root/'metrics.csv',index=False);frames=[]
    for name,v in pred.items():
        a=rows[['event_uid','shot','strength_band','geometry','test_block']].copy();a['model']=name;a['pred_x'],a['pred_y']=v.T;a['error_mm']=np.linalg.norm(v-y,axis=1);frames.append(a)
    df=pd.concat(frames,ignore_index=True);df.to_csv(root/'predictions.csv',index=False)
    pairs=[]
    contrasts=[]
    for h in [4,8]:
        contrasts.extend([(f'{h}_relative_SVR1','2_relative_SVR1'),(f'{h}_full_SVR1','2_full_SVR1'),(s['selected'][str(h)],s['selected']['2'])])
    for group,mask in masks(rows).items():
        if group not in ['weak','medium','strong']:continue
        for name,ref in sorted(set(contrasts)):
            delta=np.linalg.norm(pred[name][mask]-y[mask],axis=1)-np.linalg.norm(pred[ref][mask]-y[mask],axis=1)
            delta=pd.DataFrame({'uid':rows.loc[mask,'event_uid'],'delta':delta}).groupby('uid').delta.mean().to_numpy()
            boot=np.random.default_rng(2026091103).choice(delta,(10000,len(delta))).mean(1)
            pairs.append(dict(model=name,reference=ref,group=group,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ci9875=np.quantile(boot,[.00625,.99375]).tolist()))
    _write_json(root/'paired.json',pairs)
    blocks=[]
    for block in range(3):
        mask=rows.test_block==block;a=summarize({n:v[mask] for n,v in pred.items()},rows[mask]);a['test_block']=block;blocks.append(a)
    pd.concat(blocks).to_csv(root/'block_metrics.csv',index=False)
    old=set(trainrows.generation_seed)
    for folder in root.parent.iterdir():
        if folder==root:continue
        for path in (folder/'events').glob('*.json'):
            seed=read(path).get('generation_seed')
            if seed is not None:old.add(seed)
    assert not set(rows.generation_seed)&old and rows.event_uid.nunique()==540
    _write_json(root/'audit.json',dict(status='passed',train_events=1080,validation_events=180,test_events=540,seed_overlap=0,paired_prefixes=True,hashes='passed',rei_answer_rates=answers,
        timing_ms_per_shot_batch=timings,timing_scope='Batch throughput of model.predict only, excludes feature extraction/loading. Not single-shot latency. Observation costs2.048/4.096/8.192ms before prediction.'))
    print(m[m.model.isin([f'{h}_relative_SVR1' for h in HORIZONS]+list(s['selected'].values()))&m.group.isin(['weak','medium','strong'])].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate_fit','train','generate_test','evaluate']);parser.add_argument('root',type=Path);a=parser.parse_args()
    if a.stage.startswith('generate'):generate(a.root,test=a.stage=='generate_test')
    else:globals()[a.stage](a.root)
