"""4ms feature recipes crossed with identical SVR gamma values.

Automatic selection/PCA see training events only. Fresh test follows freezing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
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
from .long_observation import edges_for,summarize
from .specialist_study import read
from .strength_benchmark import masks
from .temporal_diagnosis import rei_center

TEST_SEEDS=(2026091121,2026091221,2026091321)
GAMMAS={'g265':.57/265,'g120':.57/120,'g72':.57/72}
BASELINE='relative__g120'


def raw_features(raw,quiet,cut):
    if raw.shape[1:]!=(24,4095):raise ValueError('expected24 checks x4095 rounds')
    e=edges_for(4,cut);rates=np.stack([raw[...,a:b].mean(-1) for a,b in zip(e[:-1],e[1:])],-1)
    excess=rates-np.asarray(quiet)[None,:,None];post=excess[:,:,1:];positive=np.maximum(post,0)
    level=positive.mean(1,keepdims=True);relative=positive/(level+.003)
    signed=post/(np.abs(post).mean(1,keepdims=True)+.003)
    root=np.sqrt(positive);root=root/(root.mean(1,keepdims=True)+np.sqrt(.003))
    flat=lambda z:z.reshape(len(raw),-1).astype(np.float32)
    return dict(full=np.column_stack([flat(excess),flat(relative),np.full(len(raw),cut/2047)]).astype(np.float32),
        relative=flat(relative),relative_level=np.column_stack([flat(relative),np.log1p(level[:,0,:]/.003)]).astype(np.float32),
        early=flat(relative[:,:,:3]),late=flat(relative[:,:,3:]),delta=flat(np.diff(relative,axis=2)),
        signed=flat(signed),sqrt=flat(root),log=flat(np.log1p(relative)))


def fit_reduction(x,y):
    scores=np.mean([np.nan_to_num(f_regression(x,y[:,axis])[0],nan=0,posinf=0) for axis in range(2)],axis=0)
    top=np.argsort(-scores,kind='stable')[:60]
    scaler=StandardScaler().fit(x);pca=PCA(n_components=32,svd_solver='full').fit(scaler.transform(x))
    return dict(top=top,scaler=scaler,pca=pca,scores=scores)


def reduce_features(xs,reduction):
    xs=dict(xs);x=xs['relative'];xs['top60']=x[:,reduction['top']]
    xs['pca32']=reduction['pca'].transform(reduction['scaler'].transform(x)).astype(np.float32)
    return xs


def prepare(root):
    root.mkdir(exist_ok=False);parent=root.parent/'long_observation_20260910';p=copy.deepcopy(read(parent/'protocol.json'))
    p['base']['time']['end_ms']=4.100;p['circuit']['rounds']=4096
    p.update(parent=str(parent.resolve()),test_seeds=list(TEST_SEEDS),
        features='11 recipes:full,relative,relative_level,early,late,delta,signed,sqrt,log,top60,pca32.',
        fitting='Reuse1080 train/180 validation physical events from previous8ms study, take4.096ms prefix only. Supervised top60 uses average training-only coordinate F scores; PCA32/scaler fit only on training.',
        search='Every feature recipe crossed with SAME 3 gamma values:.57/265,.57/120,.57/72. SVR C1 epsilon.1. Plus Ridge1000 each recipe.44 fits; no dimension-dependent gamma hidden in recipe.',
        baseline=BASELINE,selection='Equal medium/strong validation mean minimum; each p90 <= baseline+.05mm and weak mean <= baseline+.02mm; baseline eligible. Freeze unconstrained minimum separately.',
        primary='Selected minus relative__g120 medium/strong means:97.5% paired-event bootstrap intervals,2 co-primary comparisons. All feature/gamma and secondary contrasts descriptive95%.',
        test='Fresh540 physical events,180 per strength,3 independent roots x180 balanced events,2 shots each. Strong ellipses excluded from fitting/selection.',
        source_hash=_hash_file(Path(__file__)),parent_manifest_hash=_hash_file(parent/'fit_manifest.json'),
        dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['api.py','dataset.py','distance_study.py','fault_response.py','simulator.py','stim_qec.py','long_observation.py']})
    _write_json(root/'protocol.json',p)


def fit_candidate(args):
    root,name,kind=args;rows=pd.read_csv(root/'fit_rows.csv');x=np.load(root/f'x_{name}.npy');tr=rows.role=='train';v=rows.role=='validation';y=rows[['epicenter_row','epicenter_col']].to_numpy()
    reg=Ridge(alpha=1000) if kind=='Ridge' else MultiOutputRegressor(SVR(C=1,gamma=GAMMAS[kind],epsilon=.1))
    model=make_pipeline(StandardScaler(),reg);model.fit(x[tr],y[tr]);key=f'{name}__{kind}'
    joblib.dump(model,root/f'{key}.joblib');np.save(root/f'validation_{key}.npy',model.predict(x[v]));print('Fit',key,flush=True)


def train(root):
    p=read(root/'protocol.json');parent=Path(p['parent']);manifest=read(parent/'fit_manifest.json');assert _hash_file(parent/'fit_manifest.json')==p['parent_manifest_hash']
    ids=read(parent/'ids.json');rows=[];parts={}
    for i in ids['train']+ids['validation']:
        path=parent/'events'/f'{i:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:raw=z['d5'][...,:4095]
        for name,x in raw_features(raw,p['preprocessing']['quiet'],p['preprocessing']['cut']).items():parts.setdefault(name,[]).append(x)
        row=read(path.with_suffix('.json'));rows.extend([dict(row,shot=j) for j in range(2)])
    rows=pd.DataFrame(rows);xs={n:np.concatenate(v) for n,v in parts.items()};tr=rows.role=='train';v=rows.role=='validation';y=rows[['epicenter_row','epicenter_col']].to_numpy()
    assert rows.groupby('role').event_uid.nunique().to_dict()=={'train':1080,'validation':180}
    assert not set(rows[tr].event_uid)&set(rows[v].event_uid)
    reduction=fit_reduction(xs['relative'][tr],y[tr]);joblib.dump(reduction,root/'reduction.joblib');xs=reduce_features(xs,reduction)
    for name,x in xs.items():np.save(root/f'x_{name}.npy',x)
    rows.to_csv(root/'fit_rows.csv',index=False)
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(fit_candidate,[(root,n,k) for n in xs for k in [*GAMMAS,'Ridge']]))
    pred={path.stem.removeprefix('validation_'):np.load(path) for path in root.glob('validation_*.npy')}
    metrics=summarize(pred,rows[v]);metrics.to_csv(root/'validation_metrics.csv',index=False)
    ref=metrics[metrics.model==BASELINE].set_index('group');objectives={};eligible=[]
    for name,g in metrics.groupby('model'):
        g=g.set_index('group');objectives[name]=float(g.loc[['medium','strong'],'mean_mm'].mean())
        if all(g.loc[b,'p90_mm']<=ref.loc[b,'p90_mm']+.05 for b in ['medium','strong']) and g.loc['weak','mean_mm']<=ref.loc['weak','mean_mm']+.02:eligible.append(name)
    selection=dict(selected=min(eligible,key=objectives.get),unconstrained=min(objectives,key=objectives.get),eligible=eligible,objective=objectives,
        dimensions={n:x.shape[1] for n,x in xs.items()},protocol_hash=_hash_file(root/'protocol.json'),source_hash=_hash_file(Path(__file__)),
        hashes={f.name:_hash_file(f) for f in root.glob('*.joblib')},status='Frozen before fresh test generation')
    _write_json(root/'selection.json',selection);print('SELECTED',selection['selected'],'unconstrained',selection['unconstrained'],flush=True)


def verify(root):
    p=read(root/'protocol.json');s=read(root/'selection.json')
    assert s['source_hash']==_hash_file(Path(__file__)) and s['protocol_hash']==_hash_file(root/'protocol.json')
    for name,digest in p['dependencies'].items():assert _hash_file(Path(__file__).with_name(name))==digest
    for name,digest in s['hashes'].items():assert _hash_file(root/name)==digest
    return p,s


_WORKER=None
def initialize(root):
    global _WORKER
    root=Path(root);_WORKER=root,read(root/'protocol.json')


def event_job(i):
    root,p=_WORKER;block,local=divmod(i,180);seed=TEST_SEEDS[block]
    c,label=event_configuration(p['base'],p['sampling'],local,seed,123)
    sim=run_simulation(c,coords_mm=np.asarray(p['geometry']['union_mm']))
    xyz=fields_to_pauli(sim,p['circuit'],np.asarray(p['geometry']['distances']['5']['union_indices']))
    raw,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(seed+1,local,5)%(2**63-1),shots=2)
    assert raw.shape==(2,24,4095)
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(label,event=i,event_uid=f'feature-{seed}:{local}',role='test',test_seed=seed)
    path=root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,d5=raw)
    path.with_suffix('.npz.tmp').replace(path);_write_json(path.with_suffix('.json'),row);return i


def generate(root):
    verify(root);(root/'events').mkdir(exist_ok=True)
    with _generation_lock(root):
        done=[i for i in range(540) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            fs=[pool.submit(event_job,i) for i in range(540) if i not in done]
            for f in as_completed(fs):
                done.append(f.result())
                if len(done)%90==0:print('Generated',len(done),'/540',flush=True)
        _write_json(root/'manifest.json',dict(selection_hash=_hash_file(root/'selection.json'),hashes={f.name:_hash_file(f) for f in (root/'events').iterdir()}))


def evaluate(root):
    p,s=verify(root);manifest=read(root/'manifest.json');assert manifest['selection_hash']==_hash_file(root/'selection.json')
    rows=[];parts={};rei=[];geo=p['geometry']['distances']['5']
    for i in range(540):
        path=root/'events'/f'{i:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:raw=z['d5']
        for name,x in raw_features(raw,p['preprocessing']['quiet'],p['preprocessing']['cut']).items():parts.setdefault(name,[]).append(x)
        r=read(path.with_suffix('.json'));rows.extend([dict(r,shot=j) for j in range(2)])
        rei.append(rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=1024,circuit_repetitions=1))
    rows=pd.DataFrame(rows);rows.to_csv(root/'test_rows.csv',index=False);y=rows[['epicenter_row','epicenter_col']].to_numpy()
    xs=reduce_features({n:np.concatenate(v) for n,v in parts.items()},joblib.load(root/'reduction.joblib'))
    pred={Path(name).stem:joblib.load(root/name).predict(xs[Path(name).stem.split('__')[0]]) for name in s['hashes'] if name!='reduction.joblib'}
    fit=pd.read_csv(root/'fit_rows.csv');center=fit[fit.role=='train'][['epicenter_row','epicenter_col']].mean().to_numpy()
    re=np.concatenate(rei);valid=np.isfinite(re).all(1);pred['REI']=np.where(valid[:,None],re,center);pred['Prior']=np.broadcast_to(center,y.shape)
    m=summarize(pred,rows);m.to_csv(root/'metrics.csv',index=False);frames=[]
    for name,v in pred.items():
        a=rows[['event_uid','shot','strength_band','geometry','test_seed']].copy();a['model']=name;a['pred_x'],a['pred_y']=v.T;a['error_mm']=np.linalg.norm(v-y,axis=1);frames.append(a)
    pd.concat(frames,ignore_index=True).to_csv(root/'predictions.csv',index=False)
    contrasts={(s['selected'],BASELINE),(s['unconstrained'],BASELINE)}
    contrasts.update((f'{name}__{g}',f'relative__{g}') for name in xs if name!='relative' for g in GAMMAS)
    contrasts.update((f'relative__{g}',BASELINE) for g in GAMMAS if g!='g120')
    pairs=[]
    for group,mask in masks(rows).items():
        if group not in ['weak','medium','strong']:continue
        for name,ref in sorted(contrasts):
            delta=np.linalg.norm(pred[name][mask]-y[mask],axis=1)-np.linalg.norm(pred[ref][mask]-y[mask],axis=1)
            delta=pd.DataFrame({'uid':rows.loc[mask,'event_uid'],'delta':delta}).groupby('uid').delta.mean().to_numpy()
            boot=np.random.default_rng(2026091127).choice(delta,(10000,len(delta))).mean(1)
            pairs.append(dict(group=group,model=name,reference=ref,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ci975=np.quantile(boot,[.0125,.9875]).tolist()))
    _write_json(root/'paired.json',pairs)
    blocks=[]
    for seed in TEST_SEEDS:
        mask=rows.test_seed==seed;a=summarize({n:v[mask] for n,v in pred.items()},rows[mask]);a['test_seed']=seed;blocks.append(a)
    pd.concat(blocks).to_csv(root/'test_seed_metrics.csv',index=False)
    old=set(fit.generation_seed)
    for folder in root.parent.iterdir():
        if folder==root:continue
        for path in (folder/'events').glob('*.json'):
            seed=read(path).get('generation_seed')
            if seed is not None:old.add(seed)
    assert rows.event_uid.nunique()==540 and not set(rows.generation_seed)&old
    _write_json(root/'audit.json',dict(status='passed',test_events=540,train_events=1080,validation_events=180,old_seed_overlap=0,hashes='passed',rei_answer_rate=float(valid.mean())))
    print(m[m.model.isin([s['selected'],s['unconstrained'],BASELINE,'REI'])&m.group.isin(['weak','medium','strong'])].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','generate','evaluate']);parser.add_argument('root',type=Path);args=parser.parse_args();globals()[args.stage](args.root)
