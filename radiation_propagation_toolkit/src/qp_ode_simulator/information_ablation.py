"""Controlled information/observation-duration ablation with retraining.

All windows start at the same time; short windows may end before impact.
These are censored observations, not noise-free or truth-aligned windows.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from . import strength_benchmark as benchmark
from .dataset import _write_json,_generation_lock
from .distance_growth import features,load_new,training_indices
from .distance_study import load as old_load,ordinary
from .error_routing_study import bank_predict,metrics
from .localization import _hash_file
from .specialist_study import load_fit,read
from .temporal_diagnosis import rei_center

VARIANTS=('full','excess_only','relative_only','time_average','no_spatial','global_quiet','window512','window1024')
SEEDS=(2026091081,2026091181,2026091281)


def representations(raw,preprocessing):
    x=features(raw,preprocessing);n=len(x);excess=x[:,:96].reshape(n,24,4)
    cut=preprocessing['cut'];ticks=raw.shape[2]
    edges=np.array([0,cut,cut+(ticks-cut)//3,cut+2*(ticks-cut)//3,ticks])
    global_q=dict(preprocessing,quiet=np.full(24,np.mean(preprocessing['quiet'])).tolist())
    return dict(full=x,excess_only=x[:,:96],relative_only=x[:,96:168],
        time_average=(excess*np.diff(edges)[None,None,:]/ticks).sum(2),
        no_spatial=excess.mean(1),global_quiet=features(raw,global_q),
        window512=features(raw[...,:511],preprocessing),window1024=features(raw[...,:1023],preprocessing))


def current(root,x):
    p=read(root/'protocol.json');base=Path(p['benchmark_parent']);previous=Path(p['previous'])
    middle=joblib.load(previous/'medium_svr.joblib').predict(x)
    result,_=bank_predict(base,x,{'Current':middle})
    return result['Current']


def prepare(root):
    root.mkdir(exist_ok=False);previous=root.parent/'error_routing_20260910';p=read(previous/'protocol.json')
    p.update(previous=str(previous.resolve()),test_seeds=list(SEEDS),variants=list(VARIANTS),
        study='Eight information representations x Ridge/SVR/ET; 24 fits, unchanged training events, paired fresh test.',
        selection='Equal medium/strong mean; each p90 <= Current+.02mm, weak mean <= Current+.03mm; Current eligible. Also freeze unconstrained choice.',
        primary='Validation-selected minus Current, medium and strong means, paired whole-event bootstrap97.5%. All feature/window contrasts are exploratory descriptive95%, no familywise significance claims.',
        fitting='2880 train,360 historical validation, same split and same fitting events in all 24 conditions. Ridge alpha1000, SVR C1 gamma .57/dimension epsilon.1, ET256 leaf3 seed41. Fixed recipes; not exhaustive independently optimized ablations.',
        caveats='Full and short observations are SAME events, not extra independent events. Short windows may contain no event. Full endpoint is offline and cannot guide earlier decisions. No physical noise or hardware intervention in this study.',
        source_hash=_hash_file(Path(__file__)))
    _write_json(root/'protocol.json',p)


def fit_variant(args):
    root,variant=args;rows,_,y=load_fit(root);x=np.load(root/f'x_{variant}.npy');tr=rows.role=='train';v=rows.role=='validation'
    predictions={}
    for family in ['Ridge','SVR','ET']:
        if family=='Ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=1000))
        elif family=='SVR':model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=1,gamma=.57/x.shape[1],epsilon=.1)))
        else:model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=3,random_state=41,n_jobs=2)
        model.fit(x[tr],y[tr]);name=f'{variant}__{family}';joblib.dump(model,root/f'{name}.joblib')
        predictions[name]=model.predict(x[v]);print('Fit',name,flush=True)
    np.savez_compressed(root/f'validation_{variant}.npz',**predictions)


def train(root):
    p=read(root/'protocol.json');parent=Path(p['parent']);source=Path(p['source'])
    expected,baseline,y=load_fit(root)
    old,orows=old_load(Path(read(parent/'protocol.json')['parent']),5,{'train'});orows['origin']='old'
    new,nrows=load_new(parent,{'train','validation'})
    rows=pd.concat([orows,nrows],ignore_index=True);raw=np.concatenate([old,new]);del old,new
    keep=np.r_[training_indices(rows,2880),np.flatnonzero((rows.role=='validation')&ordinary(rows))]
    rows=rows.iloc[keep].reset_index(drop=True);raw=raw[keep]
    assert rows[['event_uid','shot']].equals(expected[['event_uid','shot']])
    xs=representations(raw,p['preprocessing']);np.testing.assert_allclose(xs['full'],baseline,rtol=1e-6,atol=1e-6)
    for name,x in xs.items():np.save(root/f'x_{name}.npy',x)
    del xs,raw
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(fit_variant,[(root,v) for v in VARIANTS]))
    val=rows.role=='validation';predictions={'Current':current(root,baseline[val])}
    for variant in VARIANTS:
        with np.load(root/f'validation_{variant}.npz') as z:predictions.update({n:z[n] for n in z.files})
    scores=metrics(predictions,rows[val],y[val]);scores.to_csv(root/'validation_metrics.csv',index=False)
    ref=scores[scores.model=='Current'].set_index('group');objectives={};eligible=[]
    for name,g in scores.groupby('model'):
        g=g.set_index('group');objectives[name]=float(g.loc[['medium','strong'],'mean_mm'].mean())
        if all(g.loc[b,'p90_mm']<=ref.loc[b,'p90_mm']+.02 for b in ['medium','strong']) and g.loc['weak','mean_mm']<=ref.loc['weak','mean_mm']+.03:eligible.append(name)
    s=dict(selected=min(eligible,key=objectives.get),unconstrained=min(objectives,key=objectives.get),eligible=eligible,objective=objectives,
        source_hash=_hash_file(Path(__file__)),protocol_hash=_hash_file(root/'protocol.json'),
        model_hashes={f.name:_hash_file(f) for f in root.glob('*.joblib')},
        previous_selection_hash=_hash_file(Path(p['previous'])/'selection.json'),
        baseline_hashes={str(f.resolve()):_hash_file(f) for folder in [Path(p['benchmark_parent']),Path(p['previous'])] for f in folder.glob('*.joblib')},
        seed_model_hashes={str(f.resolve()):_hash_file(f) for f in Path(p['benchmark_parent']).glob('seed*/*.joblib')},
        dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['dataset.py','distance_study.py','distance_growth.py','strength_benchmark.py','error_routing_study.py','adaptive_localization.py','simulator.py','api.py','fault_response.py','stim_qec.py']},
        history=read(Path(p['previous'])/'selection.json')['history'],center=y[rows.role=='train'].mean(0).tolist(),status='Frozen before fresh test generation')
    _write_json(root/'selection.json',s);print('SELECTED',s['selected'],'unconstrained',s['unconstrained'],flush=True)


def verify(root):
    p=read(root/'protocol.json');s=read(root/'selection.json')
    assert s['source_hash']==_hash_file(Path(__file__)) and s['protocol_hash']==_hash_file(root/'protocol.json')
    for name,digest in s['model_hashes'].items():assert _hash_file(root/name)==digest
    for field in ['baseline_hashes','seed_model_hashes']:
        for name,digest in s[field].items():assert _hash_file(Path(name))==digest
    for name,digest in s['dependencies'].items():assert _hash_file(Path(__file__).with_name(name))==digest
    return p,s


def initialize(root):
    benchmark.initialize(root);benchmark.TEST_SEEDS=SEEDS


def generate(root):
    verify(root);(root/'events').mkdir(exist_ok=True)
    with _generation_lock(root):
        done=[i for i in range(1080) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            futures=[pool.submit(benchmark.event_job,i) for i in range(1080) if i not in done]
            for f in as_completed(futures):
                done.append(f.result())
                if len(done)%108==0:print('Generated',len(done),'/1080',flush=True)
        _write_json(root/'manifest.json',dict(selection_hash=_hash_file(root/'selection.json'),hashes={f.name:_hash_file(f) for f in (root/'events').iterdir()}))


def evaluate(root):
    p,s=verify(root);manifest=read(root/'manifest.json');assert manifest['selection_hash']==_hash_file(root/'selection.json')
    rows=[];raw=[]
    for i in range(1080):
        path=root/'events'/f'{i:04d}.npz'
        for ext in ['.json','.npz']:assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        r=read(path.with_suffix('.json'));rows.extend([dict(r,shot=j) for j in range(2)])
        with np.load(path) as z:raw.append(z['d5'])
    raw=np.concatenate(raw);rows=pd.DataFrame(rows);rows.to_csv(root/'test_rows.csv',index=False)
    y=rows[['epicenter_row','epicenter_col']].to_numpy();xs=representations(raw,p['preprocessing'])
    predictions={'Current':current(root,xs['full'])}
    for name in s['model_hashes']:
        variant=Path(name).stem.split('__')[0];predictions[Path(name).stem]=joblib.load(root/name).predict(xs[variant])
    geo=p['geometry']['distances']['5']
    for name,ticks in [('REI',2047),('REI512',511),('REI1024',1023)]:
        result=rei_center(raw[...,:ticks],geo['sites_mm'],geo['physical_mm'],history_length=min(ticks,s['history']),circuit_repetitions=1)
        valid=np.isfinite(result).all(1);predictions[name]=np.where(valid[:,None],result,s['center'])
        _write_json(root/f'{name}_answer_rate.json',dict(rate=float(valid.mean()),history=min(ticks,s['history'])))
    frames=[]
    for name,pred in predictions.items():
        f=rows[['event_uid','shot','strength_band','geometry','test_seed']].copy();f['model']=name
        f['pred_x'],f['pred_y']=pred.T;f['error_mm']=np.linalg.norm(pred-y,axis=1);frames.append(f)
    df=pd.concat(frames,ignore_index=True);df.to_csv(root/'predictions.csv',index=False)
    scores=metrics(predictions,rows,y);scores.to_csv(root/'metrics.csv',index=False)
    contrasts=[]
    comparisons={(s['selected'],'Current'),(s['unconstrained'],'Current')}
    comparisons.update((f'{variant}__{family}',f'full__{family}') for variant in VARIANTS if variant!='full' for family in ['SVR','Ridge','ET'])
    for group,mask in benchmark.masks(rows).items():
        if group not in ['weak','medium','strong']:continue
        for name,reference in sorted(comparisons):
            delta=np.linalg.norm(predictions[name][mask]-y[mask],axis=1)-np.linalg.norm(predictions[reference][mask]-y[mask],axis=1)
            delta=pd.DataFrame({'uid':rows.loc[mask,'event_uid'],'delta':delta}).groupby('uid').delta.mean().to_numpy()
            rng=np.random.default_rng(2026091087);boot=rng.choice(delta,(10000,len(delta))).mean(1)
            contrasts.append(dict(group=group,model=name,reference=reference,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ci975=np.quantile(boot,[.0125,.9875]).tolist()))
    _write_json(root/'paired.json',contrasts)
    # Shared event IDs, not independent observations, across every condition.
    source=pd.read_csv(Path(p['source'])/'rows.csv');old=set(source.generation_seed)
    for folder in root.parent.iterdir():
        if folder==root:continue
        for path in (folder/'events').glob('*.json'):
            seed=read(path).get('generation_seed')
            if seed is not None:old.add(seed)
    unique=rows.drop_duplicates('event_uid');assert unique.generation_seed.nunique()==1080 and not set(unique.generation_seed)&old
    for test in SEEDS:
        count=unique[unique.test_seed==test].groupby(['strength_band','geometry','propagation_law','epicenter_region']).size()
        assert len(count)==36 and (count==10).all()
    blocks=[]
    for test in SEEDS:
        mask=rows.test_seed==test;a=metrics({n:v[mask] for n,v in predictions.items()},rows[mask],y[mask]);a['test_seed']=test;blocks.append(a)
    pd.concat(blocks).to_csv(root/'test_seed_metrics.csv',index=False)
    _write_json(root/'audit.json',dict(status='passed',events=1080,old_seed_overlap=0,hashes='passed',balanced_strata=True,selection=s['selected']))
    print(scores[scores.group.isin(['medium','strong'])].pivot(index='model',columns='group',values='mean_mm').to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','generate','evaluate']);parser.add_argument('root',type=Path)
    args=parser.parse_args();globals()[args.stage](args.root)
