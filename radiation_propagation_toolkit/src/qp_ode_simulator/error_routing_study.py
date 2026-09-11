"""Event-cross-fitted error-directed mixtures, frozen before a fresh d5 test.

No truth metadata is accepted by inference. Strength is a training/evaluation
stratum, not an input. Prior validation is reused for selection, never fitting.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import strength_benchmark as benchmark
from .dataset import _write_json, _generation_lock
from .distance_growth import features
from .localization import _hash_file
from .specialist_study import load_fit, read
from .temporal_diagnosis import rei_center

SEEDS = (41, 42, 43)
TEST_SEEDS = (2026091061, 2026091161, 2026091261)
BANK = ('Ridge', 'SVR', 'ExtraTrees', 'ThreeHard')


def make_svr():
    return make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1, gamma=.57/169, epsilon=.1)))


def event_folds(rows, n_splits=3):
    """All shots of a physical event must stay together, strength stratified."""
    unique = rows.drop_duplicates('event_uid')
    folds = np.full(len(rows), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026091067)
    for fold, (_, held) in enumerate(splitter.split(unique, unique.strength_band)):
        folds[rows.event_uid.isin(unique.iloc[held].event_uid)] = fold
    assert (folds >= 0).all()
    return folds


def fit_bank(x, y, bands, destination):
    destination.mkdir(exist_ok=True)
    for name, allowed, kind in [('ridge',[0,1,2],'r'), ('svr',[0,1,2],'s'), ('medium',[1],'r'), ('strong',[2],'s')]:
        mask = np.isin(bands, allowed)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1000)) if kind == 'r' else make_svr()
        model.fit(x[mask], y[mask]); joblib.dump(model, destination/f'{name}.joblib')
    for seed in SEEDS:
        folder = destination/f'seed{seed}'; folder.mkdir(exist_ok=True)
        for name, allowed, leaf in [('global',[0,1,2],3), ('weak',[0],16), ('strong',[2],3)]:
            mask = np.isin(bands, allowed)
            model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=leaf, random_state=seed, n_jobs=2)
            model.fit(x[mask], y[mask]); joblib.dump(model, folder/f'{name}.joblib')
        gate = ExtraTreesClassifier(n_estimators=256, min_samples_leaf=16, random_state=seed, n_jobs=2)
        gate.fit(x, bands); joblib.dump(gate, folder/'gate.joblib')


def bank_predict(folder, x, medium_variants=None):
    det = {n:joblib.load(folder/f'{n}.joblib').predict(x) for n in ['ridge','svr','medium','strong']}
    byseed = {}
    for seed in SEEDS:
        models = {n:joblib.load(folder/f'seed{seed}'/f'{n}.joblib') for n in ['global','weak','strong','gate']}
        et = models['global'].predict(x); base = .5*(et+det['svr'])
        weak = models['weak'].predict(x); strong = .5*(models['strong'].predict(x)+det['strong'])
        gate = models['gate'].predict_proba(x)
        assert list(models['gate'].classes_) == [0,1,2]
        hard = np.eye(3)[gate.argmax(1)]
        pred = dict(Ridge=det['ridge'], SVR=det['svr'], ExtraTrees=et, Blend=base)
        for name, middle in {'ThreeHard':det['medium'], **(medium_variants or {})}.items():
            specialist = hard[:,0,None]*weak + hard[:,1,None]*middle + hard[:,2,None]*strong
            pred[name] = .5*(base+specialist)
        byseed[seed] = pred
    ensemble = {name:np.mean([byseed[s][name] for s in SEEDS],axis=0) for name in byseed[41]}
    return ensemble, byseed


def stack_input(x, predictions):
    return np.column_stack([x, predictions.reshape(len(x), -1)])


def mix(predictions, weights):
    if predictions.ndim != 3 or predictions.shape[-1] != 2 or weights.shape != predictions.shape[:2]:
        raise ValueError('expected N x K x 2 predictions and N x K weights')
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(1),1):
        raise ValueError('weights must be probabilities')
    return (predictions*weights[:,:,None]).sum(1)


def train_gate(x, predictions, y, bands, regularization, tail, seed):
    torch.set_num_threads(2); torch.manual_seed(seed)
    scaler = StandardScaler().fit(stack_input(x, predictions))
    z = torch.tensor(scaler.transform(stack_input(x,predictions)), dtype=torch.float32)
    pr = torch.tensor(predictions,dtype=torch.float32); target=torch.tensor(y,dtype=torch.float32)
    layer = torch.nn.Linear(z.shape[1],len(BANK)); torch.nn.init.normal_(layer.weight,std=.002)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(layer.parameters(),lr=.015)
    for _ in range(400):
        optimizer.zero_grad()
        w = torch.softmax(layer(z),dim=1); estimate=(w[:,:,None]*pr).sum(1)
        error=torch.sqrt(((estimate-target)**2).sum(1)+1e-8)
        losses=[]
        for band, weight in [(0,.2),(1,.4),(2,.4)]:
            e=error[bands==band]
            losses.append(weight*(e.mean()+tail*torch.topk(e,max(1,int(np.ceil(.1*len(e))))).values.mean()))
        loss=sum(losses)+regularization*layer.weight.square().sum()
        loss.backward(); optimizer.step()
    return dict(scaler=scaler, weight=layer.weight.detach().numpy(), bias=layer.bias.detach().numpy(), seed=seed,
                regularization=regularization,tail=tail)


def gate_predict(state, x, predictions):
    z=state['scaler'].transform(stack_input(x,predictions))
    logits=z@state['weight'].T+state['bias']; logits-=logits.max(1,keepdims=True)
    w=np.exp(logits);w/=w.sum(1,keepdims=True)
    return mix(predictions,w)


def metrics(predictions, rows, y):
    records=[]
    for name, pred in predictions.items():
        error=np.linalg.norm(pred-y,axis=1)
        for group,mask in benchmark.masks(rows).items():
            if np.any(mask):records.append(dict(model=name,group=group,events=rows.loc[mask,'event_uid'].nunique(),**benchmark.summary(error[mask])))
    return pd.DataFrame(records)


def diagnose(predictions, rows, y):
    out=[]
    for name,pred in predictions.items():
        a=rows.copy(); a['error_mm']=np.linalg.norm(pred-y,axis=1)
        for condition in ['epicenter_region','geometry','propagation_law','event_onset_ms','maximum_distance_mm','axis_ratio']:
            if pd.api.types.is_numeric_dtype(a[condition]):
                a['bin']=pd.qcut(a[condition],3,duplicates='drop').astype(str)
            else:a['bin']=a[condition].astype(str)
            for (band,value),r in a.groupby(['strength_band','bin']):
                out.append(dict(model=name,strength_band=int(band),condition=condition,value=value,events=r.event_uid.nunique(),**benchmark.summary(r.error_mm)))
    return pd.DataFrame(out)


def prepare(root):
    root.mkdir(exist_ok=False)
    parent=root.parent/'strength_benchmark_20260909';p=read(parent/'protocol.json')
    p.update(benchmark_parent=str(parent.resolve()),test_seeds=list(TEST_SEEDS),events=1080,
        primary='Validation-selected candidate minus ThreeHard, medium and strong means, paired event bootstrap97.5%; other comparisons descriptive95%.',
        search='3 middle choices; 6 direct Euclidean-error gates (L2=.001,.01,.1 x CVaR-tail=0,.2), each 3 optimization seeds; 2 OOF difficult-stratum weighted SVRs; base models eligible.',
        selection='Minimize equal medium/strong validation means, subject to each medium/strong p90 <= ThreeHard p90 + .02mm and weak mean <= ThreeHard + .03mm. Also freeze unconstrained minimum for secondary comparison.',
        fitting='2880 train events only; 3-fold strength-stratified event OOF; base OOF predictions average ET seeds41/42/43. All base preprocessing fitted within each fold. 360 historical validation events only for selection; never fit meta-models on validation.',
        limitations='Historical validation reused; strong elliptical events excluded from train and validation. Fresh test balanced as previous study; not cross-device or cross-simulator validation.',
        source_hash=_hash_file(Path(__file__)),dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['strength_benchmark.py','distance_growth.py','specialist_study.py','dataset.py','fault_response.py','simulator.py','stim_qec.py','distance_study.py','api.py']})
    _write_json(root/'protocol.json',p)


def train(root):
    p=read(root/'protocol.json');rows,x,y=load_fit(root)
    trainmask=rows.role=='train';valmask=rows.role=='validation'
    rt=rows[trainmask].reset_index(drop=True);xt=x[trainmask];yt=y[trainmask];bt=rt.strength_band.to_numpy()
    rv=rows[valmask].reset_index(drop=True);xv=x[valmask];yv=y[valmask]
    folds=event_folds(rt)
    rt[['event_uid','shot','strength_band']].assign(fold=folds).to_csv(root/'oof_folds.csv',index=False)
    oof=np.empty((len(rt),len(BANK),2))
    for fold in range(3):
        folder=root/f'fold{fold}';held=folds==fold
        assert not set(rt.loc[held,'event_uid'])&set(rt.loc[~held,'event_uid'])
        fit_bank(xt[~held],yt[~held],bt[~held],folder)
        predictions,_=bank_predict(folder,xt[held]);oof[held]=np.stack([predictions[n] for n in BANK],axis=1)
        print('OOF fold complete',fold,flush=True)
    np.save(root/'oof_predictions.npy',oof)
    middle=make_svr();middle.fit(xt[bt==1],yt[bt==1]);joblib.dump(middle,root/'medium_svr.joblib')
    parent=Path(p['benchmark_parent']);glob=joblib.load(parent/'svr.joblib').predict(xv)
    predictions,_=bank_predict(parent,xv,{'MiddleGlobalSVR':glob,'MiddleOnlySVR':middle.predict(xv)})
    valbank=np.stack([predictions[n] for n in BANK],axis=1)
    # Upweight difficult spatial/shape strata based only on OOF training errors.
    rt['oof_error']=np.linalg.norm(oof[:,-1]-yt,axis=1)
    groupcols=['strength_band','epicenter_region','geometry']
    stratum=rt.groupby(groupcols).oof_error.transform('mean').to_numpy()
    average=rt.groupby('strength_band').oof_error.transform('mean').to_numpy()
    for power in [1,2]:
        weights=np.clip((stratum/average)**power,.5,3)
        weights[bt==0]=1
        model=make_svr();model.fit(xt,yt,multioutputregressor__sample_weight=weights)
        name=f'StratumSVR{power}';joblib.dump(model,root/f'{name}.joblib')
        predictions[name]=model.predict(xv)
    rt.groupby(groupcols).agg(events=('event_uid','nunique'),oof_error=('oof_error','mean')).to_csv(root/'training_difficult_strata.csv')
    for reg in [.001,.01,.1]:
        for tail in [0,.2]:
            name=f'ErrorGate_r{reg}_t{tail}'
            states=[train_gate(xt,oof,yt,bt,reg,tail,seed) for seed in SEEDS]
            joblib.dump(states,root/f'{name}.joblib')
            predictions[name]=np.mean([gate_predict(s,xv,valbank) for s in states],axis=0)
            print('Gate complete',name,flush=True)
    # Existing Overlap, unchanged, added as reference but not eligible for primary selection.
    prob=np.mean([joblib.load(parent/f'seed{s}'/'gate.joblib').predict_proba(xv) for s in SEEDS],axis=0)
    low=joblib.load(parent/'low.joblib').predict(xv);high=joblib.load(parent/'high.joblib').predict(xv)
    parts=[]
    for seed in SEEDS:
        q=joblib.load(parent/f'seed{seed}'/'gate.joblib').predict_proba(xv)
        weight=np.choose(q.argmax(1),[1.,.5,0.])
        et=joblib.load(parent/f'seed{seed}'/'global.joblib').predict(xv)
        parts.append(.5*(.5*(glob+et)+weight[:,None]*low+(1-weight[:,None])*high))
    predictions['Overlap']=np.mean(parts,axis=0)
    scores=metrics(predictions,rv,yv);scores.to_csv(root/'validation_metrics.csv',index=False)
    diagnose({n:predictions[n] for n in ['ThreeHard','SVR','Overlap']},rv,yv).to_csv(root/'validation_failure_analysis.csv',index=False)
    eligible=[];r=scores[scores.model=='ThreeHard'].set_index('group');objectives={}
    for name,g in scores.groupby('model'):
        if name=='Overlap':continue
        g=g.set_index('group');objectives[name]=float(.5*(g.loc['medium','mean_mm']+g.loc['strong','mean_mm']))
        if all(g.loc[k,'p90_mm']<=r.loc[k,'p90_mm']+.02 for k in ['medium','strong']) and g.loc['weak','mean_mm']<=r.loc['weak','mean_mm']+.03:
            eligible.append(name)
    chosen=min(eligible,key=objectives.get);unconstrained=min(objectives,key=objectives.get)
    selection=dict(selected=chosen,unconstrained=unconstrained,eligible=eligible,objective=objectives,
        status='frozen before any fresh test generation',source_hash=_hash_file(Path(__file__)),protocol_hash=_hash_file(root/'protocol.json'),
        local_hashes={str(f.relative_to(root)):_hash_file(f) for f in root.rglob('*.joblib')},
        parent_hashes={str(f.relative_to(parent)):_hash_file(f) for f in parent.rglob('*.joblib')},
        history=read(parent/'selection.json')['history'],center=yt.mean(0).tolist())
    _write_json(root/'selection.json',selection)
    print('SELECTED',chosen,'unconstrained',unconstrained,flush=True)
    print(scores[scores.model.isin([chosen,unconstrained,'ThreeHard','SVR'])].to_string(index=False),flush=True)


def initialize(root):
    benchmark.initialize(root)
    benchmark.TEST_SEEDS=tuple(read(Path(root)/'protocol.json')['test_seeds'])


def verify(root):
    p=read(root/'protocol.json');s=read(root/'selection.json')
    assert s['source_hash']==_hash_file(Path(__file__)) and s['protocol_hash']==_hash_file(root/'protocol.json')
    for name,digest in p['dependencies'].items():assert _hash_file(Path(__file__).with_name(name))==digest
    for name,digest in s['local_hashes'].items():assert _hash_file(root/name)==digest
    for name,digest in s['parent_hashes'].items():assert _hash_file(Path(p['benchmark_parent'])/name)==digest
    return p,s


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
    raw=[];rows=[]
    for i in range(1080):
        path=root/'events'/f'{i:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:bits=z['d5']
        assert bits.shape==(2,24,2047) and np.isin(bits,[0,1]).all()
        raw.append(bits);r=read(path.with_suffix('.json'));rows.extend([dict(r,shot=j) for j in range(2)])
    raw=np.concatenate(raw);rows=pd.DataFrame(rows);y=rows[['epicenter_row','epicenter_col']].to_numpy()
    old_seeds=set()
    for other in root.parent.iterdir():
        if other!=root and other.is_dir():
            for path in (other/'events').glob('*.json'):
                value=read(path).get('generation_seed')
                if value is not None:old_seeds.add(value)
    unique=rows.drop_duplicates('event_uid');assert len(unique)==1080 and unique.generation_seed.nunique()==1080
    assert not set(unique.generation_seed)&old_seeds
    for test in TEST_SEEDS:
        counts=unique[unique.test_seed==test].groupby(['strength_band','geometry','propagation_law','epicenter_region']).size()
        assert len(counts)==36 and (counts==10).all()
    start=time.perf_counter();x=features(raw,p['preprocessing']);feature_seconds=time.perf_counter()-start
    parent=Path(p['benchmark_parent']);start=time.perf_counter()
    middle=joblib.load(root/'medium_svr.joblib').predict(x);glob=joblib.load(parent/'svr.joblib').predict(x)
    pred,byseed=bank_predict(parent,x,{'MiddleGlobalSVR':glob,'MiddleOnlySVR':middle})
    bank_seconds=time.perf_counter()-start
    bank=np.stack([pred[n] for n in BANK],axis=1);optimization_seeds={}
    for path in sorted(root.glob('ErrorGate*.joblib')):
        states=joblib.load(path);answers=[gate_predict(state,x,bank) for state in states]
        pred[path.stem]=np.mean(answers,axis=0);optimization_seeds[path.stem]=answers
    for name in ['StratumSVR1','StratumSVR2']:pred[name]=joblib.load(root/f'{name}.joblib').predict(x)
    low=joblib.load(parent/'low.joblib').predict(x);high=joblib.load(parent/'high.joblib').predict(x)
    parts=[]
    for seed in SEEDS:
        q=joblib.load(parent/f'seed{seed}'/'gate.joblib').predict_proba(x)
        weight=np.choose(q.argmax(1),[1.,.5,0.])
        parts.append(.5*(byseed[seed]['Blend']+weight[:,None]*low+(1-weight[:,None])*high))
    pred['Overlap']=np.mean(parts,axis=0)
    geo=p['geometry']['distances']['5'];start=time.perf_counter()
    rei=rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=s['history'],circuit_repetitions=1)
    rei_seconds=time.perf_counter()-start
    answered=np.isfinite(rei).all(1);pred['REI']=np.where(answered[:,None],rei,s['center'])
    pred['Prior']=np.broadcast_to(s['center'],y.shape)
    metrics(pred,rows,y).to_csv(root/'metrics.csv',index=False)
    diagnose({n:pred[n] for n in set([s['selected'],'ThreeHard','REI','SVR'])},rows,y).to_csv(root/'test_failure_analysis.csv',index=False)
    frames=[]
    for name,v in pred.items():
        f=rows[['event_uid','shot','test_seed','strength_band','geometry','epicenter_region','propagation_law','epicenter_row','epicenter_col']].copy()
        f['model']=name;f['pred_x'],f['pred_y']=v.T;f['error_mm']=np.linalg.norm(v-y,axis=1);frames.append(f)
    pd.concat(frames,ignore_index=True).to_csv(root/'predictions.csv',index=False)
    pairs=[]
    for group,mask in benchmark.masks(rows).items():
        for name in sorted(set([s['selected'],s['unconstrained'],'MiddleGlobalSVR','MiddleOnlySVR'])):
            for reference in ['ThreeHard','REI']:
                error=np.linalg.norm(pred[name][mask]-y[mask],axis=1);ref=np.linalg.norm(pred[reference][mask]-y[mask],axis=1)
                delta=pd.DataFrame({'uid':rows.loc[mask,'event_uid'],'delta':error-ref}).groupby('uid').delta.mean().to_numpy()
                rng=np.random.default_rng(2026091069);boot=rng.choice(delta,(10000,len(delta))).mean(1)
                pairs.append(dict(group=group,model=name,reference=reference,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ci975=np.quantile(boot,[.0125,.9875]).tolist()))
    _write_json(root/'paired.json',pairs)
    blocks=[]
    for test in TEST_SEEDS:
        mask=rows.test_seed==test
        a=metrics({n:v[mask] for n,v in pred.items()},rows[mask],y[mask]);a['test_seed']=test;blocks.append(a)
    pd.concat(blocks).to_csv(root/'test_seed_metrics.csv',index=False)
    variations=[]
    for name in [s['selected'],s['unconstrained'],'ThreeHard','MiddleGlobalSVR','MiddleOnlySVR']:
        if name in optimization_seeds:answers=optimization_seeds[name];kind='gate initialization only; fixed base ensemble'
        elif name in byseed[41]:answers=[byseed[seed][name] for seed in SEEDS];kind='base tree/classifier seed'
        else:continue
        for seed,v in zip(SEEDS,answers):
            a=metrics({name:v},rows,y);a['seed']=seed;a['variation']=kind;variations.append(a)
    pd.concat(variations).drop_duplicates().to_csv(root/'seed_variation.csv',index=False)
    _write_json(root/'audit.json',dict(status='passed',events=1080,old_generation_seed_overlap=0,balanced_36_strata_per_root=True,
        hashes='passed',rei_answer_rate=float(answered.mean()),selection_hash=_hash_file(root/'selection.json'),
        timing=dict(shots=len(rows),features_seconds=feature_seconds,all_base_variants_including_loading_seconds=bank_seconds,rei_batch_seconds=rei_seconds,
                    caveat='Diagnostic batch wall time, not optimized streaming latency; base bank includes multiple comparison variants.')))
    print(metrics({n:pred[n] for n in set([s['selected'],s['unconstrained'],'ThreeHard','REI','SVR'])},rows,y).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','generate','evaluate']);parser.add_argument('root',type=Path)
    args=parser.parse_args();globals()[args.stage](args.root)
