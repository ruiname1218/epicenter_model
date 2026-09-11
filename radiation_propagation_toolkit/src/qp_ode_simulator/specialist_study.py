"""Disjoint/overlapping strength experts; observed routing versus truth-only diagnostics."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .api import run_simulation
from .dataset import event_configuration, _seed, _write_json, _generation_lock
from .distance_growth import features
from .distance_study import fields_to_pauli, group_masks
from .fault_response import sample
from .localization import _hash_file
from .temporal_diagnosis import rei_center

SEED = 2026090923
EVENTS = 720
POOLS = {'weak': [0], 'medium': [1], 'strong': [2], 'low_overlap': [0,1], 'high_overlap': [1,2]}
PROFILES = {'strict_two': ['weak','strong'], 'overlap_two': ['low_overlap','high_overlap'], 'three': ['weak','medium','strong']}


def read(path): return json.loads(Path(path).read_text())


def mix_experts(predictions, probabilities, profile):
    """Only observed gate probabilities; truth routing is a separate caller."""
    prob = np.asarray(probabilities)
    if prob.ndim != 2 or prob.shape[1] != 3 or not np.isfinite(prob).all() or np.any(prob < 0) or not np.allclose(prob.sum(1), 1):
        raise ValueError('expected weak/medium/strong probabilities')
    names = PROFILES[profile]
    if len(names) == 3:
        return sum(prob[:, i, None]*predictions[name] for i, name in enumerate(names))
    weight = prob[:, 0]+.5*prob[:, 1]
    return weight[:, None]*predictions[names[0]]+(1-weight[:, None])*predictions[names[1]]


def prepare(root):
    root.mkdir(exist_ok=False)
    source = root.parent/'syndrome_features_20260909'; p = read(source/'protocol.json')
    protocol = dict(seed=SEED, events=EVENTS, source=str(source.resolve()), parent=p['parent'],
        base=p['base'], sampling=p['sampling'], circuit=p['circuit'], geometry=p['geometry'], response=p['response'], preprocessing=p['preprocessing'],
        pools=POOLS, profiles=PROFILES, training_events=2880, validation_events=360,
        selection='Experts: lowest validation error within their own strength pool. Gate/profile/hard-soft/parent mixture: ordinary validation mean. Also weak and strong-circle validation minima constrained to ordinary <= parent + .01 mm.',
        search='Each of five expert pools: Ridge100/1000, SVR C1 gamma .57/169 or C10 gamma .057/169, ET leaf3/16, SVR/ET blend. Four three-class gates (logistic C.1/1, ET leaf16/64). 30 regressor + 4 classifier fits.',
        routing='strict_two and overlap_two: weight Pweak+.5*Pmedium; three: class probabilities. Hard routing uses argmax one-hot. Parent-mixture fraction .5 or 1; parent itself is eligible.',
        oracle='True-strength one-hot routing is privileged diagnostic, not deployable performance or a formal information-theoretic upper bound.',
        test='Fresh 720 independent events (600 ordinary, 120 strong elliptical); 2 shots per event scored separately, bootstrap clusters both shots.',
        exclusion='Strong ellipses excluded from fitting and model selection; all historical test sets excluded.',
        primary='Validation-selected practical model minus parent on ordinary fresh test; subgroup/other profiles/oracles secondary descriptive 95% paired event bootstrap, 10000 resamples.',
        source_hash=_hash_file(Path(__file__)), cache_hashes={name:_hash_file(source/name) for name in ['rows.csv','features_baseline.npy','protocol.json']})
    _write_json(root/'protocol.json', protocol)


def load_fit(root):
    p = read(root/'protocol.json'); source = Path(p['source'])
    for name, digest in p['cache_hashes'].items(): assert _hash_file(source/name) == digest
    rows = pd.read_csv(source/'rows.csv', float_precision='round_trip')
    x = np.load(source/'features_baseline.npy'); y = rows[['epicenter_row','epicenter_col']].to_numpy()
    assert x.shape == (6480,169) and rows[rows.role == 'train'].event_uid.nunique() == 2880
    assert rows[rows.role == 'validation'].event_uid.nunique() == 360
    assert not ((rows.geometry == 'elliptical') & (rows.strength_band == 2)).any()
    assert not set(rows[rows.role == 'train'].event_uid) & set(rows[rows.role == 'validation'].event_uid)
    return rows, x, y


def fit_expert(args):
    root, pool = args; out = root/pool; out.mkdir(exist_ok=True)
    if (out/'frozen.json').exists(): return pool
    rows, x, y = load_fit(root)
    train = np.flatnonzero((rows.role == 'train') & rows.strength_band.isin(POOLS[pool]))
    val = np.flatnonzero(rows.role == 'validation')
    local = rows.iloc[val].strength_band.isin(POOLS[pool]).to_numpy()
    preds, logs = {}, {}
    configs = [('Ridge100', 'ridge', {'alpha':100}), ('Ridge1000','ridge',{'alpha':1000}),
               ('SVR1','svr',{'C':1,'gamma':.57/169}),('SVR10','svr',{'C':10,'gamma':.057/169}),
               ('ET3','et',{'min_samples_leaf':3}),('ET16','et',{'min_samples_leaf':16})]
    score = lambda v: float(np.linalg.norm(v[local]-y[val][local],axis=1).mean())
    for name, family, config in configs:
        if family == 'ridge': model = make_pipeline(StandardScaler(),Ridge(**config))
        elif family == 'svr': model = make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(**config,epsilon=.1)))
        else: model = ExtraTreesRegressor(n_estimators=256,random_state=41,n_jobs=2,**config)
        model.fit(x[train],y[train]); preds[name] = model.predict(x[val]); logs[name] = score(preds[name])
        joblib.dump(model,out/f'{name}.joblib')
        print(pool,name,'within-pool validation',logs[name],flush=True)
    svr = min(['SVR1','SVR10'],key=logs.get); et = min(['ET3','ET16'],key=logs.get)
    preds['blend'] = .5*(preds[svr]+preds[et]); logs['blend'] = score(preds['blend'])
    selected = min(logs,key=logs.get)
    np.savez_compressed(out/'validation.npz',**preds)
    info = dict(selected=selected,blend=[svr,et],scores=logs,train_events=rows.iloc[train].event_uid.nunique(),
        validation_events=rows.iloc[val[local]].event_uid.nunique(),center=y[train].mean(0).tolist(),
        training_uids=rows.iloc[train].event_uid.drop_duplicates().tolist())
    _write_json(out/'frozen.json',info)
    return pool


def train(root,workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for name in executor.map(fit_expert,[(root,pool) for pool in POOLS]): print('Expert finished',name,flush=True)
    rows,x,y = load_fit(root); train = np.flatnonzero(rows.role == 'train'); val = np.flatnonzero(rows.role == 'validation')
    labels = rows.strength_band.to_numpy(); logs = {}; probs = {}
    for name, model in [('logistic01',make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500))),
                        ('logistic1',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1500))),
                        ('ET16',ExtraTreesClassifier(n_estimators=256,min_samples_leaf=16,random_state=41,n_jobs=2)),
                        ('ET64',ExtraTreesClassifier(n_estimators=256,min_samples_leaf=64,random_state=41,n_jobs=2))]:
        model.fit(x[train],labels[train]); assert list(model.classes_) == [0,1,2]
        probs[name] = model.predict_proba(x[val]); logs[name] = float(log_loss(labels[val],probs[name],labels=[0,1,2]))
        joblib.dump(model,root/f'gate_{name}.joblib'); print('Gate',name,'log loss',logs[name],flush=True)
    np.savez_compressed(root/'gate_validation.npz',**probs); _write_json(root/'gates.json',logs)


def freeze(root):
    assert not (root/'selection.json').exists()
    p = read(root/'protocol.json'); parent = Path(p['parent'])/'n2880'
    rows,x,y = load_fit(root); val = np.flatnonzero(rows.role == 'validation'); x=x[val]; y=y[val]; labels=rows.iloc[val].strength_band.to_numpy()
    experts = {pool:read(root/pool/'frozen.json') for pool in POOLS}; epred = {}
    for pool, info in experts.items():
        with np.load(root/pool/'validation.npz') as z: epred[pool] = z[info['selected']]
    base = .5*(joblib.load(parent/'SVR.joblib').predict(x)+joblib.load(parent/'ExtraTrees.joblib').predict(x))
    score = lambda pred: {group:float(np.linalg.norm(pred[mask]-y[mask],axis=1).mean()) for group,mask in [('ordinary',np.ones(len(y),bool)),('weak',labels==0),('strong',labels==2)]}
    scores = {'parent':score(base)}; specs = {'parent':{'kind':'parent'}}
    with np.load(root/'gate_validation.npz') as z:
        for profile in PROFILES:
            for gate in z.files:
                for hard in (False,True):
                    prob = np.eye(3)[z[gate].argmax(1)] if hard else z[gate]
                    prediction = mix_experts(epred,prob,profile)
                    for fraction in (.5,1.):
                        name=f'{profile}/{gate}/{"hard" if hard else "soft"}/{fraction}'
                        specs[name]=dict(kind='practical',profile=profile,gate=gate,hard=hard,fraction=fraction)
                        scores[name]=score(base+fraction*(prediction-base))
    chosen = min(scores,key=lambda k:scores[k]['ordinary'])
    eligible = [k for k in scores if scores[k]['ordinary'] <= scores['parent']['ordinary']+.01]
    chosen_weak = min(eligible,key=lambda k:scores[k]['weak']); chosen_strong=min(eligible,key=lambda k:scores[k]['strong'])
    profiles = {profile:min([k for k in scores if k.startswith(profile+'/')],key=lambda k:scores[k]['ordinary']) for profile in PROFILES}
    oracle = {profile:score(mix_experts(epred,np.eye(3)[labels],profile)) for profile in PROFILES}
    selection = dict(primary=chosen,weak=chosen_weak,strong=chosen_strong,profiles=profiles,specs=specs,scores=scores,oracle_validation=oracle,
        experts=experts,center=read(parent/'frozen.json')['center'],source_hash=_hash_file(Path(__file__)),
        status='frozen before fresh test generation',protocol_hash=_hash_file(root/'protocol.json'),
        hashes={str(f.relative_to(root)):_hash_file(f) for f in root.rglob('*.joblib')},
        parent_hashes={name:_hash_file(parent/name) for name in ['SVR.joblib','ExtraTrees.joblib']})
    _write_json(root/'selection.json',selection)
    pd.DataFrame(scores).T.sort_values('ordinary').to_csv(root/'validation_scores.csv')
    print('Selected',chosen,scores[chosen],'weak',chosen_weak,scores[chosen_weak],'strong',chosen_strong,scores[chosen_strong],flush=True)
    print('Oracle validation',oracle,flush=True)


_WORKER=None


def initialize(root):
    global _WORKER
    _WORKER=Path(root),read(Path(root)/'protocol.json')


def event_job(i):
    root,p=_WORKER
    config,labels=event_configuration(p['base'],p['sampling'],i,SEED,123)
    simulation=run_simulation(config,coords_mm=np.asarray(p['geometry']['union_mm']))
    xyz=fields_to_pauli(simulation,p['circuit'],np.asarray(p['geometry']['distances']['5']['union_indices']))
    bits,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(SEED+1,i,5)%(2**63-1),shots=2)
    row=simulation.parameters.iloc[0].to_dict(); row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(labels,event=i,event_uid=f'd5-specialist-{SEED}:{i}',role='test')
    path=root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream: np.savez_compressed(stream,d5=bits)
    path.with_suffix('.npz.tmp').replace(path); _write_json(path.with_suffix('.json'),row)
    return i


def generate(root,workers):
    s=read(root/'selection.json'); assert s['source_hash']==_hash_file(Path(__file__))
    (root/'events').mkdir(exist_ok=True)
    with _generation_lock(root):
        done=[i for i in range(EVENTS) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),)) as executor:
            futures=[executor.submit(event_job,i) for i in range(EVENTS) if i not in done]
            for f in as_completed(futures):
                done.append(f.result())
                if len(done)%72==0: print('Fresh events',len(done),'/',EVENTS,flush=True)
        _write_json(root/'manifest.json',dict(events=EVENTS,selection_hash=_hash_file(root/'selection.json'),hashes={f.name:_hash_file(f) for f in (root/'events').iterdir()}))


def predict_expert(root,pool,info,x):
    if info['selected']=='blend': return np.mean([joblib.load(root/pool/f'{name}.joblib').predict(x) for name in info['blend']],axis=0)
    return joblib.load(root/pool/f'{info["selected"]}.joblib').predict(x)


def evaluate(root):
    p=read(root/'protocol.json'); s=read(root/'selection.json'); manifest=read(root/'manifest.json')
    assert _hash_file(Path(__file__))==s['source_hash'] and _hash_file(root/'protocol.json')==s['protocol_hash']
    assert _hash_file(root/'selection.json')==manifest['selection_hash']
    for name,digest in s['hashes'].items(): assert _hash_file(root/name)==digest
    raw,rows=[],[]
    for i in range(EVENTS):
        path=root/'events'/f'{i:04d}.npz'
        for ext in ('.npz','.json'): assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z: bits=z['d5']
        assert bits.shape==(2,24,2047) and np.isin(bits,[0,1]).all(); raw.append(bits)
        row=read(path.with_suffix('.json')); rows.extend([dict(row,shot=j) for j in range(2)])
    raw=np.concatenate(raw); rows=pd.DataFrame(rows); y=rows[['epicenter_row','epicenter_col']].to_numpy(); labels=rows.strength_band.to_numpy()
    previous,_,_=load_fit(root); assert not set(previous.generation_seed)&set(rows.generation_seed)
    x=features(raw,p['preprocessing']); parent=Path(p['parent'])/'n2880'
    for name,digest in s['parent_hashes'].items(): assert _hash_file(parent/name)==digest
    base=.5*(joblib.load(parent/'SVR.joblib').predict(x)+joblib.load(parent/'ExtraTrees.joblib').predict(x))
    experts={pool:predict_expert(root,pool,info,x) for pool,info in s['experts'].items()}
    probs={name:joblib.load(root/f'gate_{name}.joblib').predict_proba(x) for name in read(root/'gates.json')}
    selected=sorted(set(['parent',s['primary'],s['weak'],s['strong']]+list(s['profiles'].values())))
    preds={'parent':base,'prior':np.broadcast_to(s['center'],y.shape)}
    for name in selected:
        spec=s['specs'][name]
        if spec['kind']=='parent': continue
        prob=probs[spec['gate']]; prob=np.eye(3)[prob.argmax(1)] if spec['hard'] else prob
        pred=mix_experts(experts,prob,spec['profile']); preds[name]=base+spec['fraction']*(pred-base)
    for profile in PROFILES: preds['oracle/'+profile]=mix_experts(experts,np.eye(3)[labels],profile)
    for profile, name in s['profiles'].items():
        spec=s['specs'][name]; prob=probs[spec['gate']]
        prob=np.eye(3)[prob.argmax(1)] if spec['hard'] else prob
        preds['routed_only/'+profile]=mix_experts(experts,prob,profile)
        preds['oracle_matched/'+profile]=base+spec['fraction']*(preds['oracle/'+profile]-base)
    # Truth-conditioned diagnostics isolate expert quality; never operational selection.
    for pool, pred in experts.items(): preds['expert/'+pool]=pred
    geo=p['geometry']['distances']['5']; history=read(Path(p['parent'])/'selected_before_test.json')['rei']['history']
    rei=rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=history,circuit_repetitions=1)
    answered=np.isfinite(rei).all(1); preds['REI']=np.where(answered[:,None],rei,s['center'])
    frames=[]
    for name,pred in preds.items():
        f=rows[['event_uid','event','shot','geometry','propagation_law','epicenter_region','strength_band','epicenter_row','epicenter_col']].copy()
        f['model']=name; f['pred_x'],f['pred_y']=pred.T; f['error_mm']=np.linalg.norm(pred-y,axis=1); frames.append(f)
    frame=pd.concat(frames,ignore_index=True); frame.to_csv(root/'predictions.csv',index=False)
    metrics,pairs=[],[]
    for group,mask in group_masks(frame).items():
        sub=frame[mask]
        for name,f in sub.groupby('model'):
            e=f.error_mm; metrics.append(dict(group=group,model=name,events=f.event_uid.nunique(),mean_mm=float(e.mean()),p90_mm=float(e.quantile(.9)),within_1mm=float((e<=1).mean())))
        targets=sorted(set(selected+['oracle/'+profile for profile in PROFILES]+['expert/weak']))
        for name in targets:
            for reference in ('parent','prior'):
                a=sub[sub.model==name].groupby('event_uid').error_mm.mean(); b=sub[sub.model==reference].groupby('event_uid').error_mm.mean()
                assert a.index.equals(b.index); delta=(a-b).to_numpy(); rng=np.random.default_rng(SEED+20)
                boot=rng.choice(delta,(10000,len(delta))).mean(1)
                pairs.append(dict(group=group,model=name,reference=reference,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
        for profile in PROFILES:
            for name, reference in [('oracle/'+profile,'routed_only/'+profile),('oracle_matched/'+profile,s['profiles'][profile])]:
                a=sub[sub.model==name].groupby('event_uid').error_mm.mean(); b=sub[sub.model==reference].groupby('event_uid').error_mm.mean()
                assert a.index.equals(b.index); delta=(a-b).to_numpy(); rng=np.random.default_rng(SEED+20)
                boot=rng.choice(delta,(10000,len(delta))).mean(1)
                pairs.append(dict(group=group,model=name,reference=reference,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False); _write_json(root/'paired.json',pairs)
    gate_metrics={name:dict(log_loss=float(log_loss(labels,prob,labels=[0,1,2])),confusion=confusion_matrix(labels,prob.argmax(1),labels=[0,1,2]).tolist()) for name,prob in probs.items()}
    _write_json(root/'gate_test.json',gate_metrics)
    _write_json(root/'audit.json',dict(status='passed',events=EVENTS,model_hashes='passed',data_hashes='passed',seed_overlap=0,rei_answer_rate=float(answered.mean())))
    print(pd.DataFrame(metrics).query("group in ['ordinary','weak','strong_circle','strong_ellipse']").to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train','freeze','generate','evaluate']); parser.add_argument('root',type=Path); parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage in ('train','generate'): globals()[args.stage](args.root,args.workers)
    else: globals()[args.stage](args.root)
