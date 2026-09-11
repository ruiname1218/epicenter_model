"""Weak-only paired acquisition, timing diagnostics and training-only physical templates.

Privileged T1, expected detector marginals and true onset are explicitly separate
from observable inference. Fresh test generation requires frozen selections.
"""
import argparse
import copy
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .api import run_simulation
from .adaptive_localization import segmented_features, estimated_cuts
from .dataset import event_configuration, _seed, _write_json, _generation_lock
from .distance_study import fields_to_pauli
from .fault_response import sample, layout_cache
from .localization import _hash_file
from .weak_observation import reduce_background

SEED=2026090931
NTRAIN,NVAL,NTEST=1728,288,576
CASES=('nom2_fixed','nom2_estimated','nom2_multi','nom2_oracle_time','nom4_fixed','quiet2_fixed','quiet4_fixed','oracle_t1','oracle_marginals')
CONDITIONS={'nom2':('nominal',2047),'nom4':('nominal',4095),'quiet2':('quiet10',2047),'quiet4':('quiet10',4095)}


def read(path): return json.loads(Path(path).read_text())


def role(i):
    if not 0<=i<NTRAIN+NVAL+NTEST: raise ValueError('invalid event index')
    return 'train' if i<NTRAIN else 'validation' if i<NTRAIN+NVAL else 'test'


def bin_means(raw,bins=16):
    edges=np.linspace(0,raw.shape[-1],bins+1,dtype=int)
    if np.any(np.diff(edges)<=0):raise ValueError('too many bins')
    return np.stack([raw[...,a:b].mean(-1) for a,b in zip(edges[:-1],edges[1:])],-1)


def segments(raw,cuts,quiet):
    return np.concatenate([segmented_features(raw[i:i+64],np.asarray(cuts)[i:i+64],quiet) for i in range(0,len(raw),64)])


def observed_features(raw,quiet,cut,kind):
    """No event truth argument is accepted by this operational feature interface."""
    if kind=='fixed': return segments(raw,np.full(len(raw),cut),quiet)
    if kind=='estimated': return segments(raw,estimated_cuts(raw),quiet)
    if kind=='multi': return np.concatenate([segments(raw,np.full(len(raw),c),quiet) for c in (64,192,320,448,640)],axis=1)
    raise ValueError('unknown observable feature kind')


def prepare(root):
    root.mkdir(exist_ok=False); (root/'events').mkdir()
    parent=root.parent/'distance_growth_20260909'; inherited=read(parent/'protocol.json')
    base=copy.deepcopy(inherited['base']); base['time']['end_ms']=4.100
    sampling=copy.deepcopy(inherited['sampling']); sampling['generation_bands_per_us']=[sampling['generation_bands_per_us'][0]]
    circuit=copy.deepcopy(inherited['circuit']); circuit['rounds']=4096
    configs={}
    for name,factor in [('nominal',1.),('quiet10',.1)]:
        c=copy.deepcopy(circuit); c['circuit_noise']={k:v*factor for k,v in circuit['circuit_noise'].items()}; configs[name]=c
    p=dict(seed=SEED,base=base,sampling=sampling,circuits=configs,response=inherited['response'],geometry=inherited['geometry'],
        parent=str(parent.resolve()),preprocessing=inherited['preprocessing'],cases=list(CASES),
        train_events=NTRAIN,validation_events=NVAL,test_events=NTEST,shots=2,calibration_shots=256,
        split='144/24/48 cycles of 12 weak-only strata; paired conditions and shots kept in one split.',
        primary='Validation-selected observable nominal 2 ms model minus frozen parent on fresh weak test; must also beat validation-selected fixed-position reference to claim location information.',
        secondary='Nominal4ms, hypothetical background1/10 at2/4ms, and privileged true onset/noiseless T1/exact expected marginals. No test retuning.',
        inference='Operational model receives one shot of detector bits only; physics-template bank uses TRAIN physical marginals, not query truth.',
        physical_diagnostics='T1 excess-rate means at only the 49 d5 physical qubits over first2.048ms; detector expected marginals first2047 rounds; no epicenter/onset columns in these feature vectors.',
        templates='16-bin training conditional detector means; Bernoulli composite log score or shrinkage-spatial GLS score, temperature .25/1/4. Not exact joint likelihood.',
        search='9 feature cases, each Ridge alpha100/1000, SVR C1 gamma .57/d or C10 gamma .057/d, ET leaf16/64; 54 regressions. Plus 24 training-template scoring settings.',
        caveats='All events known weak/present; not a deployable strength detector or full-distribution result. Quiet10 changes hardware/background, not software-only improvement. 4ms costs longer observation.',
        bootstrap='10000 paired physical-event resamples; descriptive95% intervals, no multiplicity/training-resampling correction.',
        source_hash=_hash_file(Path(__file__)),parent_protocol_hash=_hash_file(parent/'protocol.json'))
    _write_json(root/'protocol.json',p)


_WORKER=None


def initialize(root):
    global _WORKER
    root=Path(root); p=read(root/'protocol.json')
    for config in p['circuits'].values():layout_cache(json.dumps(config,sort_keys=True))
    _WORKER=root,p


def fields(simulation,p,condition):
    sim=simulation
    if condition=='quiet10':
        sim=copy.copy(simulation); sim.physics=dict(simulation.physics)
        for key in ('t1_us','t2_us'):sim.physics[key]=reduce_background(simulation.physics[key],.1)
    return fields_to_pauli(sim,p['circuits'][condition],np.asarray(p['geometry']['distances']['5']['union_indices']))


def event_job(i):
    root,p=_WORKER; config,labels=event_configuration(p['base'],p['sampling'],i,SEED,123)
    sim=run_simulation(config,coords_mm=np.asarray(p['geometry']['union_mm']))
    payload={}
    for condition in ('nominal','quiet10'):
        bits,expected=sample(p['response'],p['circuits'][condition],*fields(sim,p,condition),seed=_seed(SEED+1,i,0 if condition=='nominal' else 1)%(2**63-1),shots=2)
        payload[condition]=bits
        # Physical expectations are stored separately and are NEVER observable query input.
        for prefix,(cond,ticks) in CONDITIONS.items():
            if cond==condition: payload['latent_'+prefix]=bin_means(expected[:,:ticks])
    indices=np.asarray(p['geometry']['distances']['5']['union_indices'])
    t1=sim.physics['t1_us'][0][:,indices].astype(np.float64)
    excess=1/t1-1/t1[:1]
    keep=np.asarray(sim.time_ms)<=2.048
    payload['latent_t1']=bin_means(excess[keep].T,4).reshape(-1)
    row=sim.parameters.iloc[0].to_dict(); row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(labels,event=i,event_uid=f'weak-info-{SEED}:{i}',role=role(i))
    path=root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**payload)
    path.with_suffix('.npz.tmp').replace(path); _write_json(path.with_suffix('.json'),row)
    return i


def generate(root,test,workers):
    p=read(root/'protocol.json')
    if test:
        s=read(root/'selection.json'); assert s['source_hash']==_hash_file(Path(__file__))
    ids=list(range(NTRAIN+NVAL,NTRAIN+NVAL+NTEST) if test else range(NTRAIN+NVAL))
    with _generation_lock(root):
        done=[i for i in ids if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),)) as executor:
            futures=[executor.submit(event_job,i) for i in ids if i not in done]
            for f in as_completed(futures):
                done.append(f.result())
                if len(done)%144==0:print('Generated',len(done),'/',len(ids),'test',test,flush=True)
        _write_json(root/('test_manifest.json' if test else 'fit_manifest.json'),dict(events=len(ids),selection_hash=_hash_file(root/'selection.json') if test else None,
            hashes={f'{i:04d}{ext}':_hash_file(root/'events'/f'{i:04d}{ext}') for i in ids for ext in ('.npz','.json')}))


def calibrate(root):
    p=read(root/'protocol.json'); config,_=event_configuration(p['base'],p['sampling'],0,SEED+10,123)
    sim=run_simulation(config,coords_mm=np.asarray(p['geometry']['union_mm']))
    quiet_sim=copy.copy(sim);quiet_sim.physics=dict(sim.physics)
    for key in ('t1_us','t2_us'):quiet_sim.physics[key]=np.broadcast_to(sim.physics[key][:,:1],sim.physics[key].shape).copy()
    calibration={}
    for condition in ('nominal','quiet10'):
        xyz=fields(quiet_sim,p,condition); batches=[]
        for i in range(16):
            bits,_=sample(p['response'],p['circuits'][condition],*xyz,seed=_seed(SEED+11,i,0 if condition=='nominal' else 1)%(2**63-1),shots=16);batches.append(bits)
        bits=np.concatenate(batches); np.savez_compressed(root/f'calibration_{condition}.npz',bits=bits)
        for prefix,(cond,ticks) in CONDITIONS.items():
            if condition!=cond:continue
            q=bits[:,:,:ticks]; mean=bin_means(q); centered=mean-mean.mean(0)
            vectors=centered.transpose(0,2,1).reshape(-1,24); cov=np.cov(vectors,rowvar=False)
            cov=.9*cov+.1*np.diag(np.diag(cov))+np.eye(24)*1e-10
            vals,vecs=np.linalg.eigh(cov); white=(vecs/np.sqrt(vals))@vecs.T
            calibration[prefix]=dict(quiet=q.mean((0,2)),mean=mean.mean(0),white=white)
    joblib.dump(calibration,root/'calibration.joblib')


def read_events(root,test):
    manifest=read(root/('test_manifest.json' if test else 'fit_manifest.json'))
    ids=range(NTRAIN+NVAL,NTRAIN+NVAL+NTEST) if test else range(NTRAIN+NVAL)
    payload={}; rows=[]
    for i in ids:
        path=root/'events'/f'{i:04d}.npz'
        for ext in ('.npz','.json'):assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:
            for name in z.files:payload.setdefault(name,[]).append(z[name])
        row=read(path.with_suffix('.json')); assert row['role']==role(i)
        rows.extend([dict(row,shot=j) for j in range(2)])
    return {name:np.concatenate(values) if not name.startswith('latent_') else np.stack(values) for name,values in payload.items()},pd.DataFrame(rows)


def make_features(payload,rows,p,cal):
    nominal=payload['nominal'][:,:,:2047]; cut=p['preprocessing']['cut']; q=cal['nom2']['quiet']
    xs={f'nom2_{kind}':observed_features(nominal,q,cut,kind) for kind in ('fixed','estimated','multi')}
    # Round t starts at (t+1)*1us; no true onset input to any other feature case.
    times=np.arange(1,2048)*p['circuits']['nominal']['round_duration_ms']+p['circuits']['nominal']['start_time_ms']
    cuts=np.searchsorted(times,rows.event_onset_ms.to_numpy())
    xs['nom2_oracle_time']=segments(nominal,cuts,q)
    for prefix in ('nom4','quiet2','quiet4'):
        condition,ticks=CONDITIONS[prefix];xs[prefix+'_fixed']=observed_features(payload[condition][:,:,:ticks],cal[prefix]['quiet'],cut,'fixed')
    xs['oracle_t1']=np.repeat(payload['latent_t1'],2,axis=0)
    xs['oracle_marginals']=np.repeat(payload['latent_nom2'].reshape(len(rows)//2,-1),2,axis=0)
    return xs


def cache(root):
    if not (root/'calibration.joblib').exists():calibrate(root)
    p=read(root/'protocol.json'); payload,rows=read_events(root,False); cal=joblib.load(root/'calibration.joblib')
    xs=make_features(payload,rows,p,cal)
    for name,x in xs.items():np.save(root/f'x_{name}.npy',x)
    for prefix,(condition,ticks) in CONDITIONS.items():
        np.save(root/f'counts_{prefix}.npy',bin_means(payload[condition][:,:,:ticks]))
        np.save(root/f'bank_{prefix}.npy',payload['latent_'+prefix][:NTRAIN])
    rows.to_csv(root/'rows.csv',index=False)
    # Exact old preprocessing for the frozen parent, not the new calibration.
    np.save(root/'x_parent.npy',observed_features(payload['nominal'][:,:,:2047],p['preprocessing']['quiet'],p['preprocessing']['cut'],'fixed'))
    _write_json(root/'cache.json',dict(dimensions={k:v.shape[1] for k,v in xs.items()},
        hashes={f.name:_hash_file(f) for f in root.iterdir() if f.suffix in ('.npy','.csv','.joblib','.npz')}))


def fit_case(args):
    root,case=args;out=root/case;out.mkdir(exist_ok=True)
    if (out/'frozen.json').exists():return case
    rows=pd.read_csv(root/'rows.csv',float_precision='round_trip'); x=np.load(root/f'x_{case}.npy'); y=rows[['epicenter_row','epicenter_col']].to_numpy()
    tr=np.flatnonzero(rows.role=='train');va=np.flatnonzero(rows.role=='validation')
    configs=[('R100','r',{'alpha':100}),('R1000','r',{'alpha':1000}),('S1','s',{'C':1,'gamma':.57/x.shape[1]}),('S10','s',{'C':10,'gamma':.057/x.shape[1]}),('E16','e',{'min_samples_leaf':16}),('E64','e',{'min_samples_leaf':64})]
    preds,logs={},{}
    for name,family,config in configs:
        if family=='r':m=make_pipeline(StandardScaler(),Ridge(**config))
        elif family=='s':m=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(**config,epsilon=.1)))
        else:m=ExtraTreesRegressor(n_estimators=256,n_jobs=2,random_state=41,**config)
        m.fit(x[tr],y[tr]);preds[name]=m.predict(x[va]);logs[name]=float(np.linalg.norm(preds[name]-y[va],axis=1).mean())
        joblib.dump(m,out/f'{name}.joblib');print(case,name,logs[name],flush=True)
    np.savez_compressed(out/'validation.npz',**preds);_write_json(out/'frozen.json',dict(selected=min(logs,key=logs.get),scores=logs))
    return case


def template_scores(observed,bank,cal,ticks,mode):
    """Bank from TRAIN ONLY. Query argument consists of observed binned bits."""
    if mode=='composite':
        lengths=np.diff(np.linspace(0,ticks,17,dtype=int))
        prob=np.clip(bank,1e-7,1-1e-7);counts=observed*lengths
        scores=counts.reshape(len(observed),-1)@np.log(prob/(1-prob)).reshape(len(bank),-1).T
        scores+=np.sum(np.log1p(-prob)*lengths,axis=(1,2))[None,:]
    elif mode=='gls':
        obs=np.einsum('ij,njb->nib',cal['white'],observed-cal['mean']).reshape(len(observed),-1)
        means=np.einsum('ij,njb->nib',cal['white'],bank-cal['mean']).reshape(len(bank),-1)
        scores=obs@means.T-.5*np.square(means).sum(1)[None,:]
    else:raise ValueError('unknown template score')
    return scores-scores.max(1,keepdims=True)


def train(root,workers):
    if not (root/'cache.json').exists():cache(root)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for case in executor.map(fit_case,[(root,c) for c in CASES]):print('Finished',case,flush=True)


def previous_weak_reference(root,x):
    from .specialist_study import predict_expert,mix_experts
    old=root.parent/'specialists_20260909';s=read(old/'selection.json');p=read(old/'protocol.json')
    for name,digest in s['hashes'].items():assert _hash_file(old/name)==digest
    parent=Path(p['parent'])/'n2880'
    for name,digest in s['parent_hashes'].items():assert _hash_file(parent/name)==digest
    base=.5*(joblib.load(parent/'SVR.joblib').predict(x)+joblib.load(parent/'ExtraTrees.joblib').predict(x))
    spec=s['specs'][s['weak']]
    if spec['kind']=='parent':return base
    experts={pool:predict_expert(old,pool,info,x) for pool,info in s['experts'].items()}
    prob=joblib.load(old/f'gate_{spec["gate"]}.joblib').predict_proba(x)
    if spec['hard']:prob=np.eye(3)[prob.argmax(1)]
    return base+spec['fraction']*(mix_experts(experts,prob,spec['profile'])-base)


def freeze(root):
    assert not (root/'selection.json').exists()
    p=read(root/'protocol.json');rows=pd.read_csv(root/'rows.csv',float_precision='round_trip');y=rows[['epicenter_row','epicenter_col']].to_numpy();tr=rows.role=='train';va=rows.role=='validation'
    centers={'train_mean':y[tr].mean(0),'geometry_center':np.array([0.,0.])}
    predictions={name:np.broadcast_to(center,(int(va.sum()),2)) for name,center in centers.items()};specs={name:dict(kind='constant',center=center.tolist()) for name,center in centers.items()}
    parent=Path(p['parent'])/'n2880';xp=np.load(root/'x_parent.npy')[va]
    predictions['parent']=.5*(joblib.load(parent/'SVR.joblib').predict(xp)+joblib.load(parent/'ExtraTrees.joblib').predict(xp));specs['parent']=dict(kind='parent')
    predictions['previous_weak_routed']=previous_weak_reference(root,xp);specs['previous_weak_routed']=dict(kind='previous_weak')
    for case in CASES:
        info=read(root/case/'frozen.json');name=case+'/'+info['selected']
        with np.load(root/case/'validation.npz') as z:predictions[name]=z[info['selected']]
        specs[name]=dict(kind='regressor',case=case,model=info['selected'])
    cal=joblib.load(root/'calibration.joblib');coords=y[tr][::2]
    for prefix,(_,ticks) in CONDITIONS.items():
        observed=np.load(root/f'counts_{prefix}.npy')[va];bank=np.load(root/f'bank_{prefix}.npy')
        for mode in ('composite','gls'):
            score=template_scores(observed,bank,cal[prefix],ticks,mode)
            for temp in (.25,1.,4.):
                name=f'{prefix}_template/{mode}/{temp}';predictions[name]=softmax(score/temp,axis=1)@coords
                specs[name]=dict(kind='template',prefix=prefix,mode=mode,temperature=temp)
    scores={k:float(np.linalg.norm(v-y[va],axis=1).mean()) for k,v in predictions.items()}
    fixed=min(centers,key=scores.get)
    groups={prefix:[fixed]+(['parent','previous_weak_routed'] if prefix=='nom2' else [])+[k for k in scores if k.startswith(prefix) and 'oracle' not in k] for prefix in CONDITIONS}
    chosen={prefix:min(candidates,key=scores.get) for prefix,candidates in groups.items()}
    diagnostics={case:next(k for k in specs if specs[k].get('case')==case) for case in ('nom2_oracle_time','oracle_t1','oracle_marginals')}
    s=dict(chosen=chosen,diagnostics=diagnostics,fixed=fixed,scores=scores,specs=specs,
        training_coordinates=coords.tolist(),source_hash=_hash_file(Path(__file__)),status='frozen before new test generation',protocol_hash=_hash_file(root/'protocol.json'),
        hashes={str(f.relative_to(root)):_hash_file(f) for f in root.rglob('*.joblib')},
        cache_hashes=read(root/'cache.json')['hashes'],parent_hashes={name:_hash_file(parent/name) for name in ('SVR.joblib','ExtraTrees.joblib')})
    s['previous_weak_selection_hash']=_hash_file(root.parent/'specialists_20260909/selection.json')
    _write_json(root/'selection.json',s);pd.DataFrame([dict(candidate=k,mean_mm=v) for k,v in scores.items()]).sort_values('mean_mm').to_csv(root/'validation_scores.csv',index=False)
    print('CHOSEN',chosen,'SCORES',{k:scores[v] for k,v in chosen.items()},'DIAGNOSTICS',{k:scores[v] for k,v in diagnostics.items()},'PARENT',scores['parent'],'FIXED',scores[fixed],flush=True)


def evaluate(root):
    p=read(root/'protocol.json');s=read(root/'selection.json');manifest=read(root/'test_manifest.json')
    assert s['source_hash']==_hash_file(Path(__file__)) and manifest['selection_hash']==_hash_file(root/'selection.json')
    for name,digest in s['hashes'].items():assert _hash_file(root/name)==digest
    for name,digest in s['cache_hashes'].items():assert _hash_file(root/name)==digest
    payload,rows=read_events(root,True);cal=joblib.load(root/'calibration.joblib');xs=make_features(payload,rows,p,cal);y=rows[['epicenter_row','epicenter_col']].to_numpy()
    parent=Path(p['parent'])/'n2880'
    for name,digest in s['parent_hashes'].items():assert _hash_file(parent/name)==digest
    xp=observed_features(payload['nominal'][:,:,:2047],p['preprocessing']['quiet'],p['preprocessing']['cut'],'fixed')
    preds={'parent':.5*(joblib.load(parent/'SVR.joblib').predict(xp)+joblib.load(parent/'ExtraTrees.joblib').predict(xp))}
    assert _hash_file(root.parent/'specialists_20260909/selection.json')==s['previous_weak_selection_hash']
    preds['previous_weak_routed']=previous_weak_reference(root,xp)
    # Score every validation-frozen per-case regressor and declared template, never rank on test.
    for name,spec in s['specs'].items():
        if spec['kind']=='constant':preds[name]=np.broadcast_to(spec['center'],y.shape)
        elif spec['kind']=='regressor':preds[name]=joblib.load(root/spec['case']/f'{spec["model"]}.joblib').predict(xs[spec['case']])
    for prefix,(condition,ticks) in CONDITIONS.items():
        observed=bin_means(payload[condition][:,:,:ticks]);bank=np.load(root/f'bank_{prefix}.npy')
        for mode in ('composite','gls'):
            score=template_scores(observed,bank,cal[prefix],ticks,mode)
            for temp in (.25,1.,4.):preds[f'{prefix}_template/{mode}/{temp}']=softmax(score/temp,axis=1)@np.asarray(s['training_coordinates'])
    frames=[]
    for name,pred in preds.items():
        f=rows[['event_uid','event','shot','geometry','propagation_law','epicenter_region','epicenter_row','epicenter_col','qp_generation_scale_per_us']].copy()
        f['model']=name;f['error_mm']=np.linalg.norm(pred-y,axis=1);f['pred_x'],f['pred_y']=pred.T;frames.append(f)
    f=pd.concat(frames,ignore_index=True);f.to_csv(root/'predictions.csv',index=False)
    metrics=[]
    for name,r in f.groupby('model'):
        e=r.error_mm;metrics.append(dict(model=name,events=NTEST,mean_mm=float(e.mean()),p90_mm=float(e.quantile(.9)),within_1mm=float((e<=1).mean())))
    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False)
    pairs=[];targets=sorted(set(s['chosen'].values())|set(s['diagnostics'].values()))
    same_training_baseline=next(name for name,spec in s['specs'].items() if spec.get('case')=='nom2_fixed')
    references=sorted(set(['parent','previous_weak_routed',s['fixed'],s['chosen']['nom2'],same_training_baseline]))
    for name in targets:
        for reference in references:
            a=f[f.model==name].groupby('event_uid').error_mm.mean();b=f[f.model==reference].groupby('event_uid').error_mm.mean();assert a.index.equals(b.index)
            delta=(a-b).to_numpy();rng=np.random.default_rng(SEED+20);boot=rng.choice(delta,(10000,len(delta))).mean(1)
            pairs.append(dict(model=name,reference=reference,difference_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    _write_json(root/'paired.json',pairs)
    print(pd.DataFrame(metrics)[lambda z:z.model.isin(targets+references)].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate_fit','train','freeze','generate_test','evaluate']);parser.add_argument('root',type=Path);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.stage.startswith('generate'):generate(args.root,args.stage=='generate_test',args.workers)
    elif args.stage=='train':train(args.root,args.workers)
    else:globals()[args.stage](args.root)
