"""d5 event learning curves, nested old-train + new events, fresh holdouts."""
import argparse
import copy
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .api import run_simulation
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .distance_study import fields_to_pauli,geometry,ordinary,predict_nn,load as old_load,group_masks
from .fault_response import sample
from .localization import _hash_file
from .adaptive_localization import segmented_features
from .temporal_localization import TemporalCNN
from .temporal_diagnosis import rei_center

SEED=2026090909
SIZES=(720,1440,2880)
EVENTS=3456


def role(i):
    if not 0<=i<EVENTS:raise ValueError('invalid event index')
    return 'train' if i//36<72 else 'validation' if i//36<84 else 'test'


def prepare(root):
    root.mkdir(exist_ok=False);(root/'events').mkdir()
    parent=root.parent/'distance_study_20260909'
    inherited=json.loads((parent/'protocol.json').read_text())
    geometry_info=json.loads((parent/'geometry.json').read_text())
    frozen=json.loads((parent/'models_d5/frozen.json').read_text())
    p=dict(seed=SEED,events=EVENTS,distance=5,sizes=list(SIZES),parent=str(parent.resolve()),
        base=inherited['base'],sampling=inherited['sampling'],circuit=inherited['circuits']['5'],
        geometry=geometry_info,preprocessing=dict(cut=frozen['cut'],quiet=frozen['quiet']),
        parent_selected=frozen,settings=inherited,
        split='new 72 train cycles, 12 validation cycles, 12 test cycles; 36 balanced strata per cycle',
        nested='720 original ordinary training events + first 24/72 new train cycles for 1440/2880',
        exclusion='strong ellipses excluded from fitting and validation selection; all old validation/test excluded from training',
        primary='fixed old-selected SVR + ExtraTrees settings, coordinate mean; compare n2880 minus n720 on ordinary new test',
        secondary='equal validation search budgets per size: Ridge/SVR/ExtraTrees, blend, 3-seed CNN',
        test='fresh 432 physical events: 360 ordinary + 72 strong ellipses; weak 144; two shots/event',
        features='same inherited cut and independent quiet calibration for all sizes; scalers fitted only on training subset',
        source_sha256=_hash_file(Path(__file__)),parent_protocol_sha256=_hash_file(parent/'protocol.json'),
        parent_model_sha256=_hash_file(parent/'models_d5/frozen.json'),parent_data_sha256=_hash_file(parent/'progress.json'),
        response=json.loads((parent/'response_d5.json').read_text()),
        source_hashes={name:_hash_file(Path(__file__).with_name(name)) for name in ['simulator.py','stim_qec.py','fault_response.py','distance_study.py','dataset.py','api.py','adaptive_localization.py','temporal_localization.py']})
    _write_json(root/'protocol.json',p)


_WORKER=None


def initialize(root):
    global _WORKER
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    geometry(p['circuit'])
    union=np.asarray(p['geometry']['union_mm']);indices=np.asarray(p['geometry']['distances']['5']['union_indices'])
    _WORKER=root,p,union,indices


def event_job(i):
    root,p,union,indices=_WORKER;start=time.monotonic()
    c,labels=event_configuration(p['base'],p['sampling'],i,SEED,123)
    simulation=run_simulation(c,coords_mm=union)
    xyz=fields_to_pauli(simulation,p['circuit'],indices)
    bits,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(SEED+1,i,5)%(2**63-1),shots=2)
    assert bits.shape==(2,24,2047)
    row=simulation.parameters.iloc[0].to_dict()
    row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(labels,event=i,event_uid=f'd5-growth-{SEED}:{i}',role=role(i))
    path=root/'events'/f'{i:04d}.npz';tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,d5=bits)
    tmp.replace(path);_write_json(path.with_suffix('.json'),row)
    return dict(event=i,seconds=time.monotonic()-start)


def generate(root,workers=8):
    root=Path(root)
    if not root.exists():prepare(root)
    p=json.loads((root/'protocol.json').read_text())
    if p['source_sha256']!=_hash_file(Path(__file__)):raise ValueError('source changed')
    for name,digest in p['source_hashes'].items():
        if _hash_file(Path(__file__).with_name(name))!=digest:raise ValueError('dependency changed: '+name)
    with _generation_lock(root):
        done=[i for i in range(EVENTS) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),)) as pool:
            futures=[pool.submit(event_job,i) for i in range(EVENTS) if i not in done]
            for future in as_completed(futures):
                result=future.result();done.append(result['event'])
                if len(done)%72==0:
                    _write_json(root/'progress.json',dict(status='generating',completed=len(done),last=result))
                    print('Generated',len(done),'/',EVENTS,'last seconds',round(result['seconds'],1),flush=True)
        hashes={f'{i:04d}{suffix}':_hash_file(root/'events'/f'{i:04d}{suffix}') for i in range(EVENTS) for suffix in ['.npz','.json']}
        _write_json(root/'progress.json',dict(status='complete',completed=EVENTS,hashes=hashes))


def load_new(root,roles):
    status=json.loads((root/'progress.json').read_text())
    if status['status']!='complete':raise ValueError('incomplete dataset')
    raw,rows=[],[]
    for i in range(EVENTS):
        if role(i) not in roles:continue
        path=root/'events'/f'{i:04d}.npz'
        for suffix in ['.npz','.json']:
            if _hash_file(path.with_suffix(suffix))!=status['hashes'][f'{i:04d}{suffix}']:raise ValueError('corrupt data')
        row=json.loads(path.with_suffix('.json').read_text())
        if row['role']!=role(i) or row['event']!=i:raise ValueError('role mismatch')
        with np.load(path) as z:
            bits=z['d5'].copy()
            if bits.shape!=(2,24,2047):raise ValueError('invalid shape')
            raw.append(bits)
        rows.extend([dict(row,shot=shot,origin='new') for shot in range(2)])
    return np.concatenate(raw),pd.DataFrame(rows)


def training_indices(rows,size):
    if size not in SIZES:raise ValueError('unknown size')
    cycles={720:0,1440:24,2880:72}[size]
    return np.flatnonzero((rows.role=='train')&ordinary(rows)&((rows.origin=='old')|((rows.origin=='new')&(rows.event<cycles*36))))


def features(raw,preprocessing):
    """Bound cumulative-count workspace independently of training size."""
    return np.concatenate([segmented_features(raw[i:i+128],np.full(len(raw[i:i+128]),preprocessing['cut']),preprocessing['quiet'])
                           for i in range(0,len(raw),128)])


def train(root,size):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());parent=Path(p['parent']);out=root/f'n{size}'
    if (out/'frozen.json').exists():return
    out.mkdir(exist_ok=True);torch.set_num_threads(2)
    if _hash_file(parent/'progress.json')!=p['parent_data_sha256']:raise ValueError('old data changed')
    old,orows=old_load(parent,5,{'train'});orows['origin']='old'
    new,nrows=load_new(root,{'train','validation'})
    if set(orows.generation_seed)&set(nrows.generation_seed):raise ValueError('seed overlap')
    raw=np.concatenate([old,new]);rows=pd.concat([orows,nrows],ignore_index=True)
    del old,new
    idx=training_indices(rows,size);val=np.flatnonzero((rows.role=='validation')&ordinary(rows))
    assert len(idx)==2*size and len(val)==720
    assert rows.iloc[idx].event_uid.nunique()==size
    y=rows[['epicenter_row','epicenter_col']].to_numpy();center=y[idx].mean(0);scale=float(y[idx].std())
    x=features(raw,p['preprocessing'])
    score=lambda pred:float(np.linalg.norm(pred-y[val],axis=1).mean())
    settings=p['settings'];logs=[];selected={};vpred={}
    for family in ['Ridge','SVR','ExtraTrees']:
        if family=='Ridge':configs=[dict(alpha=a) for a in settings['ridge_alphas']]
        elif family=='SVR':configs=[dict(C=c,gamma=.001*57/169*g) for c in settings['svr_C'] for g in settings['svr_gamma_factors']]
        else:configs=[dict(leaf=l) for l in settings['et_leaf']]
        best=np.inf
        for config in configs:
            if family=='Ridge':model=make_pipeline(StandardScaler(),Ridge(**config))
            elif family=='SVR':model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(**config,epsilon=.1)))
            else:model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=config['leaf'],random_state=41,n_jobs=2)
            model.fit(x[idx],y[idx]);prediction=model.predict(x[val]);error=score(prediction)
            logs.append(dict(family=family,config=config,validation_mm=error))
            if config==p['parent_selected']['selected'][family]['config']:
                joblib.dump(model,out/f'fixed_{family}.joblib');vpred[f'fixed_{family}']=prediction
            if error<best:
                best=error;joblib.dump(model,out/f'{family}.joblib');vpred[family]=prediction
                selected[family]=dict(config=config,validation_mm=error)
        print('n',size,family,'validation',best,flush=True)
        _write_json(out/'progress.json',logs)
    vpred['fixed_blend']=.5*(vpred['fixed_SVR']+vpred['fixed_ExtraTrees'])
    vpred['blend']=.5*(vpred['SVR']+vpred['ExtraTrees'])
    vpred['prior']=np.broadcast_to(center,(len(val),2))
    nn=[]
    for seed in settings['seeds']:
        begin=time.monotonic();torch.manual_seed(seed);model=TemporalCNN(24)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        rng=np.random.default_rng(seed);best=np.inf;state=None;chosen=0;history=[]
        for epoch in range(1,31):
            model.train();order=rng.permutation(idx)
            for start in range(0,len(order),64):
                ii=order[start:start+64];optimizer.zero_grad(set_to_none=True)
                pred=model(torch.tensor(raw[ii],dtype=torch.float32))
                loss=torch.nn.functional.mse_loss(pred,torch.tensor((y[ii]-center)/scale,dtype=torch.float32))
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            error=score(predict_nn(model,raw[val],center,scale));history.append(dict(epoch=epoch,validation_mm=error))
            if error<best:best=error;state=copy.deepcopy(model.state_dict());chosen=epoch
            if epoch%5==0:print('n',size,'CNN',seed,'epoch',epoch,'best',round(best,4),flush=True)
            if epoch-chosen>=6:break
        model.load_state_dict(state);torch.save(state,out/f'CNN_{seed}.pt');nn.append(predict_nn(model,raw[val],center,scale))
        logs.append(dict(family='CNN',seed=seed,chosen=chosen,seconds=time.monotonic()-begin,history=history))
        _write_json(out/'progress.json',logs)
        print('n',size,'CNN seed',seed,'validation',best,'seconds',round(time.monotonic()-begin),flush=True)
    vpred['CNN']=np.mean(nn,axis=0)
    for name,pred in vpred.items():selected.setdefault(name,dict(validation_mm=score(pred)))
    candidates=['Ridge','SVR','ExtraTrees','blend','CNN']
    _write_json(out/'frozen.json',dict(size=size,train_event_uids=rows.iloc[idx].event_uid.drop_duplicates().tolist(),
        selected=selected,best_family=min(candidates,key=lambda k:selected[k]['validation_mm']),
        center=center.tolist(),scale=scale,seeds=settings['seeds'],preprocessing=p['preprocessing'],
        hashes={path.name:_hash_file(path) for path in out.iterdir() if path.suffix in ['.joblib','.pt']},
        protocol_sha256=_hash_file(root/'protocol.json'),source_sha256=_hash_file(Path(__file__))))


def freeze(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    infos={str(n):json.loads((root/f'n{n}/frozen.json').read_text()) for n in SIZES}
    for n in SIZES:
        for name,digest in infos[str(n)]['hashes'].items():
            if _hash_file(root/f'n{n}'/name)!=digest:raise ValueError('modified model')
    parent=Path(p['parent']);raw,rows=load_new(root,{'validation'});val=np.flatnonzero(ordinary(rows));raw=raw[val]
    y=rows.iloc[val][['epicenter_row','epicenter_col']].to_numpy();geo=p['geometry']['distances']['5'];center=np.asarray(infos['720']['center'])
    trials=[]
    for k in p['settings']['rei_histories']:
        pred=rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=k,circuit_repetitions=1)
        answered=np.isfinite(pred).all(1);pred=np.where(answered[:,None],pred,center)
        trials.append(dict(history=k,validation_mm=float(np.linalg.norm(pred-y,axis=1).mean()),answer_rate=float(answered.mean())))
    selection=dict(models=infos,rei=min(trials,key=lambda r:r['validation_mm']),rei_trials=trials,
        chosen=min([(n,f,info['validation_mm']) for n in SIZES for f,info in infos[str(n)]['selected'].items() if f in ['Ridge','SVR','ExtraTrees','blend','CNN']],key=lambda a:a[2]),
        primary='fixed_blend n2880 minus n720, ordinary',status='frozen before first new-test prediction')
    _write_json(root/'selected_before_test.json',selection)


def evaluate(root):
    root=Path(root);torch.set_num_threads(2)
    p=json.loads((root/'protocol.json').read_text());frozen=json.loads((root/'selected_before_test.json').read_text())
    raw,rows=load_new(root,{'test'});y=rows[['epicenter_row','epicenter_col']].to_numpy()
    x=features(raw,p['preprocessing'])
    frames=[]
    def record(size,family,pred,answered=True):
        frame=rows[['event_uid','event','shot','geometry','strength_band','epicenter_region','propagation_law']].copy()
        frame['size']=size;frame['family']=family;frame['error_mm']=np.linalg.norm(pred-y,axis=1)
        frame['true_x_mm'],frame['true_y_mm']=y.T;frame['predicted_x_mm'],frame['predicted_y_mm']=pred.T;frame['answered']=answered;frames.append(frame)
    for size in SIZES:
        out=root/f'n{size}';info=frozen['models'][str(size)]
        for name,digest in info['hashes'].items():
            if _hash_file(out/name)!=digest:raise ValueError('model checksum mismatch')
        predictions={name:joblib.load(out/f'{name}.joblib').predict(x) for name in ['Ridge','SVR','ExtraTrees','fixed_Ridge','fixed_SVR','fixed_ExtraTrees']}
        predictions['blend']=.5*(predictions['SVR']+predictions['ExtraTrees']);predictions['fixed_blend']=.5*(predictions['fixed_SVR']+predictions['fixed_ExtraTrees'])
        predictions['prior']=np.broadcast_to(info['center'],(len(raw),2));nn=[]
        for seed in info['seeds']:
            model=TemporalCNN(24);model.load_state_dict(torch.load(out/f'CNN_{seed}.pt',weights_only=True))
            nn.append(predict_nn(model,raw,np.asarray(info['center']),info['scale']))
        predictions['CNN']=np.mean(nn,axis=0)
        for family,pred in predictions.items():record(size,family,pred)
    parent=Path(p['parent'])/'models_d5';old=p['parent_selected']
    for name,digest in old['hashes'].items():
        if _hash_file(parent/name)!=digest:raise ValueError('old reference changed')
    pred=.5*(joblib.load(parent/'SVR.joblib').predict(x)+joblib.load(parent/'ExtraTrees.joblib').predict(x))
    record(720,'previous_blend',pred)
    geo=p['geometry']['distances']['5']
    pred=rei_center(raw,geo['sites_mm'],geo['physical_mm'],history_length=frozen['rei']['history'],circuit_repetitions=1)
    answered=np.isfinite(pred).all(1);pred=np.where(answered[:,None],pred,old['center']);record(0,'REI',pred,answered)
    f=pd.concat(frames,ignore_index=True);f.to_csv(root/'predictions.csv',index=False)
    metrics=[];pairs=[]
    for group,mask in group_masks(f).items():
        sub=f[mask]
        for (size,family),r in sub.groupby(['size','family']):
            e=r.error_mm;metrics.append(dict(size=int(size),family=family,group=group,events=r.event.nunique(),mean_mm=float(e.mean()),p90_mm=float(e.quantile(.9)),within_1mm=float((e<=1).mean()),within_2mm=float((e<=2).mean()),answer_rate=float(r.answered.mean())))
        def compare(a,b,name):
            av=a.groupby('event_uid').error_mm.mean();bv=b.groupby('event_uid').error_mm.mean()
            if not av.index.equals(bv.index):raise ValueError('unpaired')
            delta=(av-bv).to_numpy();rng=np.random.default_rng(SEED+4);boot=rng.choice(delta,(10000,len(delta))).mean(1)
            pairs.append(dict(comparison=name,group=group,difference_mm=float(delta.mean()),ci95_mm=np.quantile(boot,[.025,.975]).tolist()))
        for family in ['Ridge','SVR','ExtraTrees','blend','CNN','fixed_Ridge','fixed_SVR','fixed_ExtraTrees','fixed_blend']:
            for size in [1440,2880]:compare(sub[(sub.family==family)&(sub['size']==size)],sub[(sub.family==family)&(sub['size']==720)],f'{family}: n{size} minus n720')
        for size in SIZES:
            for family in ['blend','fixed_blend','CNN']:
                compare(sub[(sub.family==family)&(sub['size']==size)],sub[sub.family=='REI'],f'n{size} {family} minus REI')
                compare(sub[(sub.family==family)&(sub['size']==size)],sub[(sub.family=='prior')&(sub['size']==size)],f'n{size} {family} minus prior')
    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False);_write_json(root/'paired.json',pairs)
    print(pd.DataFrame(metrics).query("family in ['fixed_blend','blend','CNN','REI','previous_blend'] and group in ['ordinary','weak','strong_circle','strong_ellipse']").to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','train','freeze','evaluate']);parser.add_argument('root');parser.add_argument('--size',type=int,choices=SIZES,default=2880);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.stage=='generate':generate(args.root,args.workers)
    elif args.stage=='train':train(args.root,args.size)
    elif args.stage=='freeze':freeze(args.root)
    else:evaluate(args.root)
