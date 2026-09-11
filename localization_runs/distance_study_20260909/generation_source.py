"""Paired d3/d5/d7 experiment with fixed physical source region and pitch.

Generate once on the union of physical sites, then sample each circuit from
the same T1/T2 fields. Query inference uses detector bits only. Run stages
separately; freeze all models before scoring the untouched test events.
"""
import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
from .adaptive_localization import segmented_features
from .dataset import event_configuration, _seed, _write_json, _generation_lock
from .fault_response import build_response, layout_cache, sample
from .localization import _hash_file
from .stim_qec import (average_t1_over_rounds, round_tick_counts,
                      gate_slice_durations_ms, scheduled_gate_slice_pauli_probabilities)
from .temporal_diagnosis import rei_center
from .temporal_localization import TemporalCNN

DISTANCES = (3, 5, 7)
SEED = 2026090901


def role(event):
    cycle = event // 36
    return 'train' if cycle < 24 else 'validation' if cycle < 32 else 'test'


def geometry(circuit):
    layout, grid_sites, order, _ = layout_cache(json.dumps(circuit, sort_keys=True))
    indices = [np.flatnonzero(np.all(layout.grid_coords == site, axis=1))[0] for site in grid_sites]
    return layout, layout.physical_coords_mm[indices], order


def union_geometry(circuits):
    layouts = {d: geometry(circuits[str(d)])[0] for d in DISTANCES}
    union = np.unique(np.concatenate([np.round(v.physical_coords_mm, 10) for v in layouts.values()]), axis=0)
    indices = {d: np.array([np.flatnonzero(np.all(union == np.round(q, 10), axis=1))[0]
                           for q in layouts[d].physical_coords_mm]) for d in DISTANCES}
    return union, indices


def fields_to_pauli(simulation, circuit, indices, quiet=False):
    layout, _, _ = geometry(circuit)
    rounds = circuit['rounds']
    starts = circuit['start_time_ms'] + np.arange(rounds) * circuit['round_duration_ms']
    counts = round_tick_counts(layout.circuit, rounds)
    durations = gate_slice_durations_ms(circuit['round_duration_ms'], counts, circuit['gate_schedule'])
    averaged = []
    for key in ['t1_us', 't2_us']:
        field = simulation.physics[key][:, :, indices]
        if quiet:
            field = np.broadcast_to(field[:, :1], field.shape).copy()
        averaged.append(average_t1_over_rounds(field, simulation.time_ms, starts, circuit['round_duration_ms']))
    values = scheduled_gate_slice_pauli_probabilities(averaged[0], durations, counts, round_t2_us=averaged[1])
    return [values[k][0] for k in ['x', 'y', 'z']]


_WORKER = None


def initialize(root):
    global _WORKER
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    union, indices = union_geometry(protocol['circuits'])
    responses = {d: json.loads((root/f'response_d{d}.json').read_text()) for d in DISTANCES}
    _WORKER = root, protocol, union, indices, responses


def event_job(event):
    root, p, union, indices, responses = _WORKER
    begin = time.monotonic()
    config, sampled = event_configuration(p['base'], p['sampling'], event, SEED, 123)
    simulation = run_simulation(config, coords_mm=union)
    row = simulation.parameters.iloc[0].to_dict()
    row = {k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(sampled, event=event, event_uid=f'distance-{SEED}:{event}', role=role(event))
    payload = {}
    for d in DISTANCES:
        circuit = p['circuits'][str(d)]
        xyz = fields_to_pauli(simulation, circuit, indices[d])
        bits, _ = sample(responses[d], circuit, *xyz, seed=_seed(SEED+1,event,d)%(2**63-1), shots=2)
        assert bits.shape == (2, d*d-1, 2047)
        payload[f'd{d}'] = bits
    path = root/'events'/f'{event:04d}.npz'
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)
    _write_json(path.with_suffix('.json'), row)
    return dict(event=event, seconds=time.monotonic()-begin, sha256=_hash_file(path))


def prepare(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    (root/'events').mkdir()
    parent = root.parent/'window_calibration_20260908/dataset/dataset_manifest.json'
    inherited = json.loads(parent.read_text())
    base = copy.deepcopy(inherited['simulator_config'])
    base['epicenter']['reference_bounds_mm'] = [[-3.,3.],[-3.,3.]]
    circuits = {}
    for d in DISTANCES:
        c = copy.deepcopy(inherited['stim_config']); c['distance'] = d; c.pop('seed',None)
        circuits[str(d)] = c
    protocol = dict(base=base, sampling=inherited['sampling'], circuits=circuits,
        events=1440, seed=SEED, shots_per_event=2, split='24 train / 8 validation / 8 test cycles of 36 balanced strata',
        exclusion='strong elliptical conjunction excluded from training and validation selection',
        fixed_source_region_mm=[[-3,3],[-3,3]], region_labels='relative to d3 reference box, not each larger code',
        inference='detector events only; REI also receives fixed layout coordinates',
        paired='one physical simulation on union coordinates; circuits use identical fields at shared coordinates; independent circuit sampling seeds',
        scope='fixed pitch, expanding coverage; not fixed-area denser sensing; no-event and detection latency not evaluated',
        models=['prior','REI','Ridge','SVR','ExtraTrees','SVR_ET_blend','CNN'],
        ridge_alphas=[10.,100.,1000.], svr_C=[1.,10.,100.], svr_gamma_factors=[.1,1.,10.],
        et_leaf=[3,8,16], et_trees=256, seeds=[41,42,43],
        cnn=dict(epochs=30,patience=6,batch=64,lr=.001,weight_decay=.001),
        rei_histories=[16,64,256,1024,2047], rei_repetitions=1, rei_spatial_multiplier=2,
        calibration_shots=128, feature_cut='median training onset, fixed at inference',
        selection='ordinary validation mean Euclidean error; all distances frozen before test evaluation',
        bootstrap='10000 paired event resamples, descriptive 95% intervals, no multiplicity correction',
        source_sha256=_hash_file(Path(__file__)), parent_sha256=_hash_file(parent))
    _write_json(root/'protocol.json', protocol)
    for d in DISTANCES:
        print('Building validated fault response d',d,flush=True)
        _write_json(root/f'response_d{d}.json',build_response(circuits[str(d)]))
    union, indices = union_geometry(circuits)
    geom = {'union_mm': union.tolist(), 'distances': {}}
    for d in DISTANCES:
        layout,sites,_ = geometry(circuits[str(d)])
        geom['distances'][str(d)] = dict(sites_mm=sites.tolist(),physical_mm=layout.physical_coords_mm.tolist(),union_indices=indices[d].tolist())
    _write_json(root/'geometry.json',geom)
    # Separate no-radiation calibration, never derived from query labels/bits.
    config,_ = event_configuration(base,protocol['sampling'],1440,SEED+2,123)
    simulation = run_simulation(config,coords_mm=union)
    for d in DISTANCES:
        xyz = fields_to_pauli(simulation,circuits[str(d)],indices[d],quiet=True)
        response = json.loads((root/f'response_d{d}.json').read_text())
        totals = np.zeros(d*d-1)
        for batch in range(16):
            bits,_ = sample(response,circuits[str(d)],*xyz,seed=_seed(SEED+3,batch,d)%(2**63-1),shots=8)
            totals += bits.sum(axis=(0,2))
        _write_json(root/f'quiet_d{d}.json',dict(rates=(totals/(128*2047)).tolist(),shots=128))


def generate(root,workers):
    root = Path(root)
    if not root.exists(): prepare(root)
    p = json.loads((root/'protocol.json').read_text())
    if p['source_sha256'] != _hash_file(Path(__file__)):
        raise ValueError('generator source changed; use a new run')
    with _generation_lock(root):
        done = [i for i in range(p['events']) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize,initargs=(str(root),)) as pool:
            futures = [pool.submit(event_job,i) for i in range(p['events']) if i not in done]
            for future in as_completed(futures):
                record = future.result(); done.append(record['event'])
                if len(done)%36 == 0:
                    _write_json(root/'progress.json',dict(status='generating',completed=len(done),last=record))
                    print('Generated',len(done),'/',p['events'],'last seconds',round(record['seconds'],1),flush=True)
        _write_json(root/'progress.json',dict(status='complete',completed=len(done),hashes={f'{i:04d}':_hash_file(root/'events'/f'{i:04d}.npz') for i in done}))


def load(root,d,roles):
    records,arrays = [],[]
    progress=json.loads((root/'progress.json').read_text())
    for i in range(1440):
        if role(i) not in roles: continue
        path=root/'events'/f'{i:04d}.npz'
        if _hash_file(path)!=progress['hashes'][f'{i:04d}']:raise ValueError('data checksum mismatch')
        row=json.loads(path.with_suffix('.json').read_text())
        if row['event']!=i or row['role']!=role(i):raise ValueError('split mismatch')
        with np.load(path) as z: arrays.append(z[f'd{d}'].copy())
        records.extend([dict(row,shot=shot) for shot in range(2)])
    return np.concatenate(arrays),pd.DataFrame(records)


def ordinary(rows):
    return ~((rows.geometry=='elliptical')&(rows.strength_band==2))


def predict_nn(model, raw, center, scale):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(raw[i:i+64],dtype=torch.float32)).numpy() for i in range(0,len(raw),64)])*scale+center


def fit(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    torch.set_num_threads(2)
    geometry_info=json.loads((root/'geometry.json').read_text())
    selections={}
    for d in DISTANCES:
        out=root/f'models_d{d}'
        if (out/'frozen.json').exists():
            selections[str(d)]=json.loads((out/'frozen.json').read_text());continue
        out.mkdir(exist_ok=True)
        raw,rows=load(root,d,{'train','validation'})
        train=np.flatnonzero((rows.role=='train')&ordinary(rows));val=np.flatnonzero((rows.role=='validation')&ordinary(rows))
        y=rows[['epicenter_row','epicenter_col']].to_numpy()
        cut=int(round(rows.iloc[train].event_onset_ms.median()*1000))
        quiet=json.loads((root/f'quiet_d{d}.json').read_text())['rates']
        x=segmented_features(raw,np.full(len(raw),cut),quiet)
        center=y[train].mean(0);scale=float(y[train].std())
        geo=geometry_info['distances'][str(d)]
        logs=[]; selected={}
        def error(pred): return float(np.linalg.norm(pred-y[val],axis=1).mean())
        for family in ['Ridge','SVR','ExtraTrees']:
            trials=[]
            if family=='Ridge':
                candidates=[(dict(alpha=a),make_pipeline(StandardScaler(),Ridge(alpha=a))) for a in p['ridge_alphas']]
            elif family=='SVR':
                candidates=[(dict(C=c,gamma=.001*57/x.shape[1]*g),make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=c,gamma=.001*57/x.shape[1]*g,epsilon=.1)))) for c in p['svr_C'] for g in p['svr_gamma_factors']]
            else:
                candidates=[(dict(leaf=leaf),ExtraTreesRegressor(n_estimators=p['et_trees'],min_samples_leaf=leaf,random_state=41,n_jobs=2)) for leaf in p['et_leaf']]
            for config,model in candidates:
                model.fit(x[train],y[train]);score=error(model.predict(x[val]));trials.append((score,model,config))
                logs.append(dict(family=family,config=config,validation_mm=score))
            score,model,config=min(trials,key=lambda item:item[0]);joblib.dump(model,out/f'{family}.joblib')
            selected[family]=dict(config=config,validation_mm=score)
            print('d',d,family,score,flush=True)
        a=joblib.load(out/'SVR.joblib').predict(x[val]);b=joblib.load(out/'ExtraTrees.joblib').predict(x[val])
        selected['SVR_ET_blend']=dict(validation_mm=error(.5*(a+b)))
        histories=[]
        for k in p['rei_histories']:
            pred=rei_center(raw[val],geo['sites_mm'],geo['physical_mm'],circuit_repetitions=1,history_length=k)
            answered=np.isfinite(pred).all(1);pred=np.where(answered[:,None],pred,center)
            histories.append(dict(history=k,validation_mm=error(pred),answer_rate=float(answered.mean())))
        selected['REI']=min(histories,key=lambda item:item['validation_mm']);logs.extend([dict(family='REI',**r) for r in histories])
        selected['prior']=dict(validation_mm=error(np.broadcast_to(center,(len(val),2))))
        nn_val=[]
        for seed in p['seeds']:
            begin=time.monotonic();torch.manual_seed(seed);model=TemporalCNN(d*d-1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            rng=np.random.default_rng(seed);best=None;best_error=np.inf;chosen=0;history=[]
            for epoch in range(1,p['cnn']['epochs']+1):
                model.train();order=rng.permutation(train)
                for start in range(0,len(order),64):
                    idx=order[start:start+64];optimizer.zero_grad(set_to_none=True)
                    pred=model(torch.tensor(raw[idx],dtype=torch.float32))
                    loss=torch.nn.functional.mse_loss(pred,torch.tensor((y[idx]-center)/scale,dtype=torch.float32))
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                score=error(predict_nn(model,raw[val],center,scale));history.append(dict(epoch=epoch,validation_mm=score))
                if score<best_error:best=copy.deepcopy(model.state_dict());best_error=score;chosen=epoch
                if epoch-chosen>=p['cnn']['patience']:break
            model.load_state_dict(best);torch.save(best,out/f'CNN_{seed}.pt');nn_val.append(predict_nn(model,raw[val],center,scale))
            logs.append(dict(family='CNN',seed=seed,chosen_epoch=chosen,history=history,seconds=time.monotonic()-begin))
            print('d',d,'CNN seed',seed,'validation',best_error,'seconds',round(time.monotonic()-begin),flush=True)
        selected['CNN']=dict(validation_mm=error(np.mean(nn_val,axis=0)),seeds=p['seeds'],aggregation='coordinate ensemble')
        frozen=dict(distance=d,selected=selected,cut=cut,quiet=quiet,center=center.tolist(),scale=scale,
            train_events=rows.iloc[train].event.nunique(),validation_events=rows.iloc[val].event.nunique(),
            best_family=min(selected,key=lambda k:selected[k]['validation_mm']),
            hashes={f.name:_hash_file(f) for f in out.iterdir() if f.suffix in ('.pt','.joblib')})
        _write_json(out/'training.json',logs);_write_json(out/'frozen.json',frozen);selections[str(d)]=frozen
    _write_json(root/'selected_before_test.json',selections)


def group_masks(rows):
    return {'ordinary':ordinary(rows),'strong_ellipse':~ordinary(rows),
            'weak':rows.strength_band==0,'medium':rows.strength_band==1,'strong':rows.strength_band==2,
            'strong_circle':(rows.strength_band==2)&(rows.geometry=='circular'),
            'all':np.ones(len(rows),dtype=bool)}


def evaluate(root):
    root=Path(root);torch.set_num_threads(2)
    selections=json.loads((root/'selected_before_test.json').read_text())
    geom=json.loads((root/'geometry.json').read_text())['distances']
    frames=[];metrics=[]
    for d in DISTANCES:
        info=selections[str(d)];out=root/f'models_d{d}'
        for name,digest in info['hashes'].items():
            if _hash_file(out/name)!=digest:raise ValueError('frozen model changed')
        raw,rows=load(root,d,{'test'});y=rows[['epicenter_row','epicenter_col']].to_numpy()
        x=segmented_features(raw,np.full(len(raw),info['cut']),info['quiet'])
        predictions={name:joblib.load(out/f'{name}.joblib').predict(x) for name in ['Ridge','SVR','ExtraTrees']}
        predictions['SVR_ET_blend']=.5*(predictions['SVR']+predictions['ExtraTrees'])
        predictions['prior']=np.broadcast_to(info['center'],(len(raw),2))
        rei=rei_center(raw,geom[str(d)]['sites_mm'],geom[str(d)]['physical_mm'],circuit_repetitions=1,history_length=info['selected']['REI']['history'])
        answered=np.isfinite(rei).all(1);predictions['REI']=np.where(answered[:,None],rei,info['center'])
        nn=[]
        for seed in info['selected']['CNN']['seeds']:
            model=TemporalCNN(d*d-1);model.load_state_dict(torch.load(out/f'CNN_{seed}.pt',weights_only=True))
            nn.append(predict_nn(model,raw,np.asarray(info['center']),info['scale']))
        predictions['CNN']=np.mean(nn,axis=0)
        for family,pred in predictions.items():
            frame=rows[['event','event_uid','shot','geometry','strength_band','epicenter_region','propagation_law']].copy()
            frame['distance']=d;frame['family']=family;frame['error_mm']=np.linalg.norm(pred-y,axis=1)
            frame['true_x_mm'],frame['true_y_mm']=y.T;frame['predicted_x_mm'],frame['predicted_y_mm']=pred.T
            frame['answered']=answered if family=='REI' else True
            frames.append(frame)
            for name,mask in group_masks(frame).items():
                sub=frame[mask];e=sub.error_mm
                metrics.append(dict(distance=d,family=family,group=name,events=sub.event.nunique(),mean_mm=e.mean(),p90_mm=e.quantile(.9),
                    within_1mm=float((e<=1).mean()),within_2mm=float((e<=2).mean()),answer_rate=float(sub.answered.mean())))
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(root/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False)
    comparisons=[]
    def compare(a,b,description,group):
        av=a.groupby('event').error_mm.mean();bv=b.groupby('event').error_mm.mean()
        if not av.index.equals(bv.index):raise ValueError('unpaired events')
        delta=(av-bv).to_numpy();rng=np.random.default_rng(SEED+4)
        boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
        comparisons.append(dict(comparison=description,group=group,events=len(delta),difference_mm=float(delta.mean()),ci95_mm=np.quantile(boot,[.025,.975]).tolist()))
    for group,mask in group_masks(frame).items():
        sub=frame[mask]
        for family in selections['3']['selected']:
            for d in (5,7):
                compare(sub[(sub.family==family)&(sub.distance==d)],sub[(sub.family==family)&(sub.distance==3)],f'{family}: d{d} minus d3',group)
        for d in DISTANCES:
            for family in ['Ridge','SVR','ExtraTrees','SVR_ET_blend','CNN']:
                for ref in ['REI','prior']:
                    compare(sub[(sub.family==family)&(sub.distance==d)],sub[(sub.family==ref)&(sub.distance==d)],f'd{d}: {family} minus {ref}',group)
    _write_json(root/'paired.json',comparisons)
    print(pd.DataFrame(metrics).query("group in ['ordinary','weak','strong_circle','strong_ellipse']").to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','fit','evaluate']);parser.add_argument('root');parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.stage=='generate':generate(args.root,args.workers)
    elif args.stage=='fit':fit(args.root)
    else:evaluate(args.root)
