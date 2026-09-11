"""Nested event learning curves with a newly generated validation/test split."""
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .adaptive_localization import segmented_features
from .dataset import iter_shards, _write_json
from .localization import _hash_file
from .temporal_localization import detector_series
from .temporal_gnn import TemporalGNN, graph_from_spec, infer

STRATA = ['geometry', 'propagation_law', 'epicenter_region', 'strength_band']


def nested_events(labels, sizes, seed=90501):
    """Balanced nested subsets; excludes the held-out strong ellipse conjunction."""
    ordinary = labels[~((labels.geometry=='elliptical') & (labels.strength_band==2))]
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(g.event_uid.to_numpy()) for _, g in ordinary.groupby(STRATA, sort=True)]
    if len(groups) != 30:
        raise ValueError('expected 30 ordinary strata')
    result = {}
    for size in sizes:
        if size % len(groups) or any(len(g)<size//len(groups) for g in groups):
            raise ValueError('insufficient balanced independent events')
        result[str(size)] = np.concatenate([g[:size//len(groups)] for g in groups]).tolist()
    return result


def fresh_split(labels, seed=90502):
    rng = np.random.default_rng(seed)
    val, test = [], []
    for _, g in labels.groupby(STRATA, sort=True):
        ids = rng.permutation(g.event_uid.to_numpy())
        if len(ids)%2:
            raise ValueError('each stratum must divide equally')
        val.extend(ids[:len(ids)//2]); test.extend(ids[len(ids)//2:])
    return val, test


def run(runs):
    runs = Path(runs)
    out = runs/'learning_curve_20260908/experiment'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    parent = runs/'model_comparison_20260908/experiment/model.json'
    info = json.loads(parent.read_text())
    spec = info['specification']
    sizes, seeds = [360,720,1440,1800], [41,42,43]
    sources = [runs/'window_calibration_20260908/dataset',
               runs/'temporal_diagnosis_20260908/fresh_dataset',
               runs/'adaptive_features_20260908/fresh_dataset',
               runs/'model_comparison_20260908/fresh_dataset']
    fresh = runs/'learning_curve_20260908/fresh_dataset'
    _write_json(out/'protocol.json', dict(sizes=sizes, seeds=seeds, sources=list(map(str,sources)),
        status='old tests explicitly promoted to training; new validation/test events',
        fixed_features=dict(cut=info['fixed_cut'], quiet=info['quiet']),
        selection='new ordinary validation mean error; no held-out ellipse selection',
        ridge_alpha=100, svr=dict(C=10, gamma=.001, epsilon=.1),
        gnn=dict(epochs=40, patience=8, lr=.001, batch=64, weight_decay=.001),
        parent_sha256=_hash_file(parent), source_sha256=_hash_file(Path(__file__))))

    def collect(paths):
        arrays, frames = [], []
        for path in paths:
            for data, labels in iter_shards(path):
                a = detector_series(data,spec)
                shots = len(a)//len(labels)
                rows = labels.iloc[np.repeat(np.arange(len(labels)),shots)].reset_index(drop=True)
                rows['shot'] = np.tile(np.arange(shots),len(labels))
                arrays.append(a); frames.append(rows)
        return np.concatenate(arrays), pd.concat(frames,ignore_index=True)

    raw, rows = collect(sources)
    new, new_rows = collect([fresh])
    old_labels = rows.drop_duplicates('event_uid')
    new_labels = new_rows.drop_duplicates('event_uid')
    assert len(old_labels)*2==len(rows)
    for field in ['event_uid','generation_seed','syndrome_seed']:
        assert not set(rows[field]) & set(new_rows[field])
    assert not rows.duplicated(['event_uid','shot']).any()
    subsets = nested_events(old_labels,sizes)
    val_ids, test_ids = fresh_split(new_labels)
    _write_json(out/'split.json',dict(training=subsets,validation=val_ids,test=test_ids,
                                     manifests={str(p):_hash_file(p/'dataset_manifest.json') for p in sources+[fresh]}))
    val = np.flatnonzero(new_rows.event_uid.isin(val_ids) & ~((new_rows.geometry=='elliptical') & (new_rows.strength_band==2)))
    # Fixed inherited feature construction; scaler and targets fitted on each training subset only.
    x = segmented_features(raw,np.full(len(raw),info['fixed_cut']),info['quiet'])
    vx = segmented_features(new[val],np.full(len(val),info['fixed_cut']),info['quiet'])
    y = rows[['epicenter_row','epicenter_col']].to_numpy()
    vy = new_rows[['epicenter_row','epicenter_col']].to_numpy()[val]
    adjacency, static, supports = graph_from_spec(spec)
    _write_json(out/'geometry.json',dict(specification=spec,adjacency=adjacency.tolist(),node_features=static.tolist(),supports=supports))
    scale = max(float(np.ptp(spec['geometry']['circuit_physical_coords_mm'],axis=0).max()/2),1e-3)
    fitted, logs = [], []
    for size in sizes:
        idx = np.flatnonzero(rows.event_uid.isin(subsets[str(size)]))
        assert len(idx)==size*2
        center = y[idx].mean(0)
        folder = out/f'n{size}'; folder.mkdir()
        fitted.append((size,'constant',-1,None,center))
        for family in ['ridge','svr']:
            estimator = Ridge(alpha=100) if family=='ridge' else MultiOutputRegressor(SVR(C=10,gamma=.001,epsilon=.1))
            model = make_pipeline(StandardScaler(),estimator).fit(x[idx],y[idx])
            error = float(np.linalg.norm(model.predict(vx)-vy,axis=1).mean())
            joblib.dump(model,folder/f'{family}.joblib')
            fitted.append((size,family,-1,model,center))
            logs.append(dict(size=size,family=family,seed=-1,validation_error_mm=error))
            print(size,family,'validation',error,flush=True)
        for seed in seeds:
            torch.manual_seed(seed)
            model = TemporalGNN(adjacency,static,True)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            rng = np.random.default_rng(seed)
            best, best_error, chosen, history = None,np.inf,0,[]
            begin = time.perf_counter()
            for epoch in range(1,41):
                model.train(); order=rng.permutation(idx)
                for start in range(0,len(order),64):
                    ii=order[start:start+64]; optimizer.zero_grad(set_to_none=True)
                    p=model(torch.tensor(raw[ii],dtype=torch.float32))
                    loss=nn.functional.mse_loss(p,torch.tensor((y[ii]-center)/scale,dtype=torch.float32))
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                error=float(np.linalg.norm(infer(model,new[val],center,scale)-vy,axis=1).mean())
                history.append(dict(epoch=epoch,error_mm=error))
                if error<best_error:
                    best,best_error,chosen=copy.deepcopy(model.state_dict()),error,epoch
                if epoch-chosen>=8: break
            model.load_state_dict(best); torch.save(best,folder/f'gnn_seed{seed}.pt')
            logs.append(dict(size=size,family='gnn',seed=seed,validation_error_mm=best_error,
                             epoch=chosen,history=history,center=center.tolist(),scale=scale,
                             seconds=time.perf_counter()-begin))
            fitted.append((size,'gnn',seed,model,center))
            print(size,'gnn',seed,'validation',best_error,'epoch',chosen,flush=True)
        _write_json(out/'training_progress.json',logs)
    scores=pd.DataFrame(logs).groupby(['size','family']).validation_error_mm.mean()
    selected=scores.idxmin()
    _write_json(out/'selected_before_test.json',dict(size=int(selected[0]),family=selected[1],
        validation=logs,hashes={str(p.relative_to(out)):_hash_file(p) for p in out.glob('n*/*')}))
    # First test prediction occurs only after every model and selection are frozen.
    test=np.flatnonzero(new_rows.event_uid.isin(test_ids))
    tx=segmented_features(new[test],np.full(len(test),info['fixed_cut']),info['quiet'])
    ty=new_rows[['epicenter_row','epicenter_col']].to_numpy()[test]
    frames=[]
    for size,family,seed,model,center in fitted:
        if family=='constant':
            p=np.broadcast_to(center,(len(test),2)).copy()
        elif family=='gnn':
            restored=TemporalGNN(adjacency,static,True)
            restored.load_state_dict(torch.load(out/f'n{size}'/f'gnn_seed{seed}.pt',weights_only=True))
            p=infer(restored,new[test],center,scale)
        else:
            p=joblib.load(out/f'n{size}'/f'{family}.joblib').predict(tx)
        f=new_rows.iloc[test][['event_uid','shot','strength_band','geometry','epicenter_region']].copy()
        f['size'],f['family'],f['seed']=size,family,seed
        f['split']=np.where((f.geometry=='elliptical')&(f.strength_band==2),'ood','id')
        f['true_x_mm'],f['true_y_mm']=ty.T
        f['predicted_x_mm'],f['predicted_y_mm']=p.T
        f['error_mm']=np.linalg.norm(p-ty,axis=1)
        frames.append(f)
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'predictions.csv',index=False)
    frame.groupby(['size','family','split','strength_band']).error_mm.mean().to_csv(out/'strength_metrics.csv')
    metrics=frame.groupby(['size','family','split']).error_mm.agg(['mean','median',lambda x:x.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    comparisons=[]
    for family in ['ridge','svr','gnn']:
        for split in ['id','ood']:
            sub=frame[(frame.family==family)&(frame.split==split)]
            base=sub[sub['size']==360].groupby('event_uid').error_mm.mean()
            for size in sizes[1:]:
                delta=(sub[sub['size']==size].groupby('event_uid').error_mm.mean()-base).to_numpy()
                rng=np.random.default_rng(90503)
                boot=[rng.choice(delta,len(delta)).mean() for _ in range(10000)]
                comparisons.append(dict(family=family,split=split,size=size,minus_n360_mm=float(delta.mean()),
                                        event_ci95_mm=np.quantile(boot,[.025,.975]).tolist()))
    _write_json(out/'paired_growth.json',comparisons)
    print(metrics.to_string(),flush=True)
    print('Selected:',selected,flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
