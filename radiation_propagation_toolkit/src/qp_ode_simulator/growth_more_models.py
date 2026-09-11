"""Additional model learning curves on the existing, now exploratory holdout."""
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from .adaptive_localization import segmented_features
from .dataset import iter_shards, _write_json
from .localization import _hash_file
from .temporal_localization import TemporalCNN, detector_series
from .temporal_gnn import infer


def tree_model(family, setting, seed):
    if family == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=128, min_samples_leaf=setting,
                                   n_jobs=2, random_state=seed)
    if family == 'hist_gradient_boosting':
        # No internal random shot split: external event validation only.
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=setting, learning_rate=.05,
            l2_regularization=10., early_stopping=False, random_state=seed))
    raise ValueError('unknown tree family')


def paired_delta(candidate, reference):
    keys = ['event_uid', 'shot']
    a = candidate.groupby(keys).error_mm.mean()
    b = reference.groupby(keys).error_mm.mean()
    if not a.index.equals(b.index):
        raise ValueError('paired comparison sample identity mismatch')
    delta = (a-b).groupby('event_uid').mean().to_numpy()
    rng = np.random.default_rng(90504)
    boot = [rng.choice(delta, len(delta)).mean() for _ in range(10000)]
    return dict(difference_mm=float(delta.mean()), ci95_mm=np.quantile(boot,[.025,.975]).tolist())


def run(runs):
    runs = Path(runs).resolve()
    previous = runs/'learning_curve_20260908/experiment'
    out = runs/'growth_more_models_20260908'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    old_protocol = json.loads((previous/'protocol.json').read_text())
    split = json.loads((previous/'split.json').read_text())
    spec = json.loads((previous/'geometry.json').read_text())['specification']
    sizes, seeds = [360,720,1440,1800], [41,42,43]
    protocol = dict(status='additional exploratory evaluation; holdout previously inspected',
        sizes=sizes, seeds=seeds, extra_trees_leaf=[4,12], boosting_leaf=[8,16],
        cnn=dict(epochs=40,patience=8,lr=.001,weight_decay=.001,batch=64),
        selection='ordinary validation mean single-model error; per family and training size',
        source_sha256=_hash_file(Path(__file__)), previous_split_sha256=_hash_file(previous/'split.json'),
        fixed_features=old_protocol['fixed_features'], specification=spec)
    _write_json(out/'protocol.json',protocol)
    for path, digest in split['manifests'].items():
        if _hash_file(Path(path)/'dataset_manifest.json') != digest:
            raise ValueError('dataset manifest changed')

    def collect(paths):
        arrays, frames = [], []
        for path in paths:
            for data, labels in iter_shards(Path(path)):
                a = detector_series(data,spec)
                shots = len(a)//len(labels)
                frame = labels.iloc[np.repeat(np.arange(len(labels)),shots)].reset_index(drop=True)
                frame['shot'] = np.tile(np.arange(shots),len(labels))
                arrays.append(a); frames.append(frame)
        return np.concatenate(arrays),pd.concat(frames,ignore_index=True)

    raw, rows = collect(old_protocol['sources'])
    new, new_rows = collect([runs/'learning_curve_20260908/fresh_dataset'])
    for field in ['event_uid','generation_seed','syndrome_seed']:
        if set(rows[field]) & set(new_rows[field]):
            raise ValueError('training/evaluation overlap')
    assert not set(split['validation']) & set(split['test'])
    val = np.flatnonzero(new_rows.event_uid.isin(split['validation']) &
                         ~((new_rows.geometry=='elliptical') & (new_rows.strength_band==2)))
    y = rows[['epicenter_row','epicenter_col']].to_numpy()
    vy = new_rows[['epicenter_row','epicenter_col']].to_numpy()[val]
    fixed = protocol['fixed_features']
    x = segmented_features(raw,np.full(len(raw),fixed['cut']),fixed['quiet'])
    vx = segmented_features(new[val],np.full(len(val),fixed['cut']),fixed['quiet'])
    scale = max(float(np.ptp(spec['geometry']['circuit_physical_coords_mm'],axis=0).max()/2),1e-3)
    logs, models = [], []
    for size in sizes:
        idx = np.flatnonzero(rows.event_uid.isin(split['training'][str(size)]))
        assert len(idx)==2*size
        center = y[idx].mean(0)
        folder=out/f'n{size}';folder.mkdir()
        for family, settings in [('extra_trees',[4,12]),('hist_gradient_boosting',[8,16])]:
            for setting in settings:
                for seed in (seeds if family=='extra_trees' else [41]):
                    begin=time.perf_counter()
                    model=tree_model(family,setting,seed).fit(x[idx],y[idx])
                    error=float(np.linalg.norm(model.predict(vx)-vy,axis=1).mean())
                    path=folder/f'{family}_{setting}_seed{seed}.joblib'
                    joblib.dump(model,path)
                    logs.append(dict(size=size,family=family,setting=setting,seed=seed,
                        validation_error_mm=error,seconds=time.perf_counter()-begin))
                    models.append((size,family,setting,seed,path,center))
                print(size,family,setting,'trained',flush=True)
        for seed in seeds:
            torch.manual_seed(seed);model=TemporalCNN(raw.shape[1])
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            rng=np.random.default_rng(seed)
            best,best_error,chosen,history=None,np.inf,0,[]
            begin=time.perf_counter()
            for epoch in range(1,41):
                model.train();order=rng.permutation(idx)
                for start in range(0,len(order),64):
                    ii=order[start:start+64];optimizer.zero_grad(set_to_none=True)
                    p=model(torch.tensor(raw[ii],dtype=torch.float32))
                    loss=nn.functional.mse_loss(p,torch.tensor((y[ii]-center)/scale,dtype=torch.float32))
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                error=float(np.linalg.norm(infer(model,new[val],center,scale)-vy,axis=1).mean())
                history.append(dict(epoch=epoch,error_mm=error))
                if error<best_error:
                    best,best_error,chosen=copy.deepcopy(model.state_dict()),error,epoch
                if epoch-chosen>=8:break
            path=folder/f'cnn_seed{seed}.pt';torch.save(best,path)
            logs.append(dict(size=size,family='cnn',setting=0,seed=seed,validation_error_mm=best_error,
                seconds=time.perf_counter()-begin,epoch=chosen,history=history,
                center=center.tolist(),scale=scale,parameters=sum(p.numel() for p in model.parameters())))
            models.append((size,'cnn',0,seed,path,center))
            print(size,'cnn',seed,'validation',best_error,flush=True)
        _write_json(out/'progress.json',logs)
    scores=pd.DataFrame(logs).groupby(['size','family','setting']).validation_error_mm.mean()
    selected={f'{size}/{family}':int(scores.loc[(size,family)].idxmin())
              for size in sizes for family in ['cnn','extra_trees','hist_gradient_boosting']}
    winner=scores.idxmin()
    _write_json(out/'selected_before_test.json',dict(selected=selected,
        overall=dict(size=int(winner[0]),family=winner[1],setting=int(winner[2])),validation=logs,
        hashes={str(p.relative_to(out)):_hash_file(p) for p in out.glob('n*/*')}))
    test=np.flatnonzero(new_rows.event_uid.isin(split['test']))
    tx=segmented_features(new[test],np.full(len(test),fixed['cut']),fixed['quiet'])
    ty=new_rows[['epicenter_row','epicenter_col']].to_numpy()[test]
    frames=[]
    for size,family,setting,seed,path,center in models:
        if selected[f'{size}/{family}']!=setting:continue
        if family=='cnn':
            model=TemporalCNN(raw.shape[1]);model.load_state_dict(torch.load(path,weights_only=True))
            p=infer(model,new[test],center,scale)
        else:p=joblib.load(path).predict(tx)
        if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
        f=new_rows.iloc[test][['event_uid','shot','strength_band','geometry','epicenter_region']].copy()
        f['size'],f['family'],f['seed'],f['setting']=size,family,seed,setting
        f['split']=np.where((f.geometry=='elliptical')&(f.strength_band==2),'ood','id')
        f['true_x_mm'],f['true_y_mm']=ty.T
        f['predicted_x_mm'],f['predicted_y_mm']=p.T
        f['error_mm']=np.linalg.norm(p-ty,axis=1);frames.append(f)
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(out/'predictions.csv',index=False)
    baseline=pd.read_csv(previous/'predictions.csv')
    identity=['event_uid','shot']
    a=frame.groupby(identity)[['true_x_mm','true_y_mm']].first()
    b=baseline.groupby(identity)[['true_x_mm','true_y_mm']].first()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a,b)
    combined=pd.concat([baseline,frame],ignore_index=True)
    metrics=combined.groupby(['size','family','split']).error_mm.agg(['mean','median',lambda v:v.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    combined.groupby(['size','family','split','strength_band']).error_mm.mean().to_csv(out/'strength_metrics.csv')
    paired=[]
    for family in ['cnn','extra_trees','hist_gradient_boosting']:
        for split_name in ['id','ood']:
            candidate=frame[(frame.family==family)&(frame['size']==1800)&(frame.split==split_name)]
            for ref_family,ref_size in [('svr',1800),('gnn',1800),(family,360)]:
                ref=combined[(combined.family==ref_family)&(combined['size']==ref_size)&(combined.split==split_name)]
                paired.append(dict(family=family,split=split_name,reference=ref_family,
                    reference_size=ref_size,**paired_delta(candidate,ref)))
    _write_json(out/'paired.json',paired)
    print(metrics.to_string(),flush=True);print('Additional model validation winner:',winner,flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
