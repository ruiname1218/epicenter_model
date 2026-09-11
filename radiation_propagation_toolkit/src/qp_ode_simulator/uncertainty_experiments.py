"""Validation-selected blending, shrinkage, grid posterior and empirical templates.

Template matching here is NOT a forward-physics likelihood reconstruction.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .adaptive_localization import segmented_features
from .dataset import iter_shards, _write_json
from .localization import _hash_file
from .temporal_localization import detector_series
from .growth_more_models import paired_delta


def weighted_median(points, weights, steps=100):
    """Smoothed Weiszfeld geometric median for Euclidean position error."""
    points=np.asarray(points); weights=np.asarray(weights)
    if points.ndim==2: points=np.broadcast_to(points,(len(weights),*points.shape))
    p=np.sum(points*weights[...,None],axis=1)/weights.sum(1,keepdims=True)
    for _ in range(steps):
        w=weights/np.maximum(np.linalg.norm(points-p[:,None,:],axis=2),1e-6)
        updated=np.sum(points*w[...,None],axis=1)/w.sum(1,keepdims=True)
        # Fixed iteration count makes predictions independent of batch members.
        p=updated
    return p


def grid_labels(y, edges, bins):
    cell=np.clip(((y-edges[0])/(edges[1]-edges[0])*bins).astype(int),0,bins-1)
    return cell[:,0]*bins+cell[:,1]


def predict_candidates(artifact, x, signal):
    """No event truth, strength labels or onset labels are consumed."""
    svr=artifact['svr'].predict(x)
    et=np.mean([m.predict(x) for m in artifact['et']],axis=0)
    predictions={'svr':svr,'extra_trees_ensemble':et}
    prior=artifact['prior']
    mixtures={}
    for alpha in [0.,.25,.5,.75,1.]:
        value=(1-alpha)*svr+alpha*et
        name=f'blend_{alpha:g}';predictions[name]=value;mixtures[name]=value
    for name,value in mixtures.items():
        for tau in artifact['thresholds']:
            w=np.maximum(signal,0)/(np.maximum(signal,0)+tau)
            predictions[f'{name}_shrink_{tau:.6g}']=w[:,None]*value+(1-w[:,None])*prior
    predictions['prior_median']=np.broadcast_to(prior,(len(x),2)).copy()
    probabilities={}
    for name,entry in artifact['grids'].items():
        prob=entry['model'].predict_proba(x)
        predictions[name+'_mean']=prob@entry['centers']
        predictions[name+'_median']=weighted_median(entry['centers'],prob)
        probabilities[name]=prob
    scaled=artifact['scaler'].transform(x)
    distances,indices=artifact['neighbors'].kneighbors(scaled,n_neighbors=512)
    for k in [32,128,512]:
        for temperature in [.5,2.]:
            d=distances[:,:k]
            # Relative distance weights avoid all-zero underflow; scale fitted on training queries.
            logits=-(d*d-d[:,:1]**2)/(2*(artifact['distance_scale']*temperature)**2)
            w=np.exp(logits);w/=w.sum(1,keepdims=True)
            pts=artifact['template_y'][indices[:,:k]]
            name=f'template_k{k}_t{temperature:g}'
            predictions[name+'_mean']=np.sum(pts*w[...,None],axis=1)
            predictions[name+'_median']=weighted_median(pts,w)
    return predictions,probabilities


def run(runs):
    runs=Path(runs).resolve();out=runs/'uncertainty_experiments_20260908';out.mkdir(exist_ok=False)
    previous=runs/'learning_curve_20260908/experiment'
    split=json.loads((previous/'split.json').read_text())
    old=json.loads((previous/'protocol.json').read_text())
    spec=json.loads((previous/'geometry.json').read_text())['specification']
    policy=dict(status='exploratory reuse of inspected holdout',train_events=1800,
        selection='ordinary validation mean Euclidean error; choose before test scoring',
        blending=[0,.25,.5,.75,1],shrink_quantiles=[.1,.25,.5,.75],
        grid_bins=[5,9],grid_leaf=[4,12],template_k=[32,128,512],template_temperature=[.5,2.],
        decoders=['mean','geometric_median'],uncertainty='uncalibrated grid probabilities',
        template_scope='empirical event-average syndrome features; NOT physics likelihood',
        previous_split_sha256=_hash_file(previous/'split.json'),source_sha256=_hash_file(Path(__file__)))
    _write_json(out/'protocol.json',policy)
    for path,digest in split['manifests'].items():
        if _hash_file(Path(path)/'dataset_manifest.json')!=digest:raise ValueError('manifest changed')

    def collect(paths):
        arrays,frames=[],[]
        for path in paths:
            for data,labels in iter_shards(Path(path)):
                a=detector_series(data,spec);shots=len(a)//len(labels)
                f=labels.iloc[np.repeat(np.arange(len(labels)),shots)].reset_index(drop=True)
                f['shot']=np.tile(np.arange(shots),len(labels));arrays.append(a);frames.append(f)
        return np.concatenate(arrays),pd.concat(frames,ignore_index=True)

    raw,rows=collect(old['sources']);new,nrows=collect([runs/'learning_curve_20260908/fresh_dataset'])
    for key in ['event_uid','generation_seed','syndrome_seed']:
        assert not set(rows[key])&set(nrows[key])
    idx=np.flatnonzero(rows.event_uid.isin(split['training']['1800']))
    val=np.flatnonzero(nrows.event_uid.isin(split['validation'])&~((nrows.geometry=='elliptical')&(nrows.strength_band==2)))
    test=np.flatnonzero(nrows.event_uid.isin(split['test']))
    quiet=np.asarray(old['fixed_features']['quiet']);cut=old['fixed_features']['cut']
    x=segmented_features(raw[idx],np.full(len(idx),cut),quiet)
    vx=segmented_features(new[val],np.full(len(val),cut),quiet)
    y=rows[['epicenter_row','epicenter_col']].to_numpy()[idx]
    vy=nrows[['epicenter_row','epicenter_col']].to_numpy()[val]
    signal=raw[idx].mean((1,2))-quiet.mean()
    thresholds=np.maximum(np.quantile(np.maximum(signal,0),[.1,.25,.5,.75]),1e-5)
    prior=weighted_median(y,np.ones((1,len(y))))[0]
    tree_root=runs/'growth_more_models_20260908'
    tree_selection=json.loads((tree_root/'selected_before_test.json').read_text())
    leaf=tree_selection['selected']['1800/extra_trees']
    svr_path=previous/'n1800/svr.joblib'
    tree_paths=[tree_root/f'n1800/extra_trees_{leaf}_seed{s}.joblib' for s in [41,42,43]]
    parent_hashes=json.loads((previous/'selected_before_test.json').read_text())['hashes']
    assert _hash_file(svr_path)==parent_hashes['n1800/svr.joblib']
    for p in tree_paths:assert _hash_file(p)==tree_selection['hashes'][str(p.relative_to(tree_root))]
    artifact=dict(svr=joblib.load(svr_path),et=[joblib.load(p) for p in tree_paths],
                  prior=prior,thresholds=np.unique(thresholds).tolist(),grids={})
    edges=np.stack([y.min(0)-1e-6,y.max(0)+1e-6])
    for bins in [5,9]:
        labels=grid_labels(y,edges,bins)
        for leaf in [4,12]:
            model=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=leaf,n_jobs=2,random_state=41).fit(x,labels)
            centers=np.stack([y[labels==c].mean(0) for c in model.classes_])
            artifact['grids'][f'grid{bins}_leaf{leaf}']=dict(model=model,centers=centers,edges=edges,bins=bins)
    # Average the two training shots per event, so a template never crosses a split.
    ids=rows.iloc[idx].event_uid.to_numpy()
    table=pd.DataFrame(x);table['event_uid']=ids
    target=pd.DataFrame(y);target['event_uid']=ids
    template=table.groupby('event_uid').mean();template_y=target.groupby('event_uid').mean().loc[template.index].to_numpy()
    scaler=StandardScaler().fit(template)
    scaled=scaler.transform(template.to_numpy())
    neighbors=NearestNeighbors().fit(scaled)
    d,_=neighbors.kneighbors(scaled,n_neighbors=33)
    artifact.update(scaler=scaler,neighbors=neighbors,template_y=template_y,
                    distance_scale=max(float(np.median(d[:,-1])),1e-6))
    joblib.dump(artifact,out/'models.joblib',compress=3)
    validation,_=predict_candidates(artifact,vx,new[val].mean((1,2))-quiet.mean())
    errors={name:float(np.linalg.norm(p-vy,axis=1).mean()) for name,p in validation.items()}
    selected=min(errors,key=errors.get)
    _write_json(out/'selected_before_test.json',dict(selected=selected,validation=errors,
        artifact_sha256=_hash_file(out/'models.joblib'),prior=prior.tolist(),thresholds=artifact['thresholds'],
        parent_hashes={str(p):_hash_file(p) for p in [svr_path,*tree_paths]}))
    print('Selected:',selected,'validation',errors[selected],flush=True)
    # Verify portable reload before touching test predictions.
    restored=joblib.load(out/'models.joblib')
    check,_=predict_candidates(restored,vx[:2],new[val[:2]].mean((1,2))-quiet.mean())
    for name in check:np.testing.assert_allclose(check[name],validation[name][:2])
    tx=segmented_features(new[test],np.full(len(test),cut),quiet)
    predictions,probabilities=predict_candidates(restored,tx,new[test].mean((1,2))-quiet.mean())
    ty=nrows[['epicenter_row','epicenter_col']].to_numpy()[test]
    frames=[]
    for name,p in predictions.items():
        assert np.isfinite(p).all()
        f=nrows.iloc[test][['event_uid','shot','strength_band','geometry','epicenter_region']].copy()
        f['candidate']=name;f['split']=np.where((f.geometry=='elliptical')&(f.strength_band==2),'ood','id')
        f['true_x_mm'],f['true_y_mm']=ty.T;f['predicted_x_mm'],f['predicted_y_mm']=p.T
        f['error_mm']=np.linalg.norm(p-ty,axis=1);frames.append(f)
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'predictions.csv',index=False)
    metrics=frame.groupby(['candidate','split']).error_mm.agg(['mean','median',lambda v:v.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    frame.groupby(['candidate','split','strength_band']).error_mm.mean().to_csv(out/'strength_metrics.csv')
    paired=[]
    for split_name in ['id','ood']:
        a=frame[(frame.candidate==selected)&(frame.split==split_name)]
        for ref in ['svr','extra_trees_ensemble']:
            b=frame[(frame.candidate==ref)&(frame.split==split_name)]
            paired.append(dict(selected=selected,reference=ref,split=split_name,**paired_delta(a,b)))
    _write_json(out/'paired.json',paired)
    # Save masses with sample identity, class IDs and support centers; not confidence guarantees.
    np.savez_compressed(out/'grid_probabilities.npz',event_uid=nrows.iloc[test].event_uid.to_numpy(dtype=str),
        shot=nrows.iloc[test].shot.to_numpy(),**probabilities)
    scores=[]
    for name,prob in probabilities.items():
        entry=artifact['grids'][name];labels=grid_labels(ty,entry['edges'],entry['bins'])
        onehot=labels[:,None]==entry['model'].classes_[None,:]
        scores.append(dict(candidate=name,nll=float(-np.log(np.maximum((prob*onehot).sum(1),1e-12)).mean()),
                           brier=float((((prob-onehot)**2).sum(1)+(~onehot.any(1))).mean()),
                           unseen_class_samples=int((~onehot.any(1)).sum())))
    _write_json(out/'probability_scores.json',scores)
    print('Candidates:',len(errors),flush=True);print(metrics.to_string(),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
