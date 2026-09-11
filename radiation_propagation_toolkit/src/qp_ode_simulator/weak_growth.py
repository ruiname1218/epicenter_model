"""Double independent training events; observed-only weak-signal experiments."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .adaptive_localization import segmented_features
from .attention_mdn_gp import collect, MixturePosition, nn_predict
from .data_growth import STRATA, nested_events
from .dataset import _write_json
from .localization import _hash_file
from .growth_more_models import paired_delta
from .uncertainty_experiments import weighted_median


def new_split(labels,seed=90601):
    if labels.event_uid.duplicated().any() or labels.groupby(STRATA).ngroups!=36:
        raise ValueError('expected unique events in 36 strata')
    rng=np.random.default_rng(seed);result={'train':[],'validation':[],'test':[]}
    for _,group in labels.groupby(STRATA,sort=True):
        ids=rng.permutation(group.event_uid.to_numpy())
        if len(ids)!=50:raise ValueError('expected 50 new events per stratum')
        for key,part in [('train',ids[:30]),('validation',ids[30:40]),('test',ids[40:])]:
            result[key].extend(part.tolist())
    return result


def correlations(raw,cut):
    """Three post-cut spatial connected coincidences + lag-one autocovariances."""
    raw=np.asarray(raw);n,sites,ticks=raw.shape
    pair=np.triu_indices(sites,1)
    boundaries=np.linspace(cut,ticks,4,dtype=int);output=[]
    for start in range(0,n,128):
        block=raw[start:start+128].astype(np.float32);columns=[]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            v=block[:,:,lo:hi];mean=v.mean(2)
            cov=np.einsum('bit,bjt->bij',v,v)/v.shape[2]-mean[:,:,None]*mean[:,None,:]
            lag=(v[:,:,:-1]*v[:,:,1:]).mean(2)-v[:,:,:-1].mean(2)*v[:,:,1:].mean(2)
            columns.extend([cov[:,pair[0],pair[1]],lag])
        output.append(np.concatenate(columns,axis=1))
    return np.concatenate(output)


def fit_svr(x,y,weights=None,gamma=.001):
    scaler=StandardScaler().fit(x)
    estimator=MultiOutputRegressor(SVR(C=10,gamma=gamma,epsilon=.1)).fit(scaler.transform(x),y,sample_weight=weights)
    return make_pipeline(scaler,estimator)


def candidate_predictions(artifact,x,augmented,signal):
    predictions={name:model.predict(augmented if name.startswith('corr_') else x)
                 for name,model in artifact['regressors'].items()}
    for name,models in artifact['forests'].items():
        predictions[name]=np.mean([m.predict(x) for m in models],axis=0)
    center=artifact['prior']
    predictions['prior']=np.broadcast_to(center,(len(x),2)).copy()
    for size in [1800,3600]:
        predictions[f'blend_n{size}']=.5*(predictions[f'svr_n{size}']+predictions[f'et_n{size}'])
    base=predictions['blend_n3600'];prob=artifact['weak_gate'].predict_proba(x)[:,1]
    for specialist in ['prior','weak_ridge','weak_svr']:
        predictions[f'gate_{specialist}']=prob[:,None]*predictions[specialist]+(1-prob[:,None])*base
    for tau in artifact['thresholds']:
        weight=np.maximum(signal,0)/(np.maximum(signal,0)+tau)
        predictions[f'signal_shrink_{tau:g}']=weight[:,None]*base+(1-weight[:,None])*center
    # A deterministic soft gate, not true-strength routing.
    predictions['weak_probability']=prob
    return predictions


def run(runs):
    runs=Path(runs).resolve();root=runs/'weak_growth_20260908';out=root/'experiment';out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    parent=runs/'learning_curve_20260908/experiment'
    old=json.loads((parent/'protocol.json').read_text());spec=json.loads((parent/'geometry.json').read_text())['specification']
    sources=[*old['sources'],str(runs/'learning_curve_20260908/fresh_dataset'),str(runs/'attention_mdn_gp_20260908/fresh_dataset')]
    policy=dict(sources=sources,train_sizes=[1800,3600],old_tests_promoted_to_training=True,
        new_counts=dict(train=1080,validation=360,test=360),seeds=[41,42,43],
        low_weight_multipliers=[3.,10.],gate='ExtraTreesClassifier leaf12; trained low-band label; observed features only at inference',
        correlation_features=108,correlation_svr_gammas=[.001,.001*57/165],
        weak_selection='minimum low-band validation error with ordinary mean <= baseline + .02 mm',
        primary='minimum ordinary validation mean',baseline='blend_n3600',source_sha256=_hash_file(Path(__file__)))
    _write_json(out/'protocol.json',policy)
    raw,rows=collect(sources,spec);new,nrows=collect([root/'new_dataset'],spec)
    assert rows.event_uid.nunique()==3240 and len(rows)==6480
    assert nrows.event_uid.nunique()==1800 and len(nrows)==3600
    assert not rows.duplicated(['event_uid','shot']).any()
    assert not nrows.duplicated(['event_uid','shot']).any()
    for key in ['event_uid','generation_seed','syndrome_seed']:
        assert not set(rows[key])&set(nrows[key])
    parts=new_split(nrows.drop_duplicates('event_uid'))
    newtrain=np.flatnonzero(nrows.event_uid.isin(parts['train']))
    poolrows=pd.concat([rows,nrows.iloc[newtrain]],ignore_index=True)
    pool=np.concatenate([raw,new[newtrain]])
    subsets=nested_events(poolrows.drop_duplicates('event_uid'),[1800,3600],seed=90602)
    _write_json(out/'split.json',dict(**parts,subsets=subsets,manifests={str(p):_hash_file(Path(p)/'dataset_manifest.json') for p in [*sources,root/'new_dataset']}))
    val=np.flatnonzero(nrows.event_uid.isin(parts['validation'])&~((nrows.geometry=='elliptical')&(nrows.strength_band==2)))
    test=np.flatnonzero(nrows.event_uid.isin(parts['test']))
    train=np.flatnonzero(poolrows.event_uid.isin(subsets['3600']))
    assert len(train)==7200 and len(val)==600 and len(test)==720
    raw=pool[train];rows=poolrows.iloc[train].reset_index(drop=True)
    fixed=old['fixed_features'];quiet=np.asarray(fixed['quiet']);cut=fixed['cut']
    x=segmented_features(raw,np.full(len(raw),cut),quiet)
    vx=segmented_features(new[val],np.full(len(val),cut),quiet)
    augmented=np.concatenate([x,correlations(raw,cut)],axis=1)
    vaugmented=np.concatenate([vx,correlations(new[val],cut)],axis=1)
    y=rows[['epicenter_row','epicenter_col']].to_numpy();vy=nrows[['epicenter_row','epicenter_col']].to_numpy()[val]
    low=rows.strength_band.to_numpy()==0;vlow=nrows.strength_band.to_numpy()[val]==0
    signal=raw.mean((1,2))-quiet.mean();vsignal=new[val].mean((1,2))-quiet.mean()
    prior=weighted_median(y,np.ones((1,len(y))))[0]
    artifact=dict(regressors={},forests={},prior=prior,thresholds=np.unique(np.maximum(np.quantile(np.maximum(signal,0),[.1,.25,.5]),1e-5)).tolist(),specification=spec,fixed_features=fixed)
    for size in [1800,3600]:
        idx=np.flatnonzero(rows.event_uid.isin(subsets[str(size)]))
        artifact['regressors'][f'ridge_n{size}']=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(x[idx],y[idx])
        artifact['regressors'][f'svr_n{size}']=fit_svr(x[idx],y[idx])
        artifact['forests'][f'et_n{size}']=[ExtraTreesRegressor(n_estimators=128,min_samples_leaf=4,n_jobs=2,random_state=s).fit(x[idx],y[idx]) for s in [41,42,43]]
        print('Fitted baselines',size,flush=True)
    for factor in [3.,10.]:
        weights=np.where(low,factor,1.);weights/=weights.mean()
        artifact['regressors'][f'weighted_svr_{factor:g}']=fit_svr(x,y,weights)
        artifact['forests'][f'weighted_et_{factor:g}']=[ExtraTreesRegressor(n_estimators=128,min_samples_leaf=4,n_jobs=2,random_state=s).fit(x,y,sample_weight=weights) for s in [41,42,43]]
        print('Fitted low weighting',factor,flush=True)
    artifact['regressors']['weak_ridge']=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(x[low],y[low])
    artifact['regressors']['weak_svr']=fit_svr(x[low],y[low])
    artifact['weak_gate']=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=12,n_jobs=2,random_state=41).fit(x,low)
    artifact['regressors']['corr_ridge']=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(augmented,y)
    artifact['regressors']['corr_svr']=fit_svr(augmented,y)
    artifact['regressors']['corr_svr_dimscale']=fit_svr(augmented,y,gamma=.001*x.shape[1]/augmented.shape[1])
    joblib.dump(artifact,out/'models.joblib',compress=3)
    vp=candidate_predictions(artifact,vx,vaugmented,vsignal);vp.pop('weak_probability')
    # MDN baseline growth and low-weight NLL ablation, all seeds combined by predicted coordinate mean.
    mdn_specs=[(1800,1.),(3600,1.),(3600,3.),(3600,10.)]
    mdn_records=[];mdn_paths={}
    for size,factor in mdn_specs:
        idx=np.flatnonzero(rows.event_uid.isin(subsets[str(size)]))
        scaler=StandardScaler().fit(x[idx]);xx=scaler.transform(x[idx]);vv=scaler.transform(vx)
        center=y[idx].mean(0);scale=max(float(y[idx].std()),1e-6)
        weights=np.where(low[idx],factor,1.);weights/=weights.mean()
        name=f'mdn_n{size}_w{factor:g}';folder=out/name;folder.mkdir()
        joblib.dump(dict(scaler=scaler,center=center,scale=scale),folder/'preprocessing.joblib')
        paths=[];preds=[]
        for seed in [41,42,43]:
            torch.manual_seed(seed);model=MixturePosition(4,64)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            rng=np.random.default_rng(seed);best=None;best_score=np.inf;chosen=0;history=[]
            for epoch in range(1,81):
                model.train();order=rng.permutation(len(idx))
                for start in range(0,len(order),64):
                    ii=order[start:start+64];optimizer.zero_grad(set_to_none=True)
                    params=model(torch.tensor(xx[ii],dtype=torch.float32))
                    nll=MixturePosition.nll(params,torch.tensor((y[idx[ii]]-center)/scale,dtype=torch.float32))
                    loss=(nll*torch.tensor(weights[ii],dtype=torch.float32)).mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite MDN loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                pred,_=nn_predict(model,vv,center,scale);error=np.linalg.norm(pred-vy,axis=1)
                # Training objective weighting is also reflected in the validation objective.
                score=float(np.average(error,weights=np.where(vlow,factor,1.)))
                history.append(dict(epoch=epoch,weighted_validation_error_mm=score))
                if score<best_score:best,best_score,chosen=copy.deepcopy(model.state_dict()),score,epoch
                if epoch-chosen>=12:break
            path=folder/f'seed{seed}.pt';torch.save(best,path);paths.append(path)
            model.load_state_dict(best);pred,_=nn_predict(model,vv,center,scale);preds.append(pred)
            mdn_records.append(dict(name=name,seed=seed,epoch=chosen,history=history))
            print(name,seed,'epoch',chosen,flush=True)
        vp[name]=np.mean(preds,axis=0);mdn_paths[name]=paths
    validation={name:dict(mean_mm=float(np.linalg.norm(p-vy,axis=1).mean()),low_mm=float(np.linalg.norm(p[vlow]-vy[vlow],axis=1).mean())) for name,p in vp.items()}
    overall=min(validation,key=lambda n:validation[n]['mean_mm'])
    limit=validation['blend_n3600']['mean_mm']+.02
    eligible=[n for n,m in validation.items() if m['mean_mm']<=limit]
    weak_selected=min(eligible,key=lambda n:validation[n]['low_mm'])
    _write_json(out/'selected_before_test.json',dict(overall=overall,weak_selected=weak_selected,validation=validation,
        mdn_records=mdn_records,hashes={str(p.relative_to(out)):_hash_file(p) for p in [out/'models.joblib',*out.glob('mdn*/*')]}))
    print('Frozen:',overall,weak_selected,flush=True)
    # Only now compute held-out test features and predictions.
    tx=segmented_features(new[test],np.full(len(test),cut),quiet)
    ta=np.concatenate([tx,correlations(new[test],cut)],axis=1)
    restored=joblib.load(out/'models.joblib')
    reload_v=candidate_predictions(restored,vx[:2],vaugmented[:2],vsignal[:2])
    for name,p in reload_v.items():
        if name!='weak_probability':np.testing.assert_allclose(p,vp[name][:2],atol=1e-8)
    predictions=candidate_predictions(restored,tx,ta,new[test].mean((1,2))-quiet.mean())
    probability=predictions.pop('weak_probability')
    for name,paths in mdn_paths.items():
        pre=joblib.load(paths[0].parent/'preprocessing.joblib');data=pre['scaler'].transform(tx);ps=[]
        for path in paths:
            model=MixturePosition(4,64);model.load_state_dict(torch.load(path,weights_only=True))
            p,_=nn_predict(model,data,pre['center'],pre['scale']);ps.append(p)
        predictions[name]=np.mean(ps,axis=0)
    truth=nrows[['epicenter_row','epicenter_col']].to_numpy()[test];frames=[]
    for name,p in predictions.items():
        if not np.isfinite(p).all():raise ValueError('nonfinite predictions')
        f=nrows.iloc[test][['event_uid','shot','strength_band','geometry','epicenter_region']].copy()
        f['candidate']=name;f['split']=np.where((f.geometry=='elliptical')&(f.strength_band==2),'ood','id')
        f['true_x_mm'],f['true_y_mm']=truth.T;f['predicted_x_mm'],f['predicted_y_mm']=p.T
        f['weak_probability']=probability;f['error_mm']=np.linalg.norm(p-truth,axis=1);frames.append(f)
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'predictions.csv',index=False)
    metrics=frame.groupby(['candidate','split']).error_mm.agg(['mean','median',lambda v:v.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    frame.groupby(['candidate','split','strength_band']).error_mm.mean().to_csv(out/'strength_metrics.csv')
    comparisons=[]
    pairs=[('svr_n3600','svr_n1800'),('et_n3600','et_n1800'),('blend_n3600','blend_n1800'),('mdn_n3600_w1','mdn_n1800_w1'),
           (overall,'blend_n3600'),(weak_selected,'blend_n3600'),(weak_selected,'prior')]
    for candidate,reference in pairs:
        for condition in ['id','ood','low']:
            sub=frame[frame.strength_band==0] if condition=='low' else frame[frame.split==condition]
            comparisons.append(dict(candidate=candidate,reference=reference,condition=condition,
                **paired_delta(sub[sub.candidate==candidate],sub[sub.candidate==reference])))
    _write_json(out/'paired.json',comparisons)
    print(metrics.to_string(),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
