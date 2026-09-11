"""Validation-frozen comparisons on paired weak-event observation windows.

Physical templates integrate over a finite prior bank of simulation hypotheses.
Their binned marginal likelihood is a composite score, NOT a joint likelihood.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .adaptive_localization import segmented_features
from .dataset import _write_json
from .localization import _hash_file
from .weak_growth import fit_svr
from .uncertainty_experiments import weighted_median
from .growth_more_models import paired_delta


def binned(values,width=32):
    values=np.asarray(values)
    starts=np.arange(0,values.shape[-1],width)
    lengths=np.diff(np.r_[starts,values.shape[-1]])
    return np.add.reduceat(values.astype(np.float64),starts,axis=-1),lengths


def read_events(root,ids,condition,rounds,templates=False):
    raw=[];bank=[];rows=[];sites=None
    for event in ids:
        label=json.loads((root/'events'/f'event_{event:04d}.json').read_text())
        if templates and label['role']!='train':raise ValueError('held-out latent template access')
        with np.load(root/'events'/f'event_{event:04d}.npz') as z:
            if label['role']!='train' and any(k.endswith('_expected') for k in z.files):
                raise ValueError('held-out latent arrays must not be generated')
            if sites is None:sites=z['sites']
            np.testing.assert_array_equal(sites,z['sites'])
            raw.append(z[condition][:,:,:rounds-1])
            if templates:
                sums,lengths=binned(z[condition+'_expected'][:,:rounds-1])
                bank.append(sums/lengths)
        for shot in range(len(raw[-1])):rows.append(dict(label,shot=shot))
    return np.concatenate(raw),pd.DataFrame(rows),None if not templates else np.asarray(bank),sites


def onset_weights(raw,cuts,quiet):
    """Observed change scores; these are heuristic weights, not calibrated timing posteriors."""
    ticks=raw.shape[-1];cumulative=raw.mean(1).cumsum(1)
    before=cumulative[:,cuts-1]/cuts
    after=(cumulative[:,-1,None]-cumulative[:,cuts-1])/(ticks-cuts)
    p=float(np.mean(quiet))
    scale=np.sqrt(p*(1-p)*(1/cuts+1/(ticks-cuts))/raw.shape[1])
    return softmax(np.clip((after-before)/scale,-10,10),axis=1)


def fit_noise_whitener(raw,bank,quiet):
    """Train-only within-bin detector covariance of observed-minus-physical means."""
    counts,lengths=binned(raw)
    scale=np.sqrt(quiet[:,None]*(1-quiet[:,None])*lengths)
    residual=(counts-np.repeat(bank,2,axis=0)*lengths)/scale
    vectors=residual.transpose(0,2,1).reshape(-1,8)
    covariance=np.cov(vectors,rowvar=False)
    covariance=.9*covariance+.1*np.diag(np.diag(covariance))
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    return (eigenvectors*(1/np.sqrt(np.maximum(eigenvalues,1e-5))))@eigenvectors.T


def predict_case(artifact,raw):
    """Only observed bits plus training artifacts; no query event parameters."""
    raw=np.asarray(raw)[:,:,:artifact['rounds']-1]
    if raw.shape[1:]!=(8,artifact['rounds']-1):raise ValueError('wrong detector/time dimensions')
    quiet=artifact['quiet'];n=len(raw)
    fixed=segmented_features(raw,np.full(n,345),quiet)
    result={name:m.predict(fixed) for name,m in artifact['fixed'].items()}
    result['fixed_et']=np.mean([m.predict(fixed) for m in artifact['forests']],axis=0)
    result['fixed_blend']=.5*(result['fixed_svr']+result['fixed_et'])
    cuts=np.arange(50,651,100);weights=onset_weights(raw,cuts,quiet)
    aligned={name:[] for name in artifact['aligned']}
    for cut in cuts:
        features=segmented_features(raw,np.full(n,cut),quiet)
        for name,m in artifact['aligned'].items():aligned[name].append(m.predict(features))
    for name,values in aligned.items():
        p=np.stack(values,axis=1)
        result[name+'_uniform']=p.mean(1)
        result[name+'_observed']=np.sum(p*weights[:,:,None],axis=1)
    counts,lengths=binned(raw)
    p=artifact['template_probabilities'];hypotheses=artifact['template_coordinates']
    logits=counts.reshape(n,-1)@np.log(p/(1-p)).reshape(len(p),-1).T
    logits+=np.sum(np.log1p(-p)*lengths,axis=(1,2))[None,:]
    logits-=logits.max(1,keepdims=True)
    scale=np.sqrt(quiet[:,None]*(1-quiet[:,None])*lengths)
    whitened=np.einsum('ij,njb->nib',artifact['noise_whitener'],counts/scale).reshape(n,-1)
    means=np.einsum('ij,njb->nib',artifact['noise_whitener'],p*lengths/scale).reshape(len(p),-1)
    gls=whitened@means.T-.5*np.square(means).sum(1)[None,:]
    gls-=gls.max(1,keepdims=True)
    for family,score in [('physical',logits),('physical_gls',gls)]:
        for beta in [0.,.03,.1,.3,1.,3.]:
            probability=softmax(beta*score,axis=1)
            result[f'{family}_mean_b{beta:g}']=probability@hypotheses
            result[f'{family}_median_b{beta:g}']=weighted_median(hypotheses,probability)
    result['prior']=np.broadcast_to(artifact['prior'],(n,2)).copy()
    for p in result.values():
        if p.shape!=(n,2) or not np.isfinite(p).all():raise ValueError('invalid predictions')
    return result


def train(root):
    root=Path(root).resolve();out=root/'experiment';out.mkdir(exist_ok=False)
    generation=json.loads((root/'protocol.json').read_text());progress=json.loads((root/'progress.json').read_text())
    if progress['status']!='complete' or len(progress['completed'])!=720:raise ValueError('incomplete acquisition')
    for record in progress['completed']:
        for suffix,digest in record['hashes'].items():
            if _hash_file(root/'events'/f"event_{record['event']:04d}.{suffix}")!=digest:raise ValueError('changed event file')
    split=generation['split'];train_ids=split['train'];val_ids=split['validation'];test_ids=split['test']
    assert (len(train_ids),len(val_ids),len(test_ids))==(432,144,144)
    assert len(set(train_ids+val_ids+test_ids))==720
    policy=dict(selection='minimum validation mean Euclidean error separately per acquisition and family; include prior',
        conditions=['nominal','quiet10'],rounds=[2048,4096,8192],seeds=[41,42,43],
        physical_inverse_temperatures=[0.,.03,.1,.3,1.,3.],bin_rounds=32,
        onset_candidates_rounds=list(range(50,651,100)),
        onset_training='supervised true-onset segmentation; prediction marginalizes candidate cuts using uniform or observed-change weights',
        template_semantics='training simulated exact categorical-Pauli detector marginals only; finite nuisance Monte Carlo bank; binned composite score is not a full joint likelihood',
        gls_semantics='alternative Gaussian score with train-only within-bin 8x8 residual covariance, 10 percent off-diagonal shrinkage; not a full spatiotemporal likelihood',
        weak_conditional_study=True,no_background_removal_from_observed_data=True,
        baseline='best fixed-segmentation model by validation; prior also separately reported',
        confidence_interval='paired physical-event bootstrap 10000; shots clustered; no multiplicity correction',
        source_sha256=_hash_file(Path(__file__)),generation_protocol_sha256=_hash_file(root/'protocol.json'))
    _write_json(out/'protocol.json',policy)
    selections={};models={}
    for condition in policy['conditions']:
        for rounds in policy['rounds']:
            case=f'{condition}_r{rounds}';folder=out/case;folder.mkdir()
            raw,rows,bank,sites=read_events(root,train_ids,condition,rounds,True)
            vr,vrows,_,vsites=read_events(root,val_ids,condition,rounds)
            np.testing.assert_array_equal(sites,vsites)
            # All templates have the same fixed hardware baseline; first 32
            # rounds precede the earliest physical onset (50 us).
            quiet=bank[:,:,0].mean(0)
            x=segmented_features(raw,np.full(len(raw),345),quiet)
            y=rows[['epicenter_row','epicenter_col']].to_numpy()
            vy=vrows[['epicenter_row','epicenter_col']].to_numpy()
            cuts=np.searchsorted((np.arange(1,rounds)+.5)*.001,rows.event_onset_ms.to_numpy())
            aligned=segmented_features(raw,np.clip(cuts,1,rounds-4),quiet)
            artifact=dict(rounds=rounds,condition=condition,quiet=quiet,sites=sites,
                fixed=dict(fixed_ridge=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(x,y),fixed_svr=fit_svr(x,y)),
                forests=[ExtraTreesRegressor(n_estimators=128,min_samples_leaf=4,random_state=s,n_jobs=2).fit(x,y) for s in [41,42,43]],
                aligned=dict(aligned_ridge=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(aligned,y),aligned_svr=fit_svr(aligned,y)),
                template_probabilities=np.clip(bank,1e-8,1-1e-8),template_coordinates=y[::2],
                noise_whitener=fit_noise_whitener(raw,bank,quiet),
                prior=weighted_median(y,np.ones((1,len(y))))[0])
            path=folder/'model.joblib';joblib.dump(artifact,path,compress=3);models[case]=path
            predictions=predict_case(artifact,vr)
            metrics={name:float(np.linalg.norm(p-vy,axis=1).mean()) for name,p in predictions.items()}
            selected={'overall':min(metrics,key=metrics.get)}
            for family,prefix in [('fixed','fixed_'),('onset','aligned_'),('physical','physical_')]:
                selected[family]=min((k for k in metrics if k.startswith(prefix)),key=metrics.get)
            selections[case]=dict(selected=selected,validation_mean_mm=metrics,model_sha256=_hash_file(path))
            # Verify saved-artifact inference before touching held-out test input.
            reloaded=predict_case(joblib.load(path),vr[:2])
            for k,v in reloaded.items():np.testing.assert_allclose(v,predictions[k][:2],atol=1e-8)
            print(case,selected,{k:metrics[v] for k,v in selected.items()},flush=True)
    best_nominal=min((c for c in selections if c.startswith('nominal')),key=lambda c:selections[c]['validation_mean_mm'][selections[c]['selected']['overall']])
    _write_json(out/'selected_before_test.json',dict(cases=selections,best_nominal_case=best_nominal))
    print('Frozen selections; evaluating held-out data',flush=True)
    frames=[]
    for case,path in models.items():
        artifact=joblib.load(path)
        raw,rows,_,sites=read_events(root,test_ids,artifact['condition'],artifact['rounds'])
        np.testing.assert_array_equal(sites,artifact['sites'])
        truth=rows[['epicenter_row','epicenter_col']].to_numpy()
        for name,p in predict_case(artifact,raw).items():
            frame=rows[['event_uid','shot','geometry','propagation_law','epicenter_region']].copy()
            frame['case']=case;frame['candidate']=name;frame['true_x_mm'],frame['true_y_mm']=truth.T
            frame['predicted_x_mm'],frame['predicted_y_mm']=p.T;frame['error_mm']=np.linalg.norm(p-truth,axis=1)
            frames.append(frame)
    data=pd.concat(frames,ignore_index=True);data.to_csv(out/'predictions.csv',index=False)
    metrics=data.groupby(['case','candidate']).error_mm.agg(['mean','median',lambda s:s.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    def sample(case,family):
        candidate='prior' if family=='prior' else selections[case]['selected'][family]
        return data[(data.case==case)&(data.candidate==candidate)]
    contrasts=[]
    def compare(case,family,reference,reference_family,kind):
        contrasts.append(dict(kind=kind,case=case,family=family,reference_case=reference,reference_family=reference_family,
            candidate=sample(case,family).candidate.iloc[0],reference_candidate=sample(reference,reference_family).candidate.iloc[0],
            **paired_delta(sample(case,family),sample(reference,reference_family))))
    for case in models:
        for family in ['onset','physical']:compare(case,family,case,'fixed','method')
        compare(case,'overall',case,'prior','information_vs_prior')
        condition,rounds=case.split('_r')
        if rounds!='2048':compare(case,'overall',f'{condition}_r2048','overall','duration')
        if condition=='quiet10':compare(case,'overall',f'nominal_r{rounds}','overall','background')
    _write_json(out/'paired.json',contrasts)
    selected_rows=[]
    for case in models:
        for family in ['overall','fixed','onset','physical','prior']:
            f=sample(case,family)
            selected_rows.append(dict(case=case,family=family,candidate=f.candidate.iloc[0],mean_mm=f.error_mm.mean(),p90_mm=f.error_mm.quantile(.9)))
    pd.DataFrame(selected_rows).to_csv(out/'selected_metrics.csv',index=False)
    print(pd.DataFrame(selected_rows).to_string(index=False),flush=True)


if __name__=='__main__':
    import sys
    train(sys.argv[1])
