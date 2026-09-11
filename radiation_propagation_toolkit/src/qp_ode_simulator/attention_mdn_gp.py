"""Three bounded model families; train/selection separate from fresh scoring."""
import copy
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .adaptive_localization import segmented_features
from .dataset import iter_shards, _write_json
from .localization import _hash_file
from .temporal_localization import TemporalCNN, detector_series
from .temporal_gnn import graph_from_spec
from .growth_more_models import paired_delta


class AttentionPosition(nn.Module):
    def __init__(self, static, layers=1):
        super().__init__()
        self.temporal=TemporalCNN(len(static)).temporal
        self.register_buffer('static',torch.tensor(static,dtype=torch.float32))
        self.project=nn.Linear(133,32)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(
            32,4,dim_feedforward=64,dropout=.1,activation='gelu',batch_first=True),layers,
            enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Linear(len(static)*32,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,2))

    def forward(self,x):
        b,s,t=x.shape
        encoded=self.temporal(x.reshape(b*s,1,t)*10).reshape(b,s,128)
        features=torch.cat((encoded,x.mean(2,keepdim=True)*10,self.static.expand(b,-1,-1)),dim=2)
        return self.head(self.encoder(self.project(features)).flatten(1))


class MixturePosition(nn.Module):
    def __init__(self, components=2, hidden=64):
        super().__init__();self.components=components
        self.net=nn.Sequential(nn.Linear(57,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,components*5))

    def forward(self,x):
        z=self.net(x).reshape(-1,self.components,5)
        return z[:,:,0].log_softmax(1),z[:,:,1:3],z[:,:,3:5].clamp(-4.,2.)

    @staticmethod
    def nll(parameters,y):
        logw,mu,logs=parameters
        terms=-.5*(((y[:,None,:]-mu)*torch.exp(-logs))**2).sum(2)-logs.sum(2)-math.log(2*math.pi)
        return -torch.logsumexp(logw+terms,dim=1)


def nn_predict(model,x,center,scale):
    model.eval();points=[];params=[]
    with torch.no_grad():
        for start in range(0,len(x),64):
            output=model(torch.tensor(np.array(x[start:start+64]),dtype=torch.float32))
            if isinstance(output,tuple):
                logw,mu,logs=output
                points.append((logw.exp()[:,:,None]*mu).sum(1).numpy())
                params.append(tuple(v.numpy() for v in output))
            else:points.append(output.numpy())
    posterior=tuple(np.concatenate([p[i] for p in params]) for i in range(3)) if params else None
    return np.concatenate(points)*scale+center,posterior


def collect(paths,spec):
    arrays,frames=[],[]
    for path in paths:
        for data,labels in iter_shards(Path(path)):
            raw=detector_series(data,spec);shots=len(raw)//len(labels)
            rows=labels.iloc[np.repeat(np.arange(len(labels)),shots)].reset_index(drop=True)
            rows['shot']=np.tile(np.arange(shots),len(labels));arrays.append(raw);frames.append(rows)
    return np.concatenate(arrays),pd.concat(frames,ignore_index=True)


def train(runs):
    runs=Path(runs).resolve();out=runs/'attention_mdn_gp_20260908/experiment';out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    base=runs/'learning_curve_20260908/experiment'
    old=json.loads((base/'protocol.json').read_text());split=json.loads((base/'split.json').read_text())
    spec=json.loads((base/'geometry.json').read_text())['specification']
    configs=[dict(name=f'attention_l{l}',family='attention',layers=l) for l in [1,2]]
    configs += [dict(name=f'mdn_k{k}_h{h}',family='mdn',components=k,hidden=h) for k,h in [(2,64),(4,64),(4,128)]]
    configs += [dict(name=f'gp_l{l}_noise{n}',family='gp',length=l,noise=n) for l in [5.,15.] for n in [.2,1.]]
    info=dict(specification=spec,fixed_features=old['fixed_features'],configs=configs,
              selection='ordinary validation Euclidean error; NN seed-mean; posterior mean coordinates',
              seeds=[41,42,43],gp_optimizer=None,source_sha256=_hash_file(Path(__file__)),
              old_manifests=split['manifests'],split_sha256=_hash_file(base/'split.json'),
              epochs=dict(attention=40,mdn=80),patience=dict(attention=8,mdn=12))
    _write_json(out/'protocol.json',info)
    for path,digest in split['manifests'].items():
        assert _hash_file(Path(path)/'dataset_manifest.json')==digest
    raw,rows=collect(old['sources'],spec)
    vr,vrows=collect([runs/'learning_curve_20260908/fresh_dataset'],spec)
    idx=np.flatnonzero(rows.event_uid.isin(split['training']['1800']))
    val=np.flatnonzero(vrows.event_uid.isin(split['validation'])&~((vrows.geometry=='elliptical')&(vrows.strength_band==2)))
    assert len(idx)==3600 and len(val)==600
    y=rows[['epicenter_row','epicenter_col']].to_numpy()[idx]
    vy=vrows[['epicenter_row','epicenter_col']].to_numpy()[val]
    center=y.mean(0);scale=max(float(y.std()),1e-6)
    fixed=info['fixed_features']
    x=segmented_features(raw[idx],np.full(len(idx),fixed['cut']),fixed['quiet'])
    vx=segmented_features(vr[val],np.full(len(val),fixed['cut']),fixed['quiet'])
    scaler=StandardScaler().fit(x);x=scaler.transform(x);vx=scaler.transform(vx)
    _,static,_=graph_from_spec(spec)
    joblib.dump(dict(scaler=scaler,center=center,scale=scale,static=static),out/'preprocessing.joblib')
    logs=[]
    for config in configs:
        name,family=config['name'],config['family']
        folder=out/name;folder.mkdir()
        if family=='gp':
            kernel=Matern(length_scale=config['length'],length_scale_bounds='fixed',nu=1.5)+WhiteKernel(config['noise'],noise_level_bounds='fixed')
            model=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,optimizer=None,normalize_y=False).fit(x,(y-center)/scale)
            p=model.predict(vx)*scale+center
            error=float(np.linalg.norm(p-vy,axis=1).mean());joblib.dump(model,folder/'model.joblib')
            logs.append(dict(name=name,family=family,seed=-1,validation_error_mm=error))
            print(name,'validation',error,flush=True);continue
        for seed in [41,42,43]:
            torch.manual_seed(seed)
            model=AttentionPosition(static,config['layers']) if family=='attention' else MixturePosition(config['components'],config['hidden'])
            data=raw[idx] if family=='attention' else x
            valid=vr[val] if family=='attention' else vx
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            rng=np.random.default_rng(seed);best=None;best_error=np.inf;chosen=0;history=[]
            for epoch in range(1,info['epochs'][family]+1):
                model.train();order=rng.permutation(len(y))
                for start in range(0,len(order),64):
                    ii=order[start:start+64];optimizer.zero_grad(set_to_none=True)
                    output=model(torch.tensor(data[ii],dtype=torch.float32))
                    target=torch.tensor((y[ii]-center)/scale,dtype=torch.float32)
                    loss=nn.functional.mse_loss(output,target) if family=='attention' else MixturePosition.nll(output,target).mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                p,_=nn_predict(model,valid,center,scale);error=float(np.linalg.norm(p-vy,axis=1).mean())
                history.append(dict(epoch=epoch,validation_error_mm=error))
                if error<best_error:best,best_error,chosen=copy.deepcopy(model.state_dict()),error,epoch
                if epoch-chosen>=info['patience'][family]:break
            torch.save(best,folder/f'seed{seed}.pt')
            record=dict(name=name,family=family,seed=seed,validation_error_mm=best_error,epoch=chosen,
                        history=history,parameters=sum(p.numel() for p in model.parameters()))
            logs.append(record);_write_json(folder/f'seed{seed}.json',record)
            print(name,seed,'validation',best_error,'epoch',chosen,flush=True)
        _write_json(out/'progress.json',logs)
    scores=pd.DataFrame(logs).groupby(['family','name']).validation_error_mm.mean()
    selected={family:scores.loc[family].idxmin() for family in ['attention','mdn','gp']}
    _write_json(out/'selected_before_test.json',dict(selected=selected,overall=scores.idxmin()[1],validation=logs,
        hashes={str(p.relative_to(out)):_hash_file(p) for p in out.glob('*/*') if p.suffix in ['.pt','.joblib']}))
    print('Frozen selection:',selected,flush=True)


def evaluate(runs):
    runs=Path(runs).resolve();root=runs/'attention_mdn_gp_20260908';exp=root/'experiment'
    out=root/'confirmation';out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    info=json.loads((exp/'protocol.json').read_text());choice=json.loads((exp/'selected_before_test.json').read_text())
    prep=joblib.load(exp/'preprocessing.joblib');center,scale=prep['center'],prep['scale']
    raw,rows=collect([root/'fresh_dataset'],info['specification'])
    for path in info['old_manifests']:
        for _,old in iter_shards(Path(path)):
            for key in ['event_uid','generation_seed','syndrome_seed']:
                if set(rows[key])&set(old[key]):raise ValueError('fresh test overlap')
    fixed=info['fixed_features'];features=segmented_features(raw,np.full(len(raw),fixed['cut']),fixed['quiet'])
    x=prep['scaler'].transform(features);y=rows[['epicenter_row','epicenter_col']].to_numpy()
    frames=[];probability_scores=[]
    def record(name,seed,p):
        if not np.isfinite(p).all():raise ValueError('nonfinite predictions')
        f=rows[['event_uid','shot','strength_band','geometry','epicenter_region']].copy()
        f['candidate'],f['seed']=name,seed
        f['split']=np.where((f.geometry=='elliptical')&(f.strength_band==2),'ood','id')
        f['true_x_mm'],f['true_y_mm']=y.T;f['predicted_x_mm'],f['predicted_y_mm']=p.T
        f['error_mm']=np.linalg.norm(p-y,axis=1);frames.append(f)
    for family,name in choice['selected'].items():
        config=next(c for c in info['configs'] if c['name']==name);folder=exp/name
        if family=='gp':
            assert _hash_file(folder/'model.joblib')==choice['hashes'][f'{name}/model.joblib']
            model=joblib.load(folder/'model.joblib');mean,std=model.predict(x,return_std=True)
            p=mean*scale+center;sd=std*scale;record(name,-1,p)
            nll=(.5*((y-p)/sd)**2+np.log(sd)+.5*np.log(2*np.pi)).sum(1)
            np.savez_compressed(out/'gp_distribution.npz',mean=p,std=sd,event_uid=rows.event_uid.to_numpy(dtype=str),shot=rows.shot.to_numpy())
            probability_scores.append(dict(candidate=name,seed=-1,nll_per_mm2=float(nll.mean()),
                marginal_95_coverage=float((np.abs(y-p)<=1.96*sd).mean())))
        else:
            for seed in [41,42,43]:
                model=AttentionPosition(prep['static'],config['layers']) if family=='attention' else MixturePosition(config['components'],config['hidden'])
                assert _hash_file(folder/f'seed{seed}.pt')==choice['hashes'][f'{name}/seed{seed}.pt']
                model.load_state_dict(torch.load(folder/f'seed{seed}.pt',weights_only=True))
                p,posterior=nn_predict(model,raw if family=='attention' else x,center,scale);record(name,seed,p)
                if posterior is not None:
                    logw,mu,logs=posterior
                    nll=MixturePosition.nll(tuple(torch.tensor(a) for a in posterior),torch.tensor((y-center)/scale)).numpy()+2*np.log(scale)
                    np.savez_compressed(out/f'mdn_seed{seed}_distribution.npz',weights=np.exp(logw),means=mu*scale+center,std=np.exp(logs)*scale,event_uid=rows.event_uid.to_numpy(dtype=str),shot=rows.shot.to_numpy())
                    probability_scores.append(dict(candidate=name,seed=seed,nll_per_mm2=float(nll.mean())))
    parent=runs/'uncertainty_experiments_20260908';parent_info=json.loads((parent/'selected_before_test.json').read_text())
    assert _hash_file(parent/'models.joblib')==parent_info['artifact_sha256']
    blend=joblib.load(parent/'models.joblib');svr=blend['svr'].predict(features)
    et=np.mean([m.predict(features) for m in blend['et']],axis=0)
    record('svr',-1,svr);record('blend_0.5',-1,.5*(svr+et))
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'predictions.csv',index=False)
    metrics=frame.groupby(['candidate','split']).error_mm.agg(['mean','median',lambda a:a.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm'];metrics.to_csv(out/'metrics.csv')
    paired=[]
    for name in choice['selected'].values():
        for split in ['id','ood']:
            sub=frame[frame.split==split]
            paired.append(dict(candidate=name,split=split,**paired_delta(sub[sub.candidate==name],sub[sub.candidate=='blend_0.5'])))
    _write_json(out/'paired.json',paired);_write_json(out/'probability_scores.json',probability_scores)
    _write_json(out/'evaluation.json',dict(fresh_manifest_sha256=_hash_file(root/'fresh_dataset/dataset_manifest.json'),
        frozen_selection_sha256=_hash_file(exp/'selected_before_test.json'),parent_sha256=parent_info['artifact_sha256'],
        independent_events=int(rows.event_uid.nunique()),source_sha256=_hash_file(Path(__file__))))
    print(metrics.to_string(),flush=True)


if __name__=='__main__':
    import sys
    (evaluate if '--evaluate' in sys.argv[2:] else train)(sys.argv[1])
