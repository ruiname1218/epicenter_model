"""Pre-frozen SVR versus adapted REI, with secondary Ridge/CNN comparisons."""
import argparse
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from .rei_fair_data import HORIZONS,check,read,_write_json,_hash_file
from .strength_benchmark import summary

SEEDS=(41,42,43)


def load(root,test=False):
    prefix='test' if test else 'fit';m=read(root/f'{prefix}_cache.json')
    assert m['rows_hash']==_hash_file(root/f'{prefix}_rows.csv')
    data={}
    for name,h in m['hashes'].items():
        assert _hash_file(root/name)==h;data[name[len(prefix)+1:-4]]=np.load(root/name)
    return data,pd.read_csv(root/f'{prefix}_rows.csv')


def rei_rates(rates,sites,physical,k):
    """Exactly rei_center's fixed algorithm on sufficient rate statistics."""
    sites=np.asarray(sites);physical=np.asarray(physical);d=np.linalg.norm(physical[:,None]-physical[None,:],axis=-1);np.fill_diagonal(d,np.inf)
    threshold=2*d.min(1).mean();result=np.full((len(rates),2),np.nan)
    for i,f in enumerate(rates):
        keep=f>1/(2*k)
        if keep.sum()<=2:continue
        xy=sites[keep];w=f[keep].copy();dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);np.fill_diagonal(dist,np.inf)
        if dist.min(1).mean()>threshold:continue
        if np.ptp(w)>0:w=(w-w.min())/np.ptp(w)
        w**=2
        if w.sum()>0:result[i]=np.average(xy,axis=0,weights=w)
    return result


def rei_prediction(data,p,h,k,center):
    pred=rei_rates(data[f'r{h}_{k}'],p['sites'],p['physical'],k);valid=np.isfinite(pred).all(1)
    return np.where(valid[:,None],pred,center),valid


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(24,32,7,2,3),nn.SiLU(),nn.Conv1d(32,64,5,2,2),nn.SiLU(),
            nn.Conv1d(64,64,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(1024,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,x):return self.net(x)


def cnn_predict(bundle,observed):
    x=((observed-bundle['mean'])/bundle['std']).astype(np.float32);model=CNN();model.load_state_dict(bundle['state']);model.eval()
    with torch.no_grad():pred=np.concatenate([model(torch.from_numpy(a)).numpy() for a in np.array_split(x,max(1,(len(x)+127)//128))])
    return pred*4+bundle['center']


def fit_cnn(x,y,tr,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);mean=x[tr].mean((0,2),keepdims=True);std=np.maximum(x[tr].std((0,2),keepdims=True),.001);center=y[tr].mean(0)
    inputs=torch.from_numpy(((x-mean)/std).astype(np.float32));target=torch.from_numpy(((y-center)/4).astype(np.float32));model=CNN()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);best=float('inf');saved=None
    for epoch in range(80):
        model.train()
        for ix in np.array_split(rng.permutation(np.flatnonzero(tr)),int(np.ceil(tr.sum()/64))):
            loss=nn.functional.mse_loss(model(inputs[ix]),target[ix]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval()
        with torch.no_grad():pred=model(inputs[~tr]).numpy()*4+center
        error=np.linalg.norm(pred-y[~tr],axis=1).mean()
        if error<best:
            best=error;saved=dict(state={k:v.detach().clone() for k,v in model.state_dict().items()},mean=mean,std=std,center=center,seed=seed,epoch=epoch+1,validation_mean=float(error))
    return saved


def train(root):
    if (root/'selection.json').exists():raise ValueError('frozen study')
    p=check(root);torch.set_num_threads(2);data,rows=load(root);tr=(rows.role=='train').to_numpy();y=rows[['epicenter_row','epicenter_col']].to_numpy()
    assert set(rows.loc[tr,'event_uid']).isdisjoint(rows.loc[~tr,'event_uid'])
    center=y[tr].mean(0);chosen={};history={};hashes={};records=[];cnnlog={}
    for h in HORIZONS:
        for family,key in [('SVR',f'x{h}'),('SVR_noquiet',f'u{h}'),('Ridge',f'x{h}')]:
            x=data[key];options={};configs={}
            params=[(c,g) for c in [.1,1.,10.] for g in [.057,.57]] if family.startswith('SVR') else [(a,None) for a in [1.,10.,100.,1000.]]
            for c,g in params:
                name=f'h{h}_{family}_{c}_{g}';model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=c,gamma=g/x.shape[1],epsilon=.1))) if family.startswith('SVR') else make_pipeline(StandardScaler(),Ridge(alpha=c))
                model.fit(x[tr],y[tr]);pred=model.predict(x[~tr]);error=float(np.linalg.norm(pred-y[~tr],axis=1).mean())
                options[name]=(model,error);configs[name]=dict(C_or_alpha=c,gamma_numerator=g,dimension=x.shape[1]);records.append(dict(model=name,horizon=h,**summary(np.linalg.norm(pred-y[~tr],axis=1))))
            selected=min(options,key=lambda k:options[k][1]);name=f'h{h}_{family}';joblib.dump(options[selected][0],root/f'{name}.joblib');hashes[f'{name}.joblib']=_hash_file(root/f'{name}.joblib')
            chosen[name]=dict(candidate=selected,validation_mean=options[selected][1],**configs[selected]);print('Selected',name,chosen[name],flush=True)
        preds=[]
        for seed in SEEDS:
            start=time.perf_counter();b=fit_cnn(data[f't{h}'],y,tr,seed);name=f'h{h}_CNN_{seed}';joblib.dump(b,root/f'{name}.joblib');hashes[f'{name}.joblib']=_hash_file(root/f'{name}.joblib')
            preds.append(cnn_predict(b,data[f't{h}'][~tr]));cnnlog[name]=dict(epoch=b['epoch'],validation_mean=b['validation_mean'],seconds=time.perf_counter()-start);print(name,cnnlog[name],flush=True)
        records.append(dict(model=f'h{h}_CNN',horizon=h,**summary(np.linalg.norm(np.mean(preds,axis=0)-y[~tr],axis=1))))
        candidates={}
        vdata={k:v[~tr] for k,v in data.items()}
        for k in sorted(set([256,512,1024,h*1024-1])):
            pred,valid=rei_prediction(vdata,p,h,k,center);error=np.linalg.norm(pred-y[~tr],axis=1)
            candidates[k]=float(error.mean());records.append(dict(model=f'h{h}_REI_K{k}',horizon=h,answer_rate=float(valid.mean()),**summary(error)))
        history[str(h)]=dict(selected=min(candidates,key=candidates.get),validation_mean=candidates)
        print('REI window',h,history[str(h)],flush=True)
    pd.DataFrame(records).to_csv(root/'validation_metrics.csv',index=False)
    _write_json(root/'selection.json',dict(selected=chosen,rei_history=history,cnn=cnnlog,center=center.tolist(),hashes=hashes,
        source_hash=_hash_file(Path(__file__)),protocol_hash=_hash_file(root/'protocol.json'),quiet_hash=_hash_file(root/'quiet.json'),fit_cache_hash=_hash_file(root/'fit_cache.json'),
        primary='ID h4 SVR minus REI_full all-strength mean',status='Frozen before test generation'))


def verify(root):
    check(root);s=read(root/'selection.json')
    assert s['source_hash']==_hash_file(Path(__file__))
    for key,file in [('protocol_hash','protocol.json'),('quiet_hash','quiet.json'),('fit_cache_hash','fit_cache.json')]:assert s[key]==_hash_file(root/file)
    for name,h in s['hashes'].items():assert _hash_file(root/name)==h
    return s


def groups(rows):
    b=rows.strength_band
    return dict(all=np.ones(len(rows),bool),weak=(b==0).to_numpy(),medium=(b==1).to_numpy(),strong=(b==2).to_numpy(),
        circular=(rows.geometry=='circular').to_numpy(),elliptical=(rows.geometry=='elliptical').to_numpy(),
        ballistic=(rows.propagation_law=='ballistic').to_numpy(),diffusive=(rows.propagation_law=='diffusive').to_numpy(),
        **{f'{band}_{shape}':((b==i)&(rows.geometry==shape)).to_numpy() for i,band in enumerate(['weak','medium','strong']) for shape in ['circular','elliptical']})


def evaluate(root):
    s=verify(root);p=read(root/'protocol.json');assert read(root/'test_manifest.json')['selection_hash']==_hash_file(root/'selection.json')
    torch.set_num_threads(2);data,rows=load(root,True);y=rows[['epicenter_row','epicenter_col']].to_numpy();pred={};answers={};timings={}
    for h in HORIZONS:
        for family,key in [('SVR',f'x{h}'),('Ridge',f'x{h}'),('SVR_noquiet',f'u{h}')]:
            name=f'h{h}_{family}';m=joblib.load(root/f'{name}.joblib');start=time.perf_counter();pred[h,family]=m.predict(data[key]);timings[name]=1000*(time.perf_counter()-start)/len(rows)
        values=[];start=time.perf_counter()
        for seed in SEEDS:
            b=joblib.load(root/f'h{h}_CNN_{seed}.joblib');v=cnn_predict(b,data[f't{h}']);pred[h,f'CNN_seed{seed}']=v;values.append(v)
        pred[h,'CNN']=np.mean(values,axis=0);timings[f'h{h}_CNN']=1000*(time.perf_counter()-start)/len(rows)
        for name,k in [('REI_full',h*1024-1),('REI_1024',1024),('REI_valK',int(s['rei_history'][str(h)]['selected']))]:
            start=time.perf_counter();pred[h,name],answers[h,name]=rei_prediction(data,p,h,k,np.asarray(s['center']));timings[f'h{h}_{name}']=1000*(time.perf_counter()-start)/len(rows)
        pred[h,'Prior']=np.repeat(np.asarray(s['center'])[None],len(rows),axis=0)
    frames=[];metrics=[];seedmetrics=[];paired=[];rng=np.random.default_rng(20260911671)
    for (h,name),v in pred.items():
        err=np.linalg.norm(v-y,axis=1);f=rows[['event_uid','shot','seed','domain','strength_band','geometry','propagation_law']].copy();f['horizon']=h;f['model']=name;f['pred_x'],f['pred_y']=v.T;f['error_mm']=err;f['answered']=answers.get((h,name),np.ones(len(rows),bool));frames.append(f)
        for domain in ['ID','noise2','slow']:
            for group,mask in groups(rows).items():
                mask=mask&(rows.domain==domain).to_numpy()
                if not mask.any():continue
                metrics.append(dict(horizon=h,model=name,domain=domain,group=group,events=int(rows.loc[mask,'event_uid'].nunique()),answer_rate=float(f.answered[mask].mean()),**summary(err[mask])))
            for seed in rows.loc[rows.domain==domain,'seed'].unique():
                mask=((rows.domain==domain)&(rows.seed==seed)).to_numpy();seedmetrics.append(dict(horizon=h,model=name,domain=domain,seed=int(seed),**summary(err[mask])))
    for h in HORIZONS:
        for name,ref in [('SVR','REI_full'),('SVR','REI_valK'),('SVR','REI_1024'),('CNN','REI_full'),('Ridge','REI_full'),('CNN','SVR'),('Ridge','SVR'),('SVR_noquiet','REI_full'),('SVR','SVR_noquiet')]:
            diff=np.linalg.norm(pred[h,name]-y,axis=1)-np.linalg.norm(pred[h,ref]-y,axis=1)
            for domain in ['ID','noise2','slow']:
                for group,mask in groups(rows).items():
                    mask=mask&(rows.domain==domain).to_numpy()
                    event_delta=pd.DataFrame(dict(uid=rows.loc[mask,'event_uid'],delta=diff[mask])).groupby('uid').delta.mean().to_numpy()
                    boot=rng.choice(event_delta,(10000,len(event_delta))).mean(1)
                    paired.append(dict(horizon=h,model=name,reference=ref,domain=domain,group=group,difference_mm=float(event_delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                        primary=h==4 and name=='SVR' and ref=='REI_full' and domain=='ID' and group=='all'))
    pd.concat(frames).to_csv(root/'predictions.csv',index=False);pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False);pd.DataFrame(seedmetrics).to_csv(root/'seed_metrics.csv',index=False);_write_json(root/'paired.json',paired)
    fit=pd.read_csv(root/'fit_rows.csv');assert set(rows.event_uid).isdisjoint(fit.event_uid);old=set(fit.generation_seed)
    for folder in root.parent.iterdir():
        if folder==root:continue
        for path in (folder/'events').glob('*.json'):
            seed=read(path).get('generation_seed')
            if seed is not None:old.add(seed)
    assert not set(rows.generation_seed)&old
    _write_json(root/'audit.json',dict(hashes_verified=True,old_seed_overlap=0,train_events=int(fit[fit.role=='train'].event_uid.nunique()),validation_events=int(fit[fit.role=='validation'].event_uid.nunique()),
        test_physical_events=int(rows.event_uid.nunique()),condition_events=rows.groupby('domain').event_uid.nunique().to_dict(),
        batch_model_ms_per_shot=timings,timing_note='Batch prediction only, excludes feature extraction/acquisition; CNN includes checkpoint loading. Not single-event latency.',
        interpretation='Adapted REI localization on event-present windows, not original implementation superiority or streaming detection. No physical latent targets used for learning. CNN seeds are initialization repeats on common training set.'))
    print(pd.DataFrame(metrics).query("horizon==4 and domain=='ID' and model in ['SVR','REI_full','REI_valK','CNN','Ridge'] and group in ['all','weak','medium','strong']").to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('root',type=Path);a=p.parse_args();globals()[a.stage](a.root)
