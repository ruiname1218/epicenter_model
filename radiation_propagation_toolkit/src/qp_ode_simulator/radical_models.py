"""Bounded, frozen-test pilot of observation and inverse-model improvements.

Oracle inputs are deliberately segregated; operational predictors only consume
observed detector records (or their deterministic summaries). Two shots of one
event always stay in the same split. No deployment default is changed here.
"""
import argparse
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from scipy.special import expit, softmax, logit
from .radical_data import read, check, _write_json, _hash_file
from .long_observation import summarize
from .strength_benchmark import masks

ARMS=('rates_plain','coinc_plain','coinc_joint','coinc_aux')
SEEDS=(41,42,43)


def load(root,test=False):
    prefix='test' if test else 'fit';m=read(root/f'{prefix}_cache.json')
    assert _hash_file(root/f'{prefix}_rows.csv')==m['rows_hash']
    data={}
    for name,h in m['hashes'].items():
        assert _hash_file(root/name)==h
        data[name[len(prefix)+1:-4]]=np.load(root/name)
    return data,pd.read_csv(root/f'{prefix}_rows.csv')


def objective(pred,rows):
    err=np.linalg.norm(pred-rows[['epicenter_row','epicenter_col']].to_numpy(),axis=1)
    return float(np.mean([err[rows.strength_band==b].mean() for b in [1,2]]))


def regression_inputs(data,quiet):
    latent=np.maximum(data['privileged_mean']-np.asarray(quiet)[None,:,None],0)
    latent=(latent/(latent.mean(1,keepdims=True)+.003)).reshape(len(latent),-1)
    return dict(base=data['base'],dense=data['dense'],pair=data['pair'],
        oracle_onset=np.column_stack([data['base'],data['nuisance'][:,10]]),
        oracle_nuisance=np.column_stack([data['base'],data['nuisance']]),
        oracle_nuisance_only=data['nuisance'],oracle_mean=latent)


class Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(48,48,5,2,2),nn.SiLU(),nn.Conv1d(48,64,5,2,2),nn.SiLU(),
            nn.Conv1d(64,64,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(1024,128),nn.SiLU(),nn.Dropout(.1))
        self.xy=nn.Linear(128,2);self.nu=nn.Linear(128,17);self.field=nn.Linear(128,384)

    def forward(self,x):
        h=self.encoder(x);return self.xy(h),self.nu(h),self.field(h)


def temporal_input(rates,coinc,arm,mean=None,std=None):
    # The public inference path has no access to nuisance/latent/coordinate labels.
    x=np.concatenate([rates,np.zeros_like(coinc) if arm=='rates_plain' else coinc],axis=1)
    if mean is None:mean=x.mean((0,2),keepdims=True)
    if std is None:std=np.maximum(x.std((0,2),keepdims=True),.001)
    return ((x-mean)/std).astype(np.float32),mean,std


def temporal_predict(bundle,rates,coinc):
    x,_,_=temporal_input(rates,coinc,bundle['arm'],bundle['mean'],bundle['std'])
    model=Temporal();model.load_state_dict(bundle['state']);model.eval()
    with torch.no_grad():
        pred=np.concatenate([model(torch.from_numpy(a))[0].numpy() for a in np.array_split(x,max(1,(len(x)+127)//128))])
    return pred*4


def fit_temporal(data,rows,arm,seed,epochs=80):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tr=(rows.role=='train').to_numpy();va=~tr;y=rows[['epicenter_row','epicenter_col']].to_numpy(dtype=np.float32)/4
    _,mean,std=temporal_input(data['time'][tr],data['coinc'][tr],arm)
    x,_,_=temporal_input(data['time'],data['coinc'],arm,mean,std)
    nu=StandardScaler().fit(data['nuisance'][tr]);field=StandardScaler().fit(data['privileged_mean'][tr].reshape(tr.sum(),-1))
    tensors=[torch.from_numpy(a.astype(np.float32)) for a in [x,y,nu.transform(data['nuisance']),field.transform(data['privileged_mean'].reshape(len(rows),-1))]]
    model=Temporal();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=float('inf');checkpoint=None
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(np.flatnonzero(tr)),max(1,int(np.ceil(tr.sum()/64)))):
            xy,n,f=model(tensors[0][ids]);loss=nn.functional.mse_loss(xy,tensors[1][ids])
            if arm in ['coinc_joint','coinc_aux']:loss=loss+.1*nn.functional.mse_loss(n,tensors[2][ids])
            if arm=='coinc_aux':loss=loss+.1*nn.functional.mse_loss(f,tensors[3][ids])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval()
        with torch.no_grad():prediction=model(tensors[0][va])[0].numpy()*4
        score=objective(prediction,rows[va])
        if score<best:
            best=score;checkpoint=dict(state={k:v.detach().clone() for k,v in model.state_dict().items()},arm=arm,seed=seed,
                mean=mean,std=std,epoch=epoch+1,score=score,nu_scaler=nu,field_scaler=field)
    return checkpoint


def inverse_predict(observed,profiles,centers,temperature):
    """Composite binomial score, NOT the correlated detector joint likelihood."""
    rates=observed.reshape(len(observed),-1)
    widths=np.tile(np.diff(np.linspace(0,4095,17,dtype=int)),observed.shape[1])
    p=np.clip(profiles.reshape(len(profiles),-1),1e-5,1-1e-5)
    weight=np.log(p)-np.log1p(-p);offset=np.log1p(-p)@widths
    result=[]
    for chunk in np.array_split(rates,max(1,(len(rates)+63)//64)):
        score=(chunk*widths)@weight.T+offset
        result.append(softmax(score/temperature,axis=1)@centers)
    return np.concatenate(result)


def train(root):
    check(root)
    if (root/'selection.json').exists():raise ValueError('selection is frozen; use a fresh run for retraining')
    torch.set_num_threads(2);data,rows=load(root);tr=(rows.role=='train').to_numpy();va=~tr
    assert set(rows.loc[tr,'event_uid']).isdisjoint(rows.loc[va,'event_uid'])
    y=rows[['epicenter_row','epicenter_col']].to_numpy();pred={};artifacts=[];cost={}
    inputs=regression_inputs(data,read(root/'quiet.json')['base'])
    for case,x in inputs.items():
        for kind in ['SVR','ET']:
            name=f'{case}_{kind}';start=time.perf_counter()
            model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=1,epsilon=.1,gamma=.57/x.shape[1]))) if kind=='SVR' else ExtraTreesRegressor(n_estimators=256,min_samples_leaf=3,max_features=1.,random_state=41,n_jobs=2)
            model.fit(x[tr],y[tr]);pred[name]=model.predict(x[va]);joblib.dump(model,root/f'{name}.joblib');artifacts.append(f'{name}.joblib')
            cost[name]=time.perf_counter()-start;print(name,objective(pred[name],rows[va]),flush=True)
    for arm in ARMS:
        values=[]
        for seed in SEEDS:
            start=time.perf_counter();bundle=fit_temporal(data,rows,arm,seed)
            name=f'{arm}_{seed}';joblib.dump(bundle,root/f'{name}.joblib');artifacts.append(f'{name}.joblib')
            values.append(temporal_predict(bundle,data['time'][va],data['coinc'][va]));pred[name]=values[-1]
            cost[name]=dict(seconds=time.perf_counter()-start,best_epoch=bundle['epoch']);print(name,bundle['epoch'],bundle['score'],flush=True)
        pred[arm]=np.mean(values,axis=0)
    # One latent profile per physical training event. Internal early-stopping
    # validation in this surrogate cannot split repeated shots of one event.
    unique=tr&~rows.event_uid.duplicated().to_numpy()
    theta=np.column_stack([y[unique],data['nuisance'][unique]])
    latent=data['privileged_mean'][unique].reshape(unique.sum(),-1)
    scaler=StandardScaler().fit(logit(np.clip(latent,1e-5,1-1e-5)))
    forward=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,128),max_iter=400,
        early_stopping=True,n_iter_no_change=25,random_state=41,alpha=.01))
    forward.fit(theta,scaler.transform(logit(np.clip(latent,1e-5,1-1e-5))))
    rng=np.random.default_rng(2026091107);centers=y[unique][rng.choice(unique.sum(),192,replace=False)]
    nu=data['nuisance'][unique][rng.choice(unique.sum(),64,replace=False)]
    candidates=np.column_stack([np.repeat(centers,len(nu),axis=0),np.tile(nu,(len(centers),1))])
    profiles=expit(scaler.inverse_transform(forward.predict(candidates))).astype(np.float32)
    valtheta=np.column_stack([y[va],data['nuisance'][va]])
    fidelity=float(np.sqrt(np.mean((expit(scaler.inverse_transform(forward.predict(valtheta)))-data['privileged_mean'][va].reshape(va.sum(),-1))**2)))
    inverse_settings={}
    for name,pro,xy in [('forward_inverse',profiles,candidates[:,:2]),('template_inverse',latent,y[unique])]:
        options={t:inverse_predict(data['counts'][va],pro,xy,t) for t in [1.,10.,100.]}
        temperature=min(options,key=lambda t:objective(options[t],rows[va]));pred[name]=options[temperature]
        joblib.dump(dict(profiles=pro,centers=xy,temperature=temperature),root/f'{name}.joblib');artifacts.append(f'{name}.joblib');inverse_settings[name]=temperature
        print(name,temperature,objective(pred[name],rows[va]),flush=True)
    joblib.dump(dict(model=forward,scaler=scaler),root/'forward_surrogate.joblib');artifacts.append('forward_surrogate.joblib')
    pred['Prior']=np.repeat(y[tr].mean(0)[None],va.sum(),axis=0)
    summarize(pred,rows[va]).to_csv(root/'validation_metrics.csv',index=False)
    operational=['base_SVR','base_ET',*ARMS,'forward_inverse','template_inverse']
    selected=min(operational,key=lambda n:objective(pred[n],rows[va]))
    # Six co-primary means: software selection, dense SVR, pair SVR, each on
    # medium/strong. Simultaneous Bonferroni bootstrap interval level 99.1667%.
    selection=dict(selected=selected,objective={n:objective(v,rows[va]) for n,v in pred.items()},
        primary=[selected,'dense_SVR','pair_SVR'],primary_reference='base_SVR',primary_groups=['medium','strong'],
        primary_interval=1-.05/6,seed_repeats='Neural initialization seeds only; common training events. Three fresh test generation roots.',
        inverse_settings=inverse_settings,forward_validation_probability_rmse=fidelity,cost=cost,
        prior=y[tr].mean(0).tolist(),source_hash=_hash_file(Path(__file__)),protocol_hash=_hash_file(root/'protocol.json'),
        fit_cache_hash=_hash_file(root/'fit_cache.json'),quiet_hash=_hash_file(root/'quiet.json'),
        hashes={name:_hash_file(root/name) for name in artifacts})
    _write_json(root/'selection.json',selection);print('FROZEN selected',selected,flush=True)


def verify(root):
    check(root);s=read(root/'selection.json')
    for key,name in [('source_hash',Path(__file__)),('protocol_hash',root/'protocol.json'),('fit_cache_hash',root/'fit_cache.json'),('quiet_hash',root/'quiet.json')]:
        assert s[key]==_hash_file(name)
    for name,h in s['hashes'].items():assert _hash_file(root/name)==h
    assert read(root/'test_manifest.json')['selection_hash']==_hash_file(root/'selection.json')
    return s


def evaluate(root):
    s=verify(root);torch.set_num_threads(2);data,rows=load(root,True);trainrows=pd.read_csv(root/'fit_rows.csv')
    assert set(rows.event_uid).isdisjoint(trainrows.event_uid)
    assert not set(rows.generation_seed)&set(trainrows.generation_seed)
    inputs=regression_inputs(data,read(root/'quiet.json')['base']);pred={};timings={}
    for case,x in inputs.items():
        for kind in ['SVR','ET']:
            name=f'{case}_{kind}';model=joblib.load(root/f'{name}.joblib');start=time.perf_counter();pred[name]=model.predict(x);timings[name]=(time.perf_counter()-start)*1000/len(x)
    for arm in ARMS:
        values=[];start=time.perf_counter()
        for seed in SEEDS:
            name=f'{arm}_{seed}';bundle=joblib.load(root/f'{name}.joblib');pred[name]=temporal_predict(bundle,data['time'],data['coinc']);values.append(pred[name])
        pred[arm]=np.mean(values,axis=0);timings[arm]=(time.perf_counter()-start)*1000/len(rows)
    for name in ['forward_inverse','template_inverse']:
        bundle=joblib.load(root/f'{name}.joblib');start=time.perf_counter();pred[name]=inverse_predict(data['counts'],**bundle);timings[name]=(time.perf_counter()-start)*1000/len(rows)
    pred['Prior']=np.repeat(np.asarray(s['prior'])[None],len(rows),axis=0)
    from .temporal_diagnosis import rei_center
    p=read(root/'protocol.json');values=[]
    for i in read(root/'ids.json')['test']:
        with np.load(root/'events'/f'{i:04d}.npz') as z:
            values.append(rei_center(z['base'],p['sites']['base'],p['physical']['base'],history_length=1024,circuit_repetitions=1))
    rei=np.concatenate(values);valid=np.isfinite(rei).all(1);pred['Adapted_REI']=np.where(valid[:,None],rei,s['prior'])
    summarize(pred,rows).to_csv(root/'metrics.csv',index=False)
    frames=[];y=rows[['epicenter_row','epicenter_col']].to_numpy()
    for name,xy in pred.items():
        f=rows[['event_uid','shot','test_seed','strength_band','geometry']].copy();f['model']=name;f['pred_x'],f['pred_y']=xy.T
        f['error_mm']=np.linalg.norm(xy-y,axis=1);frames.append(f)
    pd.concat(frames).to_csv(root/'predictions.csv',index=False)
    blocks=[]
    for seed in sorted(rows.test_seed.unique()):
        m=rows.test_seed==seed;f=summarize({n:v[m] for n,v in pred.items()},rows[m]);f['test_seed']=seed;blocks.append(f)
    pd.concat(blocks).to_csv(root/'seed_metrics.csv',index=False)
    contrasts=[(n,'base_SVR') for n in pred if n!='base_SVR']+[
        ('coinc_plain','rates_plain'),('coinc_joint','coinc_plain'),('coinc_aux','coinc_plain'),('forward_inverse','template_inverse')]
    paired=[]
    for name,ref in contrasts:
        for band,m in masks(rows).items():
            if band not in ['weak','medium','strong']:continue
            delta=np.linalg.norm(pred[name][m]-y[m],axis=1)-np.linalg.norm(pred[ref][m]-y[m],axis=1)
            events=pd.DataFrame(dict(uid=rows.loc[m,'event_uid'],delta=delta)).groupby('uid').delta.mean().to_numpy()
            boot=np.random.default_rng(2026091119).choice(events,(10000,len(events))).mean(1)
            alpha=.05/6
            paired.append(dict(model=name,reference=ref,group=band,difference_mm=float(events.mean()),
                ci95=np.quantile(boot,[.025,.975]).tolist(),ci_primary=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),
                primary=name in s['primary'] and ref=='base_SVR' and band in ['medium','strong']))
    _write_json(root/'paired.json',paired)
    old=set()
    for folder in root.parent.iterdir():
        if folder==root:continue
        for path in (folder/'events').glob('*.json'):
            seed=read(path).get('generation_seed')
            if seed is not None:old.add(seed)
    assert not set(rows.generation_seed)&old
    _write_json(root/'audit.json',dict(test_events=int(rows.event_uid.nunique()),test_shots=len(rows),
        train_events=int(trainrows[trainrows.role=='train'].event_uid.nunique()),validation_events=int(trainrows[trainrows.role=='validation'].event_uid.nunique()),
        unseen_generation_seeds=True,artifact_hashes_verified=True,rei_answer_rate=float(valid.mean()),
        batch_prediction_ms_per_shot=timings,timing_note='Batch throughput incl. neural deserialization; excludes acquisition/features; not online single-event latency.',
        oracle_note='All oracle_* rows require unavailable truth, are diagnostics only, and were excluded from software model selection.'))
    print(summarize({n:pred[n] for n in ['base_SVR',s['selected'],'dense_SVR','pair_SVR','Adapted_REI']},rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('root',type=Path);a=p.parse_args()
    (train if a.stage=='train' else evaluate)(a.root)
