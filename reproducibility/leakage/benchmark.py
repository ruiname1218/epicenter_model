"""Controlled leakage counterparts of the radiation poster (no measured-data fit).

Run with the toolkit venv. All uncertainty resamples events, never individual shots.
Coordinates use the radiation convention: centered Stim coordinate * nominal pitch.
"""
from pathlib import Path
import json, hashlib, time, copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import joblib
import stim
from reserveq.codes import SyndromeData
from reserveq.events import inject, _get_code_graph
from qp_ode_simulator.rei_fair_models import rei_rates

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'poster_matched_v3'
SEEDS = (41, 42, 43)
NTRAIN, NVAL, NTEST = 864, 180, 540
CNN_EPOCHS = 60

def sample(circuit, shots, seed):
    return circuit.compile_detector_sampler(seed=int(seed) % (2**63-1)).sample(shots=shots)

def make_sd(bits, coords, circuit, distance=5, rounds=4, p=.003):
    return SyndromeData(bits.astype(np.uint8),coords,'rotated_surface',distance,rounds,p,circuit=circuit)

def geometry(d, rounds=32, p=.003, pitch=1.):
    # Match the radiation main comparison's three circuit-noise channels.
    factor=p/.003
    circuit=stim.Circuit.generated('surface_code:rotated_memory_z',distance=d,rounds=rounds,
        after_clifford_depolarization=.0005*factor,before_measure_flip_probability=.001*factor,
        after_reset_flip_probability=.001*factor)
    detector_coords=circuit.get_detector_coordinates()
    coords=np.asarray([detector_coords[i] for i in range(circuit.num_detectors)])
    graph = _get_code_graph(make_sd(sample(circuit, 1, 919), coords, circuit, d, rounds, p))
    raw = np.array(list(graph['qubit_coords'].values()))
    center = (raw.max(0)+raw.min(0))/2
    sites = np.unique(coords[coords[:,2]==1,:2], axis=0)
    lookup = {tuple(c): i for i,c in enumerate(coords)}
    index = np.array([[lookup[(*s,t)] for t in range(1,rounds)] for s in sites])
    assert index.size == (d*d-1)*(rounds-1)
    return circuit,coords,graph,index,(sites-center)*pitch,(raw-center)*pitch,center

def dataset(name, d=5, pitch=1., domain='full', p=.003, transport=.123, test_only=False):
    path=OUT/f'{name}.npz'
    if path.exists(): return dict(np.load(path))
    circuit,coords,graph,index,sites,physical,center=geometry(d,p=p,pitch=pitch)
    candidates=[]
    for q in graph['data_qubits']:
        xy=(np.array(graph['qubit_coords'][q])-center)*pitch
        allowed=np.array([-2,0,2]) if domain=='common9' else np.arange(-4,5,2)
        if all(np.min(abs(allowed-v))<1e-8 for v in xy): candidates.append((tuple(xy),q))
    candidates.sort()
    assert len(candidates)==(9 if domain=='common9' else 25), (name,len(candidates))
    parts=[('test',0,NTEST)] if test_only else [('train',s,NTRAIN) for s in SEEDS]+[('validation',s,NVAL) for s in SEEDS]+[('test',0,NTEST)]
    xs=[];ys=[];rows=[]
    for role,rep,n in parts:
        role_num={'train':1,'validation':2,'test':3}[role]
        rng=np.random.default_rng(700000+role_num*10000+rep)
        # Event parameters identical across layouts/noise/OOD; shot noise is independent.
        qidx=rng.integers(len(candidates),size=n); onsets=rng.integers(0,3,size=n)
        strengths=np.resize(np.arange(3),n);rng.shuffle(strengths)
        bits=sample(circuit,2*n,900000+role_num*10000+rep+int(p*1e6)).astype(np.uint8)
        for i in range(n):
            xy,q=candidates[qidx[i]]
            sd=make_sd(bits[2*i:2*i+2],coords,circuit,d,32,p);sd._code_graph=graph
            leaked,_,meta=inject(sd,fractions=(0,1,0),seed=1200000+role_num*100000+rep*1000+i,
                seed_qubits=graph['data_qubits'],forced_seed_qubit=q,forced_onset_round=int(onsets[i]),
                leak_flip=(.15,.30,.50)[strengths[i]],leak_transport_rate=transport,
                leak_decay_tau=7.63,leak_persist=32,return_metadata=True)
            assert all(m['onset_round']==onsets[i] and m['seed_qubit']==q for m in meta)
            xs.extend(leaked.shots[:,index]);ys.extend([xy,xy])
            for shot in range(2): rows.append((f'{role}_{rep}_{i}',role,rep,shot,int(strengths[i]),int(onsets[i])))
        print('generated',name,role,rep,n,flush=True)
    quiet=sample(circuit,8192,819191).astype(float)[:,index].mean(0)
    rows=pd.DataFrame(rows,columns=['event','role','rep','shot','strength','onset'])
    data=dict(x=np.asarray(xs,dtype=np.uint8),y=np.asarray(ys),quiet=quiet,sites=sites,physical=physical,
              **{c:rows[c].to_numpy().astype(str) if c in ('role','event') else rows[c].to_numpy() for c in rows})
    np.savez_compressed(path,**data)
    return data

def features(data, rounds=16, variant='F0', quiet=None):
    x=data['x'][:,:,:rounds-1].astype(float)
    q=(data['quiet'] if quiet is None else quiet)[:,:rounds-1]
    x=np.maximum(x-q,0)
    # Per-event permutations, independently regenerated for each event, shared by its shots.
    if variant in ('N1','N2'):
        axis=0 if variant=='N1' else 1
        for start in range(0,len(x),2):
            digest=hashlib.sha256((variant+str(data['event'][start])).encode()).digest()
            perm=np.random.default_rng(int.from_bytes(digest[:8],'little')).permutation(x.shape[axis+1])
            x[start:start+2]=np.take(x[start:start+2],perm,axis=axis+1)
    # Equal-width bins replace radiation's millisecond bins; all models see the same window.
    x=np.stack([x[:,:,b].mean(2) for b in np.array_split(np.arange(rounds-1),min(5,rounds-1))],axis=2)
    if variant=='F1': x=x.mean(2,keepdims=True)
    if variant=='F2': return x.mean(1)
    x=x/(x.mean(1,keepdims=True)+.003)
    return x.reshape(len(x),-1)

def fit_tab(x,y,tr,va,family):
    options=[(c,g) for c in [.1,1.,10.] for g in [.057,.57]] if family=='SVR' else [(a,None) for a in [1.,10.,100.,1000.]]
    best=None;score=np.inf;log=[]
    for c,g in options:
        est=MultiOutputRegressor(SVR(C=c,gamma=g/x.shape[1],epsilon=.1)) if family=='SVR' else Ridge(alpha=c)
        model=make_pipeline(StandardScaler(),est);model.fit(x[tr],y[tr])
        error=float(np.linalg.norm(model.predict(x[va])-y[va],axis=1).mean())
        log.append(dict(c=c,gamma_numerator=g,validation_mean=error))
        if error<score: best,score=model,error
    return best,log

def cnn_grid(data,rounds=16):
    sites=data['sites'];a=np.unique(sites[:,0]);b=np.unique(sites[:,1])
    ix=np.searchsorted(a,sites[:,0]);iy=np.searchsorted(b,sites[:,1])
    x=np.zeros((len(data['x']),rounds,len(a),len(b)),dtype=np.float32)
    x[:,:rounds-1,ix,iy]=data['x'][:,:,:rounds-1].transpose(0,2,1)
    x[:,-1,ix,iy]=1 # static geometry mask; no event metadata enters input
    return x

def fit_cnn(x,y,tr,va,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=nn.Sequential(nn.Conv2d(x.shape[1],16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(16*x.shape[2]*x.shape[3],64),nn.ReLU(),nn.Linear(64,2))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    inp=torch.from_numpy(x);target=torch.from_numpy(y.astype(np.float32)/4)
    best=np.inf;saved=None;epoch_best=0
    for epoch in range(CNN_EPOCHS):
        model.train()
        for ids in np.array_split(rng.permutation(np.flatnonzero(tr)),int(np.ceil(tr.sum()/128))):
            loss=nn.functional.mse_loss(model(inp[ids]),target[ids]);opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): err=np.linalg.norm(model(inp[va]).numpy()*4-y[va],axis=1).mean()
        if err<best: best=float(err);saved=copy.deepcopy(model.state_dict());epoch_best=epoch+1
    model.load_state_dict(saved);return model,dict(epoch=epoch_best,validation_mean=best)

def cnn_predict(model,x):
    model.eval()
    with torch.no_grad(): return np.concatenate([model(torch.from_numpy(a)).numpy()*4 for a in np.array_split(x,max(1,int(np.ceil(len(x)/256))))])

RECORDS=[];SELECTION=[]
def record(data,pred,study,condition,model,rep,valid=None):
    mask=data['role']=='test';err=np.linalg.norm(pred-data['y'][mask],axis=1)
    df=pd.DataFrame({k:data[k][mask] for k in ('event','shot','strength','onset')})
    df['study']=study;df['condition']=condition;df['model']=model;df['rep']=rep
    df['target_x']=data['y'][mask,0];df['target_y']=data['y'][mask,1]
    df['pred_x']=pred[:,0];df['pred_y']=pred[:,1];df['error']=err
    df['valid']=True if valid is None else valid
    RECORDS.append(df)

def comparison(data,study,condition,rounds=16,variant='F0',families=('SVR',),extra=None):
    x=features(data,rounds,variant);y=data['y'];te=data['role']=='test'
    for rep in SEEDS:
        tr=(data['role']=='train')&(data['rep']==rep);va=(data['role']=='validation')&(data['rep']==rep)
        assert not set(data['event'][tr])&set(data['event'][va]) and not set(data['event'][tr])&set(data['event'][te])
        for family in families:
            stem=f'{study}_{condition}_{family}_{rep}'
            if family=='CNN':
                gx=cnn_grid(data,rounds);model,log=fit_cnn(gx,y,tr,va,rep);pred=cnn_predict(model,gx[te])
                torch.save(model.state_dict(),OUT/f'{stem}.pt')
            else:
                model,log=fit_tab(x,y,tr,va,family);pred=model.predict(x[te]);joblib.dump(model,OUT/f'{stem}.joblib')
            SELECTION.append(dict(stem=stem,validation=log));record(data,pred,study,condition,family,rep)
            if extra:
                for cond,other in extra.items():
                    ot=other['role']=='test'
                    pred=cnn_predict(model,cnn_grid(other,rounds)[ot]) if family=='CNN' else model.predict(features(other,rounds,variant,quiet=data['quiet'])[ot])
                    record(other,pred,'robustness',cond,family,rep)
            print('fit',stem,flush=True)
        if study=='models':
            center=y[tr].mean(0)
            # History length chosen on validation, as in radiation comparison.
            choices=sorted(set([3,7,rounds-1]));best=None;score=np.inf
            for k in choices:
                rates=data['x'][:,:,rounds-1-k:rounds-1].mean(2)
                raw=rei_rates(rates,data['sites'],data['physical'],k)
                pred=np.where(np.isfinite(raw),raw,center)
                e=np.linalg.norm(pred[va]-y[va],axis=1).mean()
                if e<score: best=k;score=e
            SELECTION.append(dict(stem=f'REI_{rep}',history=best,validation_mean=float(score)))
            for cond,other in {condition:data,**(extra or {})}.items():
                ot=other['role']=='test';rates=other['x'][ot,:,rounds-1-best:rounds-1].mean(2)
                raw=rei_rates(rates,other['sites'],other['physical'],best);valid=np.isfinite(raw).all(1)
                pred=np.where(valid[:,None],raw,center)
                record(other,pred,'models' if cond==condition else 'robustness',cond,'Adapted REI',rep,valid)
                record(other,np.tile(center,(ot.sum(),1)),'models' if cond==condition else 'robustness',cond,'Constant',rep)
    pd.concat(RECORDS).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'selection.json').write_text(json.dumps(SELECTION,indent=2))

def summarize():
    df=pd.concat(RECORDS);res=[];paired=[];rng=np.random.default_rng(5566)
    for keys,g in df.groupby(['study','condition','model']):
        for group,h in [('all',g)]+[(s,g[g.strength==i]) for i,s in enumerate(['weak','medium','strong'])]:
            event=h.groupby('event').error.mean().to_numpy()
            boot=event[rng.integers(len(event),size=(2000,len(event)))].mean(1)
            res.append(dict(zip(['study','condition','model'],keys),group=group,events=len(event),mean=float(h.error.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),median=float(h.error.median()),p90=float(h.error.quantile(.9)),within_1=float((h.error<=1).mean()),above_4=float((h.error>4).mean()),answer_rate=float(h.valid.mean()),conditional_mean=float(h.loc[h.valid,'error'].mean()),seed_mean_min=float(h.groupby('rep').error.mean().min()),seed_mean_max=float(h.groupby('rep').error.mean().max())))
    for keys,g in df.groupby(['study','condition']):
        if 'Adapted REI' not in set(g.model):continue
        table=g.groupby(['event','model']).error.mean().unstack()
        for model in ['SVR','Ridge','CNN']:
            delta=(table[model]-table['Adapted REI']).to_numpy();boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
            paired.append(dict(study=keys[0],condition=keys[1],model=model,delta=float(delta.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    pd.DataFrame(res).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_vs_rei.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=pd.DataFrame(res);allm=metrics[metrics.group=='all']
    for study in ['models','ablation','robustness','distance','density','observation']:
        sub=allm[allm.study==study]
        fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained')
        if study in ('models','ablation'):
            labels=sub.model if study=='models' else sub.condition
            ax.bar(labels,sub['mean'],color=['#a9adb2' if v in ('F2','N1','Constant','Adapted REI') else '#347daa' for v in labels],width=.6)
            ax.errorbar(np.arange(len(sub)),sub['mean'],yerr=[sub['mean']-sub.ci_low,sub.ci_high-sub['mean']],fmt='none',ecolor='#222222',capsize=3)
        else:
            for model,g in sub.groupby('model'):ax.plot(g.condition,g['mean'],'o-',label=model)
            ax.legend(frameon=False)
        ax.set_ylabel('Mean localization error (nominal mm)');ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        fig.savefig(OUT/f'{study}.png',dpi=220);fig.savefig(OUT/f'{study}.pdf');plt.close(fig)
    print(allm.to_string(index=False),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    protocol=dict(train_events_per_rep=NTRAIN,validation_events_per_rep=NVAL,test_events=NTEST,training_seeds=SEEDS,shots_per_event=2,
        primary_rounds=16,observed_internal_rounds=15,discarded='initial and terminal partial detector rounds',onset='uniform integers 0,1,2; shared between shots',
        strength_proxy={'weak':.15,'medium':.3,'strong':.5},strength_caveat='Bernoulli corruption probability, not radiation energy or measured leakage amplitude',
        nominal=dict(distance=5,noise_reference_argument=.003,gate_noise=.0005,measurement_noise=.001,reset_noise=.001,transport=.123,decay=7.63,persist=32,pitch=1),
        coordinates='(Stim xy - full-qubit bounding-box center) * nominal pitch; mm is a chosen spatial scale, not calibrated hardware',
        density='d5 pitch1 vs d15 pitch1/3; identical full 10x10 footprint and exactly identical 25 allowed source coordinates',
        distance='d3,d5,d7 pitch1; same nine source coordinates in [-2,2]^2',
        robustness='frozen nominal models + quiet reference; p=.006 or unseen transport=.03',
        scope='detector-level synthetic injection, data-qubit seeds, rotated surface code only; no measured response comparison; no decoding claim',
        inference='one shot only; no truth strength/onset/position; 2 shots grouped only for split and uncertainty',
        uncertainty='2000 paired event bootstrap replicates; training replicas averaged for CI, min/max separately; exploratory multiple comparisons, no confirmatory p-values')
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==json.loads(json.dumps(protocol)), 'Protocol changed; use new output directory'
    protocol_path.write_text(json.dumps(protocol,indent=2))
    data=dataset('nominal');extra={'double_noise':dataset('double_noise',p=.006,test_only=True),'slow_transport':dataset('slow_transport',transport=.03,test_only=True)}
    comparison(data,'models','nominal',families=('SVR','Ridge','CNN'),extra=extra)
    for v in ('F0','F1','F2','N1','N2'):comparison(data,'ablation',v,variant=v)
    for r in (4,16,32):comparison(data,'observation',f'{r:02d}_rounds',rounds=r)
    for d in (3,5,7):comparison(dataset(f'distance_d{d}',d=d,domain='common9'),'distance',f'd{d}')
    comparison(data,'density','d05_pitch1')
    comparison(dataset('density_d15',d=15,pitch=1/3),'density','d15_pitch0.333')
    summarize()
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
