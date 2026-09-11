"""Fresh paired-window benchmark, with no latent probability learning targets."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .radical_data import read,bin_rates
from .distance_study import fields_to_pauli
from .api import run_simulation
from .fault_response import sample
from .localization import _hash_file

TRAIN_SEED=20260911501
ID_SEEDS=(20260911511,20260911521,20260911531)
SLOW_SEEDS=(20260911541,20260911551,20260911561)
HORIZONS=(2,4,8)


def feature_edges(h):
    if h not in HORIZONS:raise ValueError('unknown horizon')
    e=[0,339,908,1477,2047]
    if h>=4:e.extend([3071,4095])
    if h>=8:e.extend([6143,8191])
    return e


def features(raw,h,quiet=None):
    e=feature_edges(h);r=np.stack([raw[...,a:b].mean(-1) for a,b in zip(e[1:-1],e[2:])],-1)
    p=r if quiet is None else np.maximum(r-np.asarray(quiet)[None,:,None],0)
    return (p/(p.mean(1,keepdims=True)+.003)).reshape(len(raw),-1).astype(np.float32)


def prepare(root):
    root.mkdir(exist_ok=False);(root/'events').mkdir()
    old=read(root.parent/'radical_pilot_20260911/protocol.json')
    base=copy.deepcopy(old['base']);base['time']['end_ms']=8.196
    circuit=copy.deepcopy(old['circuits']['base']);circuit['rounds']=8192
    noisy=copy.deepcopy(circuit)
    for k,v in noisy['circuit_noise'].items():noisy['circuit_noise'][k]=v*2
    p=dict(base=base,circuit=circuit,noisy_circuit=noisy,coords=old['union'],indices=old['indices']['base'],response=old['response']['base'],
        sampling=old['sampling'],sites=old['sites']['base'],physical=old['physical']['base'],
        design='All36 strata included. Train864,validation180. ID test540=3rootsx180 plus same540 physical events observed with doubled circuit noise; slowOOD180=3rootsx60.2 shots/event. Prefix2/4/8ms of same8.192ms trace.',
        primary='4ms ID SVR minus full-window Adapted_REI, mean Euclidean error over all three strength bands. One primary comparison,95% paired event bootstrap. Every other contrast exploratory.',
        fitting='Models see observed syndrome plus xy labels only; no strength/shape auxiliary loss or gate, no latent probability targets. Train-only scaling, independent empirical quiet calibration. Validation selects mean over ALL events, no strength-specific tuning.',
        rei='Fixed Algorithm1-inspired adapted implementation; correlation_multiplier2,circuit_repetitions1. Primary history=all available internal rounds; sensitivity last1024 and validation-selected K in256/512/1024/full, fixed separately per horizon. Not an original REI reproduction.',
        stress='ID physical events paired with circuit-noise-x2; nominal calibration remains frozen. SlowOOD ballistic speed2..6m/s vs train12..40, diffusion0.5..2mm2/ms vs train5..20. Same fixed device,geometry and strength distributions.',
        source_hash=_hash_file(Path(__file__)),dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['dataset.py','radical_data.py','distance_study.py','fault_response.py','api.py','simulator.py','stim_qec.py','temporal_diagnosis.py']})
    specs=[]
    for i in range(1044):specs.append(dict(event=len(specs),local=i,seed=TRAIN_SEED,role='train' if i<864 else 'validation',domain='ID'))
    for domain,seeds,count in [('ID',ID_SEEDS,180),('slow',SLOW_SEEDS,60)]:
        # Slow test uses an even number of complete36-stratum cycles overall:
        # 3 roots x72=216 rather than cutting a stratification cycle at60.
        count=180 if domain=='ID' else 72
        for seed in seeds:
            for i in range(count):specs.append(dict(event=len(specs),local=i,seed=seed,role='test',domain=domain))
    p['design']=p['design'].replace('slowOOD180=3rootsx60','slowOOD216=3rootsx72')
    _write_json(root/'protocol.json',p);_write_json(root/'specs.json',specs)


def check(root):
    p=read(root/'protocol.json');assert p['source_hash']==_hash_file(Path(__file__))
    for n,h in p['dependencies'].items():assert _hash_file(Path(__file__).with_name(n))==h
    return p


def configuration(p,spec):
    c,label=event_configuration(p['base'],p['sampling'],spec['local'],spec['seed'],123)
    if spec['domain']=='slow':
        c['propagation'].update(apparent_speed_m_per_s=[2.,6.],ballistic_speed_m_per_s=[2.,6.],diffusion_coefficient_mm2_per_ms=[.5,2.])
    return c,label


_WORKER=None
def initialize(root):
    global _WORKER
    root=Path(root);_WORKER=root,read(root/'protocol.json')


def event_job(spec):
    root,p=_WORKER;c,label=configuration(p,spec);sim=run_simulation(c,coords_mm=np.asarray(p['coords']))
    xyz=fields_to_pauli(sim,p['circuit'],np.asarray(p['indices']))
    raw,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(spec['seed']+1,spec['local'],5)%(2**63-1),shots=2)
    payload={'raw':raw};assert raw.shape==(2,24,8191)
    if spec['role']=='test' and spec['domain']=='ID':
        raw2,_=sample(p['response'],p['noisy_circuit'],*xyz,seed=_seed(spec['seed']+2,spec['local'],5)%(2**63-1),shots=2)
        payload['noise2']=raw2
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(label,**spec,event_uid=f'fair-{spec["seed"]}:{spec["local"]}')
    path=root/'events'/f'{spec["event"]:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**payload)
    path.with_suffix('.npz.tmp').replace(path);_write_json(path.with_suffix('.json'),row);return spec['event']


def generate(root,test=False):
    check(root);specs=[s for s in read(root/'specs.json') if (s['role']=='test')==test]
    selection_hash=_hash_file(root/'selection.json') if test else None
    with _generation_lock(root):
        done=[s['event'] for s in specs if all((root/'events'/f'{s["event"]:04d}{ext}').exists() for ext in ['.npz','.json'])]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            fs=[pool.submit(event_job,s) for s in specs if s['event'] not in done]
            for f in as_completed(fs):
                done.append(f.result())
                if len(done)%72==0:print('Generated',len(done),'/',len(specs),'test',test,flush=True)
        if test:assert _hash_file(root/'selection.json')==selection_hash
        _write_json(root/('test_manifest.json' if test else 'fit_manifest.json'),dict(selection_hash=selection_hash,
            hashes={f'{s["event"]:04d}{ext}':_hash_file(root/'events'/f'{s["event"]:04d}{ext}') for s in specs for ext in ['.npz','.json']}))


def calibrate(root):
    p=check(root);c,_=event_configuration(p['base'],p['sampling'],0,TRAIN_SEED+1000,123)
    sim=run_simulation(c,coords_mm=np.asarray(p['coords']));xyz=fields_to_pauli(sim,p['circuit'],np.asarray(p['indices']),quiet=True)
    rates=[]
    for j in range(8):
        raw,_=sample(p['response'],p['circuit'],*xyz,seed=_seed(TRAIN_SEED+1001,j,0)%(2**63-1),shots=16)
        rates.append(raw.mean((0,2)))
    _write_json(root/'quiet.json',dict(rates=np.mean(rates,axis=0).tolist(),shots=128,source='independent nominal quiet syndrome samples; latent marginals discarded'))


def cache(root,test=False):
    check(root)
    if not (root/'quiet.json').exists():
        if test:raise ValueError('no frozen calibration')
        calibrate(root)
    quiet=read(root/'quiet.json')['rates'];m=read(root/('test_manifest.json' if test else 'fit_manifest.json'));rows=[];data={}
    if test:assert m['selection_hash']==_hash_file(root/'selection.json')
    for spec in read(root/'specs.json'):
        if (spec['role']=='test')!=test:continue
        path=root/'events'/f'{spec["event"]:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==m['hashes'][f'{spec["event"]:04d}{ext}']
        row=read(path.with_suffix('.json'))
        with np.load(path) as z:
            for key in ['raw','noise2'] if test and spec['domain']=='ID' else ['raw']:
                raw=z[key]
                for h in HORIZONS:
                    data.setdefault(f'x{h}',[]).append(features(raw,h,quiet));data.setdefault(f'u{h}',[]).append(features(raw,h))
                    data.setdefault(f't{h}',[]).append(bin_rates(raw[...,:h*1024-1],128))
                    for k in sorted(set([256,512,1024,h*1024-1])):
                        data.setdefault(f'r{h}_{k}',[]).append(raw[...,h*1024-1-k:h*1024-1].mean(-1).astype(np.float64))
                rows.extend([dict(row,shot=j,domain='noise2' if key=='noise2' else spec['domain']) for j in range(2)])
    prefix='test' if test else 'fit';pd.DataFrame(rows).to_csv(root/f'{prefix}_rows.csv',index=False)
    for name,v in data.items():np.save(root/f'{prefix}_{name}.npy',np.concatenate(v))
    _write_json(root/f'{prefix}_cache.json',dict(hashes={f.name:_hash_file(f) for f in root.glob(f'{prefix}_*.npy')},rows_hash=_hash_file(root/f'{prefix}_rows.csv')))
    print('Cached',prefix,len(rows),'condition-shots',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate_fit','generate_test','cache_fit','cache_test','calibrate']);p.add_argument('root',type=Path);a=p.parse_args()
    if a.stage.startswith('generate'):generate(a.root,a.stage=='generate_test')
    elif a.stage.startswith('cache'):cache(a.root,a.stage=='cache_test')
    else:globals()[a.stage](a.root)
