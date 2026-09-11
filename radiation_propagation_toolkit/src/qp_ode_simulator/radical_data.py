"""Fresh paired physical layouts and privileged targets for bounded pilot.

Ordinary inference caches contain observed counts/coincidences only. Conditional
detector means and true nuisance parameters are explicitly separate targets.
"""
import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from .api import run_simulation
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .distance_study import geometry,fields_to_pauli
from .fault_response import build_response,sample
from .localization import _hash_file
from .specialist_study import read

TRAIN_SEED=20260911091
TEST_SEEDS=(20260911101,20260911201,20260911301)


def spec(i):
    if not 0<=i<1548:raise ValueError('invalid event')
    if i<1008:return TRAIN_SEED,i,'train' if i<864 else 'validation'
    block,local=divmod(i-1008,180)
    return TEST_SEEDS[block],local,'test'


def bin_rates(raw,bins):
    edges=np.linspace(0,raw.shape[-1],bins+1,dtype=int)
    return np.stack([raw[...,a:b].mean(-1) for a,b in zip(edges[:-1],edges[1:])],-1).astype(np.float32)


def local_coincidences(raw,sites):
    d=np.linalg.norm(np.asarray(sites)[:,None]-np.asarray(sites)[None,:],axis=-1);d[d==0]=np.inf
    # Geometric nearest-neighbor coincidence channels, not an exact CX graph.
    neighbor=np.argsort(d,axis=1)[:,:3]
    product=np.mean([raw*raw[:,neighbor[:,k],:] for k in range(3)],axis=0)
    return bin_rates(product,128)


def relative_features(raw,quiet):
    edges=[0,339,908,1477,2047,3071,4095]
    rates=np.stack([raw[...,a:b].mean(-1) for a,b in zip(edges[:-1],edges[1:])],-1)
    positive=np.maximum(rates[:,:,1:]-np.asarray(quiet)[None,:,None],0)
    return (positive/(positive.mean(1,keepdims=True)+.003)).reshape(len(raw),-1).astype(np.float32)


def nuisance(rows):
    """Generator parameters only, no center or latent response summaries."""
    cols=['qp_generation_scale_per_us','initial_lambda_mm','maximum_lambda_mm','maximum_distance_mm',
          'apparent_speed_m_per_s','diffusion_coefficient_mm2_per_ms','qp_source_lifetime_ms','qp_trapping_rate_per_us','qp_recombination_rate_per_us']
    arrays=[]
    for k in cols:
        values=rows[k].to_numpy(dtype=float)
        # The unused propagation-law parameter is stored as null, not zero.
        if k=='apparent_speed_m_per_s':values=np.where(rows.propagation_law=='diffusive',0,values)
        if k=='diffusion_coefficient_mm2_per_ms':values=np.where(rows.propagation_law=='ballistic',0,values)
        arrays.append(np.log10(np.maximum(values,1e-20)))
    for k in ['axis_ratio','event_onset_ms','front_width_ms','spread_time_ms']:
        arrays.append(rows[k].to_numpy(dtype=float))
    angle=np.deg2rad(rows.angle_degrees.to_numpy(dtype=float))*2
    arrays.extend([np.sin(angle),np.cos(angle),(rows.geometry=='elliptical').to_numpy(dtype=float),(rows.propagation_law=='diffusive').to_numpy(dtype=float)])
    values=np.column_stack(arrays)
    if not np.isfinite(values).all():raise ValueError('invalid nuisance parameters')
    return values.astype(np.float32)


def prepare(root):
    root.mkdir(exist_ok=False);(root/'events').mkdir()
    inherited=read(root.parent/'feature_selection_20260910/protocol.json');circuits={}
    for name,d,pitch,center in [('base',5,1.,[0.,0.]),('dense',7,5/7,[0.,0.]),('right',5,1.,[11.,0.])]:
        c=copy.deepcopy(inherited['circuit']);c['distance']=d;c['layout'].update(qubit_pitch_mm=pitch,center_mm=center);circuits[name]=c
    layouts={name:geometry(c)[0] for name,c in circuits.items()}
    union=np.unique(np.concatenate([np.round(l.physical_coords_mm,9) for l in layouts.values()]),axis=0)
    indices={name:[int(np.flatnonzero(np.all(union==np.round(q,9),axis=1))[0]) for q in l.physical_coords_mm] for name,l in layouts.items()}
    assert not set(map(tuple,layouts['base'].physical_coords_mm))&set(map(tuple,layouts['right'].physical_coords_mm))
    for axis in range(2):np.testing.assert_allclose([layouts['base'].physical_coords_mm[:,axis].min(),layouts['base'].physical_coords_mm[:,axis].max()],
        [layouts['dense'].physical_coords_mm[:,axis].min(),layouts['dense'].physical_coords_mm[:,axis].max()])
    p=dict(base=inherited['base'],sampling=inherited['sampling'],circuits=circuits,union=union.tolist(),indices=indices,
        sites={name:geometry(c)[1].tolist() for name,c in circuits.items()},physical={n:l.physical_coords_mm.tolist() for n,l in layouts.items()},
        response={'base':inherited['response'],'right':inherited['response'],'dense':build_response(circuits['dense'])},
        training='720 fresh train,120 validation,540 test physical events;2 shots/event. Strong ellipses excluded from fit/validation. Same latent physical realization for all layouts.',
        layouts='base49 qubits, dense97 qubits same coordinate bounding box, pair98 qubits=base+disjoint right d5 patch at11mm; pair expands footprint. Fixed 1us rounds, physical density changes assumed not to change hardware noise.',
        study='Oracle nuisance diagnostics; learned forward-surrogate inversion; temporal network and physical auxiliary targets; density/multiple-patch observations. Bounded pilot, not optimized limits.',
        source_hash=_hash_file(Path(__file__)),dependencies={n:_hash_file(Path(__file__).with_name(n)) for n in ['simulator.py','api.py','stim_qec.py','fault_response.py','distance_study.py','dataset.py']})
    _write_json(root/'protocol.json',p)
    ids={k:[] for k in ['train','validation','test']}
    for i in range(1548):
        seed,local,role=spec(i);c,label=event_configuration(p['base'],p['sampling'],local,seed,123)
        if role=='test' or not (c['shape']['geometry_choices'][0]=='elliptical' and label['strength_band']==2):ids[role].append(i)
    assert [len(ids[k]) for k in ids]==[720,120,540];_write_json(root/'ids.json',ids)


_WORKER=None
def initialize(root):
    global _WORKER
    root=Path(root);_WORKER=root,read(root/'protocol.json')


def event_job(i):
    root,p=_WORKER;seed,local,role=spec(i);c,label=event_configuration(p['base'],p['sampling'],local,seed,123)
    sim=run_simulation(c,coords_mm=np.asarray(p['union']));payload={}
    for name,config in p['circuits'].items():
        xyz=fields_to_pauli(sim,config,np.asarray(p['indices'][name]))
        raw,latent=sample(p['response'][name],config,*xyz,seed=_seed(seed+1,local,['base','dense','right'].index(name))%(2**63-1),shots=2)
        payload[name]=raw
        if name=='base':payload['privileged_mean']=bin_rates(latent,16)
    row=sim.parameters.iloc[0].to_dict();row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(label,event=i,event_uid=f'radical-{seed}:{local}',role=role,test_seed=seed)
    path=root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**payload)
    path.with_suffix('.npz.tmp').replace(path);_write_json(path.with_suffix('.json'),row);return i


def check(root):
    p=read(root/'protocol.json');assert p['source_hash']==_hash_file(Path(__file__))
    for n,h in p['dependencies'].items():assert _hash_file(Path(__file__).with_name(n))==h
    return p


def generate(root,test=False):
    check(root);ids=read(root/'ids.json');ids=ids['test'] if test else ids['train']+ids['validation']
    if test:assert (root/'selection.json').exists()
    with _generation_lock(root):
        done=[i for i in ids if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),)) as pool:
            fs=[pool.submit(event_job,i) for i in ids if i not in done]
            for f in as_completed(fs):
                done.append(f.result())
                if len(done)%60==0:print('Generated',len(done),'/',len(ids),'test',test,flush=True)
        _write_json(root/('test_manifest.json' if test else 'fit_manifest.json'),dict(selection_hash=_hash_file(root/'selection.json') if test else None,
            hashes={f'{i:04d}{ext}':_hash_file(root/'events'/f'{i:04d}{ext}') for i in ids for ext in ['.npz','.json']}))


def calibrate(root):
    p=check(root);c,_=event_configuration(p['base'],p['sampling'],0,TRAIN_SEED+1000,123)
    sim=run_simulation(c,coords_mm=np.asarray(p['union']));quiet={}
    for j,(name,config) in enumerate(p['circuits'].items()):
        xyz=fields_to_pauli(sim,config,np.asarray(p['indices'][name]),quiet=True)
        rates=[]
        for batch in range(8):
            raw,_=sample(p['response'][name],config,*xyz,seed=_seed(TRAIN_SEED+1001,j,batch)%(2**63-1),shots=16)
            rates.append(raw.mean((0,2)))
        quiet[name]=np.mean(rates,axis=0).tolist()
    _write_json(root/'quiet.json',quiet)


def cache(root,test=False):
    p=check(root)
    if not (root/'quiet.json').exists():
        if test:raise ValueError('calibrate before test')
        calibrate(root)
    quiet=read(root/'quiet.json');ids=read(root/'ids.json');ids=ids['test'] if test else ids['train']+ids['validation']
    manifest=read(root/('test_manifest.json' if test else 'fit_manifest.json'))
    if test:assert manifest['selection_hash']==_hash_file(root/'selection.json')
    rows=[];values={}
    for i in ids:
        path=root/'events'/f'{i:04d}.npz'
        for ext in ['.npz','.json']:assert _hash_file(path.with_suffix(ext))==manifest['hashes'][f'{i:04d}{ext}']
        with np.load(path) as z:
            for name in ['base','dense']:
                values.setdefault(name,[]).append(relative_features(z[name],quiet[name]))
            pair=np.concatenate([z['base'],z['right']],axis=1)
            values.setdefault('pair',[]).append(relative_features(pair,quiet['base']+quiet['right']))
            values.setdefault('time',[]).append(bin_rates(z['base'],128))
            values.setdefault('coinc',[]).append(local_coincidences(z['base'],p['sites']['base']))
            values.setdefault('counts',[]).append(bin_rates(z['base'],16))
            values.setdefault('privileged_mean',[]).append(np.repeat(z['privileged_mean'][None],2,axis=0))
        row=read(path.with_suffix('.json'));rows.extend([dict(row,shot=j) for j in range(2)])
    prefix='test' if test else 'fit';rows=pd.DataFrame(rows);rows.to_csv(root/f'{prefix}_rows.csv',index=False)
    for name,v in values.items():np.save(root/f'{prefix}_{name}.npy',np.concatenate(v))
    np.save(root/f'{prefix}_nuisance.npy',nuisance(rows))
    _write_json(root/f'{prefix}_cache.json',dict(hashes={f.name:_hash_file(f) for f in root.glob(f'{prefix}_*.npy')},rows_hash=_hash_file(root/f'{prefix}_rows.csv')))
    print('Cached',prefix,len(rows),'shots',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate_fit','cache_fit','generate_test','cache_test']);parser.add_argument('root',type=Path);a=parser.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage.startswith('generate'):generate(a.root,test=a.stage=='generate_test')
    else:cache(a.root,test=a.stage=='cache_test')
