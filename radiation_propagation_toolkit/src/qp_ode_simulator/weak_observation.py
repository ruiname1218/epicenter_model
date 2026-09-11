"""Paired weak-event acquisition and physical detector-template experiments."""
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import stim

from .api import run_simulation
from .dataset import event_configuration, _seed, _write_json, _generation_lock
from .stim_qec import build_stim_layout, generate_circuit_syndromes
from .localization import _hash_file


def reduce_background(values, factor):
    """Scale hardware baseline rates, retaining the radiation-induced excess."""
    values=np.asarray(values,dtype=np.float64)
    rate=1/values;baseline=rate[:,:1,:]
    if not 0 < factor <= 1 or np.any(rate < baseline-1e-9):
        raise ValueError('invalid baseline intervention')
    return (1/(rate-(1-factor)*baseline)).astype(np.float32)


def detector_marginals(circuit):
    """Marginals of Stim's approximate-disjoint DEM, NOT an exact likelihood.

    Pauli-channel conversion uses the explicitly recorded small-error
    approximation. Detector independence is a further composite-score assumption.
    Observed syndromes are still sampled from the original Pauli circuit.
    """
    dem=circuit.detector_error_model(approximate_disjoint_errors=True).flattened()
    logparity=np.zeros(circuit.num_detectors,dtype=np.float64)
    for instruction in dem:
        if instruction.type!='error':continue
        p=instruction.args_copy()[0]
        if not 0 <= p < .5:raise ValueError('unexpected DEM probability')
        # Tiny target sets are faster to XOR in Python than to allocate/sort
        # a NumPy array for every DEM instruction (millions per long circuit).
        parity=set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                d=target.val
                if d in parity:parity.remove(d)
                else:parity.add(d)
        value=np.log1p(-2*p)
        for d in parity:logparity[d]+=value
    return -.5*np.expm1(logparity)


def event_job(args):
    event,root,base,plan,circuit_config,role=args
    start=time.monotonic();root=Path(root)
    config,sampled=event_configuration(base,plan,event,20270220,123)
    layout=build_stim_layout(circuit_config)
    simulation=run_simulation(config,coords_mm=layout.physical_coords_mm)
    payload={};row=simulation.parameters.iloc[0].to_dict()
    row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(sampled,event=event,event_uid=f'weak-observation-20270220:{event}',role=role)
    for name,factor in [('nominal',1.),('quiet10',.1)]:
        conf=copy.deepcopy(circuit_config)
        conf['seed']=_seed(20270221,event,0 if name=='nominal' else 1)%(2**63-1)
        conf['circuit_noise']={k:v*factor for k,v in conf['circuit_noise'].items()}
        condition_layout=layout if factor==1 else build_stim_layout(conf)
        t1=reduce_background(simulation.physics['t1_us'],factor)
        t2=reduce_background(simulation.physics['t2_us'],factor)
        arrays,metadata=generate_circuit_syndromes(t1,simulation.time_ms,condition_layout,conf,
            t2_us=t2,event_onset_ms=simulation.parameters.event_onset_ms.to_numpy(),
            baseline_t1_us=t1[:,0,:],baseline_t2_us=t2[:,0,:])
        coords=arrays['detector_coords'];keep=(coords[:,2]>=1)&(coords[:,2]<conf['rounds'])
        sites=np.unique(coords[keep,:2],axis=0)
        order=np.concatenate([np.flatnonzero(keep & np.all(coords[:,:2]==site,axis=1)) for site in sites])
        raw=arrays['detector_events'][0][:,order].reshape(conf['shots_per_event'],len(sites),conf['rounds']-1)
        payload[name]=raw
        if role=='train':
            annotated=stim.Circuit(metadata['first_annotated_circuit'])
            expected=detector_marginals(annotated)[order].reshape(len(sites),conf['rounds']-1)
            payload[name+'_expected']=expected.astype(np.float32)
            del annotated
        payload['sites']=sites
        del metadata,arrays
    payload['event']=np.asarray(event)
    np.savez_compressed(root/f'event_{event:04d}.npz',**payload)
    _write_json(root/f'event_{event:04d}.json',row)
    return dict(event=event,seconds=time.monotonic()-start)


def configuration(runs):
    manifest=json.loads((Path(runs)/'weak_growth_20260908/new_dataset/dataset_manifest.json').read_text())
    base=copy.deepcopy(manifest['simulator_config']);base['time']['end_ms']=8.196
    plan=copy.deepcopy(manifest['sampling']);plan['generation_bands_per_us']=[[1e-10,1e-9]]
    circuit=copy.deepcopy(manifest['stim_config']);circuit.update(rounds=8192,shots_per_event=2)
    return base,plan,circuit


def split_events(base,plan,n=720):
    groups={}
    for i in range(n):
        c,_=event_configuration(base,plan,i,20270220,123)
        key=(c['shape']['geometry_choices'][0],c['propagation']['law_choices'][0],next(iter(c['epicenter']['region_probabilities'])))
        groups.setdefault(key,[]).append(i)
    rng=np.random.default_rng(90701);split=dict(train=[],validation=[],test=[])
    for _,ids in sorted(groups.items()):
        if len(ids)!=60:raise ValueError('expected 60 events per weak stratum')
        ids=rng.permutation(ids).tolist()
        for role,part in [('train',ids[:36]),('validation',ids[36:48]),('test',ids[48:])]:split[role].extend(part)
    return split


def generate(root,resume=False):
    root=Path(root);root.mkdir(exist_ok=resume)
    base,plan,circuit=configuration(root.parent);split=split_events(base,plan)
    from . import stim_qec,simulator,api,syndrome,circuit_building
    protocol=dict(base=base,plan=plan,circuit=circuit,split=split,events=720,workers=20,
        generation_seed=20270220,syndrome_seed=20270221,device_seed=123,
        windows_rounds=[2048,4096,8192],conditions=['nominal','quiet10'],
        background_intervention='baseline 1/T1,1/T2 and residual circuit fault probabilities multiplied by 0.1; radiation excess rates unchanged',
        template_approximation='Stim approximate_disjoint_errors=True plus independent binned detector composite likelihood; not full joint likelihood',
        source_hashes={str(Path(m.__file__).resolve()):_hash_file(Path(m.__file__)) for m in [stim_qec,simulator,api,syndrome,circuit_building]},
        generator_hash=_hash_file(Path(__file__)))
    if resume:
        previous=json.loads((root/'protocol.json').read_text())
        for key in ['base','plan','circuit','split','generation_seed','syndrome_seed','device_seed']:
            if previous[key]!=protocol[key]:raise ValueError('resume settings changed: '+key)
        protocol['resume_history']=previous.get('resume_history',[])+[dict(previous_protocol=previous,
            reason='equivalence-tested circuit batching and DEM target accumulation optimizations; completed event files preserved')]
    _write_json(root/'protocol.json',protocol);(root/'events').mkdir(exist_ok=resume)
    roles={i:role for role,ids in split.items() for i in ids};completed=[]
    if resume:
        for i in range(720):
            a=root/'events'/f'event_{i:04d}.npz';b=root/'events'/f'event_{i:04d}.json'
            if not a.exists() or not b.exists():continue
            row=json.loads(b.read_text())
            if row['event']!=i or row['role']!=roles[i]:raise ValueError('resume event mismatch')
            with np.load(a) as z:
                for condition in ['nominal','quiet10']:
                    if z[condition].shape!=(2,8,8191):raise ValueError('incomplete event')
                    if roles[i]=='train' and z[condition+'_expected'].shape!=(8,8191):raise ValueError('incomplete template')
            completed.append(dict(event=i,seconds=None,preserved_on_resume=True,hashes={'npz':_hash_file(a),'json':_hash_file(b)}))
        print('Preserved',len(completed),'completed paired events',flush=True)
    done={r['event'] for r in completed}
    with _generation_lock(root),ProcessPoolExecutor(max_workers=20) as pool:
        futures=[pool.submit(event_job,(i,str(root/'events'),base,plan,circuit,roles[i])) for i in range(720) if i not in done]
        for future in as_completed(futures):
            record=future.result();i=record['event']
            record['hashes']={suffix:_hash_file(root/'events'/f'event_{i:04d}.{suffix}') for suffix in ['npz','json']}
            completed.append(record)
            if len(completed)%12==0:
                _write_json(root/'progress.json',dict(status='generating',completed=completed))
                print(f'saved {len(completed)}/720 paired events',flush=True)
    _write_json(root/'progress.json',dict(status='complete',completed=completed))


if __name__=='__main__':
    import sys
    generate(sys.argv[1],resume='--resume' in sys.argv[2:])
