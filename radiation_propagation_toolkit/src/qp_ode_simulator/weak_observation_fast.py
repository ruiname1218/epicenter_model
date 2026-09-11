"""Resume paired acquisition with tested linear Pauli-fault propagation.

Previously sampled bits are preserved. All training marginal templates are
recomputed with the exact categorical-channel parity formula.
"""
import copy
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np

from .api import run_simulation
from .dataset import event_configuration,_seed,_write_json,_generation_lock
from .localization import _hash_file
from .weak_observation import reduce_background
from .stim_qec import average_t1_over_rounds,round_tick_counts,gate_slice_durations_ms,scheduled_gate_slice_pauli_probabilities
from .fault_response import build_response,layout_cache,sample


def condition_config(config,factor):
    config=copy.deepcopy(config);config.pop('seed',None)
    config['circuit_noise']={k:v*factor for k,v in config['circuit_noise'].items()}
    radiation=config['radiation_channel']
    if radiation['targets']!='all' or radiation['timing']!='gate_slices' or radiation.get('coherent_phase_approximation',{}).get('enabled',False):
        raise ValueError('fast experiment supports all-qubit dissipative gate-slice channels only')
    return config


def event_job(args):
    event,root,base,plan,circuit_config,role,response,preserve=args
    root=Path(root);start=time.monotonic()
    config,sampled=event_configuration(base,plan,event,20270220,123)
    nominal=condition_config(circuit_config,1.)
    layout,sites,_,_=layout_cache(json.dumps(nominal,sort_keys=True))
    simulation=run_simulation(config,coords_mm=layout.physical_coords_mm)
    row=simulation.parameters.iloc[0].to_dict()
    row={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()}
    row.update(sampled,event=event,event_uid=f'weak-observation-20270220:{event}',role=role)
    saved={};archive_hashes={}
    if preserve:
        path=root/f'event_{event:04d}.npz';label=root/f'event_{event:04d}.json'
        old=json.loads(label.read_text())
        for key in ['event_uid','role','generation_seed','epicenter_row','epicenter_col','event_onset_ms']:
            if row[key]!=old[key]:raise ValueError('physical event changed on replay: '+key)
        with np.load(path) as z:
            saved={name:z[name].copy() for name in ['nominal','quiet10']}
        archive=root.parent/'legacy_files';archive.mkdir(exist_ok=True)
        for original in [path,label]:
            target=archive/original.name
            if target.exists():raise ValueError('refusing to overwrite legacy archive')
            shutil.copy2(original,target);archive_hashes[original.name]=_hash_file(target)
    payload={'event':np.asarray(event),'sites':sites}
    rounds=circuit_config['rounds'];duration=circuit_config['round_duration_ms']
    starts=circuit_config['start_time_ms']+np.arange(rounds)*duration
    counts=round_tick_counts(layout.circuit,rounds)
    durations=gate_slice_durations_ms(duration,counts,circuit_config['gate_schedule'])
    for name,factor in [('nominal',1.),('quiet10',.1)]:
        conf=condition_config(circuit_config,factor)
        averaged=[]
        for key in ['t1_us','t2_us']:
            values=reduce_background(simulation.physics[key],factor)
            averaged.append(average_t1_over_rounds(values,simulation.time_ms,starts,duration))
        pauli=scheduled_gate_slice_pauli_probabilities(averaged[0],durations,counts,round_t2_us=averaged[1])
        bits,expected=sample(response,conf,*[pauli[k][0] for k in ['x','y','z']],
            seed=_seed(20270221,event,0 if name=='nominal' else 1)%(2**63-1),shots=2)
        payload[name]=saved.get(name,bits)
        if role=='train':payload[name+'_expected']=expected.astype(np.float32)
    row['sampling_backend']='legacy_stim_preserved' if preserve else 'exact_pauli_fault_response'
    temporary=root/f'event_{event:04d}.npz.tmp'
    with temporary.open('wb') as stream:np.savez_compressed(stream,**payload)
    temporary.replace(root/f'event_{event:04d}.npz')
    _write_json(root/f'event_{event:04d}.json',row)
    return dict(event=event,seconds=time.monotonic()-start,backend=row['sampling_backend'],archived=archive_hashes)


def generate(root):
    root=Path(root);old=json.loads((root/'protocol.json').read_text())
    base,plan,circuit=old['base'],old['plan'],old['circuit'];split=old['split']
    # Build once, fork immutable caches into workers.
    response=build_response(circuit)
    for factor in [1.,.1]:layout_cache(json.dumps(condition_config(circuit,factor),sort_keys=True))
    _write_json(root/'fault_response.json',response)
    roles={i:r for r,ids in split.items() for i in ids};existing=[]
    for i in range(720):
        a=root/'events'/f'event_{i:04d}.npz';b=root/'events'/f'event_{i:04d}.json'
        if a.exists() and b.exists():
            label=json.loads(b.read_text())
            if label['event']!=i or label['role']!=roles[i]:raise ValueError('existing role mismatch')
            with np.load(a) as z:
                for condition in ['nominal','quiet10']:
                    if z[condition].shape!=(2,8,8191):raise ValueError('bad completed sample')
            existing.append(i)
    from . import fault_response,stim_qec,simulator,api,syndrome,circuit_building
    protocol=copy.deepcopy(old)
    protocol.update(previous_protocol=old,sampling_backend='exact linear categorical Pauli fault response plus independent residual Stim sampler',
        preserved_sample_events=existing,template_approximation='exact single-detector categorical Pauli marginals; binned independent or GLS scoring remains approximate',
        response_sha256=_hash_file(root/'fault_response.json'),fast_generator_hash=_hash_file(Path(__file__)),
        fast_source_hashes={str(Path(m.__file__).resolve()):_hash_file(Path(m.__file__)) for m in [fault_response,stim_qec,simulator,api,syndrome,circuit_building]},
        backend_validation='64 deterministic multifault patterns exact; 20000-shot marginal and five-detector parity agreement; same-seed reproducibility; first/interior/last-round probes')
    _write_json(root/'protocol.json',protocol)
    completed=[]
    # Query files have no latent templates and need not be rewritten.
    for i in existing:
        if roles[i]=='train':continue
        completed.append(dict(event=i,seconds=None,backend='legacy_stim_preserved',hashes={suffix:_hash_file(root/'events'/f'event_{i:04d}.{suffix}') for suffix in ['npz','json']}))
    done={r['event'] for r in completed}
    print('Preserving',len(existing),'previously sampled events; recalculating training templates exactly',flush=True)
    with _generation_lock(root),ProcessPoolExecutor(max_workers=20) as pool:
        futures=[pool.submit(event_job,(i,str(root/'events'),base,plan,circuit,roles[i],response,i in existing)) for i in range(720) if i not in done]
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
    generate(sys.argv[1])
