"""Experimental exact Pauli-fault response cache for fixed surface-code memory.

Clifford propagation is linear in Pauli faults. A Y fault is the XOR of the X
and Z responses. This samples categorical *faults*, NOT independent detectors.
Only interior detector events are supported; no logical/measurement output.
"""
import copy
from functools import lru_cache
import json

import numpy as np

from .stim_qec import build_stim_layout,round_tick_counts,inject_scheduled_gate_slice_pauli_channels


def interior(layout,rounds):
    coords=layout.detector_coords
    keep=(coords[:,2]>=1)&(coords[:,2]<rounds)
    sites=np.unique(coords[keep,:2],axis=0)
    order=np.concatenate([np.flatnonzero(keep & np.all(coords[:,:2]==s,axis=1)) for s in sites])
    if len(order)!=len(sites)*(rounds-1):raise ValueError('nonuniform interior checks')
    return sites,order


def probe(layout,counts,source_round,slice_index,qubit,basis,sites):
    shape=(len(counts),int(counts.max()),len(layout.qubit_ids))
    xyz=[np.zeros(shape) for _ in range(3)];xyz[0 if basis=='x' else 2][source_round,slice_index,qubit]=.125
    circuit=inject_scheduled_gate_slice_pauli_channels(layout.circuit,layout.qubit_ids,*xyz,counts)
    dem=circuit.detector_error_model(approximate_disjoint_errors=False).flattened()
    errors=[i for i in dem if i.type=='error']
    if len(errors)>1:raise ValueError('single fault yielded multiple independent mechanisms')
    signatures=[]
    for instruction in errors:
        for target in instruction.targets_copy():
            if not target.is_relative_detector_id():continue
            coord=layout.detector_coords[target.val]
            if not 1<=coord[2]<len(counts):continue
            site=np.flatnonzero(np.all(sites==coord[:2],axis=1))
            if len(site)!=1:raise ValueError('unknown check')
            lag=int(coord[2])-source_round
            if lag<0 or lag>2:raise ValueError('unsupported propagation horizon')
            signatures.append([int(site[0]),lag])
    return sorted(signatures)


def build_response(config):
    if config.get('task')!='surface_code:rotated_memory_z' or config['distance'] not in (3,5,7):
        raise ValueError('response cache supports d3/d5/d7 rotated memory Z only')
    short=copy.deepcopy(config);short['rounds']=6;short['circuit_noise']={k:0. for k in short.get('circuit_noise',{})}
    layout=build_stim_layout(short);counts=round_tick_counts(layout.circuit,6);sites,_=interior(layout,6)
    response=dict(sites=sites.tolist(),qubit_ids=layout.qubit_ids.tolist(),first=[],steady=[])
    for kind,r in [('first',0),('steady',2)]:
        for s in range(counts[r]):
            for q in range(len(layout.qubit_ids)):
                response[kind].append(dict(slice=s,qubit=q,
                    x=probe(layout,counts,r,s,q,'x',sites),z=probe(layout,counts,r,s,q,'z',sites)))
    # Check both the next startup round and the last observed round; boundary
    # responses must be the corresponding clipped stationary response.
    for r in [1,5]:
        for item in response['steady']:
            for basis in ['x','z']:
                expected=[v for v in item[basis] if 1<=r+v[1]<6]
                actual=probe(layout,counts,r,item['slice'],item['qubit'],basis,sites)
                if actual!=expected:raise ValueError('nonstationary fault response')
    return response


@lru_cache(maxsize=4)
def layout_cache(config_json):
    config=json.loads(config_json);layout=build_stim_layout(config)
    sites,order=interior(layout,config['rounds'])
    # Residual circuit faults only (no inserted T1/T2 channels). Require an
    # exact DEM conversion here; categorical Pauli channels are handled below.
    dem=layout.circuit.detector_error_model(approximate_disjoint_errors=False).flattened()
    parity=np.ones(layout.circuit.num_detectors)
    for instruction in dem:
        if instruction.type!='error':continue
        coefficient=1-2*instruction.args_copy()[0]
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():parity[target.val]*=coefficient
    baseline=((1-parity)/2)[order].reshape(len(sites),config['rounds']-1)
    return layout,sites,order,baseline


def _ranges(rounds,kind,lag):
    start,stop=(0,1) if kind=='first' else (1,rounds)
    lo=max(start,1-lag);hi=min(stop,rounds-lag)
    return lo,hi


def fault_bits(response,x_fault,z_fault):
    x_fault=np.asarray(x_fault,dtype=bool);z_fault=np.asarray(z_fault,dtype=bool)
    if x_fault.shape!=z_fault.shape or x_fault.ndim!=4:raise ValueError('fault axes must be shot,round,slice,qubit')
    shots,rounds,_,_=x_fault.shape;result=np.zeros((shots,len(response['sites']),rounds-1),dtype=np.uint8)
    for kind in ['first','steady']:
        for item in response[kind]:
            s,q=item['slice'],item['qubit']
            for basis,mask in [('x',x_fault),('z',z_fault)]:
                for check,lag in item[basis]:
                    lo,hi=_ranges(rounds,kind,lag)
                    if hi>lo:result[:,check,lo+lag-1:hi+lag-1]^=mask[:,lo:hi,s,q]
    return result


def exact_marginals(response,px,py,pz,baseline):
    """Exact detector marginals of the supplied categorical Pauli probabilities.

    Factors combine per independent *fault location*. Correlations between
    different detectors remain present in sampling and are not discarded.
    """
    rounds=px.shape[0];parity=1-2*np.asarray(baseline,dtype=np.float64)
    for kind in ['first','steady']:
        for item in response[kind]:
            s,q=item['slice'],item['qubit'];xs=set(map(tuple,item['x']));zs=set(map(tuple,item['z']))
            for check,lag in xs|zs:
                lo,hi=_ranges(rounds,kind,lag)
                if hi<=lo:continue
                # An observable anticommutes with exactly two of X/Y/Z.
                if (check,lag) in xs and (check,lag) in zs:a,b=px,pz
                elif (check,lag) in xs:a,b=px,py
                else:a,b=py,pz
                probability=a[lo:hi,s,q].astype(np.float64)+b[lo:hi,s,q].astype(np.float64)
                parity[check,lo+lag-1:hi+lag-1]*=1-2*probability
    return (1-parity)/2


def sample(response,config,px,py,pz,seed,shots=2):
    config=copy.deepcopy(config);config.pop('seed',None)
    layout,sites,order,baseline=layout_cache(json.dumps(config,sort_keys=True))
    np.testing.assert_array_equal(sites,response['sites'])
    np.testing.assert_array_equal(layout.qubit_ids,response['qubit_ids'])
    px,py,pz=[np.asarray(p,dtype=np.float64) for p in (px,py,pz)]
    if px.shape!=py.shape or px.shape!=pz.shape or px.shape[0]!=config['rounds']:
        raise ValueError('wrong Pauli field dimensions')
    if not all(np.isfinite(p).all() and np.all(p>=0) for p in (px,py,pz)) or np.any(px+py+pz>1+1e-7):
        raise ValueError('invalid categorical channel')
    seeds=np.random.SeedSequence(seed).generate_state(2,dtype=np.uint64)
    uniforms=np.random.default_rng(int(seeds[0])).random((shots,*px.shape))
    bits=fault_bits(response,uniforms<px+py,(uniforms>=px)&(uniforms<px+py+pz))
    residual=layout.circuit.compile_detector_sampler(seed=int(seeds[1])%(2**63-1)).sample(shots=shots)
    bits^=residual[:,order].reshape(bits.shape).astype(np.uint8)
    return bits,exact_marginals(response,px,py,pz,baseline)
