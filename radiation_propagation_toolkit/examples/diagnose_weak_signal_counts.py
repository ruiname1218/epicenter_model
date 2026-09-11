"""Privileged count-scale diagnostic; never feeds operational inference."""
import copy
import sys
from pathlib import Path

import numpy as np

from qp_ode_simulator.api import run_simulation
from qp_ode_simulator.dataset import event_configuration,_write_json
from qp_ode_simulator.fault_response import sample
from qp_ode_simulator.weak_information_study import SEED,NTRAIN,NVAL,NTEST,CONDITIONS,fields,read,bin_means

root=Path(sys.argv[1]).resolve();p=read(root/'protocol.json')
config,_=event_configuration(p['base'],p['sampling'],0,SEED+10,123)
sim=run_simulation(config,coords_mm=np.asarray(p['geometry']['union_mm']))
quiet=copy.copy(sim);quiet.physics=dict(sim.physics)
for key in ('t1_us','t2_us'):quiet.physics[key]=np.broadcast_to(sim.physics[key][:,:1],sim.physics[key].shape).copy()
result={}
for condition in ('nominal','quiet10'):
    _,expected=sample(p['response'],p['circuits'][condition],*fields(quiet,p,condition),seed=SEED+99,shots=1)
    with np.load(root/f'calibration_{condition}.npz') as z:bits=z['bits']
    for prefix,(name,ticks) in CONDITIONS.items():
        if name!=condition:continue
        lengths=np.diff(np.linspace(0,ticks,17,dtype=int));baseline=bin_means(expected[:,:ticks])
        increases=[]
        for i in range(NTRAIN+NVAL,NTRAIN+NVAL+NTEST):
            with np.load(root/'events'/f'{i:04d}.npz') as z:
                increases.append(float(np.sum((z['latent_'+prefix]-baseline)*lengths)))
        counts=bits[:,:,:ticks].sum((1,2));sd=float(counts.std(ddof=1))
        result[prefix]=dict(baseline_expected_total=float(np.sum(baseline*lengths)),calibration_total_sd=sd,
            extra_expected_count_quantiles=np.quantile(increases,[.1,.5,.9]).tolist(),
            total_count_snr_quantiles=(np.quantile(increases,[.1,.5,.9])/sd).tolist())
_write_json(root/'signal_diagnostic.json',dict(diagnostic_only=True,scope='Expected total detector count increase vs quiet shot-to-shot total count SD. Spatial/time information not captured; not an information bound.',results=result))
print(result)
