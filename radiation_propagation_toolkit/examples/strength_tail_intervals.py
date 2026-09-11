"""Supplementary paired event bootstrap for predeclared non-mean endpoints."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from qp_ode_simulator.dataset import _write_json

root=Path(sys.argv[1]).resolve();f=pd.read_csv(root/'predictions.csv');records=[]
for group,band in [('weak',0),('medium',1),('strong',2)]:
    sub=f[f.strength_band==band]
    a=sub[(sub.model=='Overlap')&(sub.training_seed==-1)].pivot(index='event_uid',columns='shot',values='error_mm')
    b=sub[sub.model=='REI'].pivot(index='event_uid',columns='shot',values='error_mm')
    assert a.index.equals(b.index) and a.columns.equals(b.columns)
    av=a.to_numpy();bv=b.to_numpy();rng=np.random.default_rng(2026090947)
    indices=rng.integers(0,len(a),(10000,len(a)))
    aa=av[indices].reshape(10000,-1);bb=bv[indices].reshape(10000,-1)
    for name,fn in [('within_1mm',lambda x:np.mean(x<=1,axis=-1)),('over_3mm',lambda x:np.mean(x>3,axis=-1)),
                    ('p90_mm',lambda x:np.quantile(x,.9,axis=-1)),('p95_mm',lambda x:np.quantile(x,.95,axis=-1))]:
        delta=fn(aa)-fn(bb)
        records.append(dict(group=group,metric=name,difference=float(fn(av.reshape(-1))-fn(bv.reshape(-1))),ci95=np.quantile(delta,[.025,.975]).tolist()))
_write_json(root/'tail_intervals.json',dict(scope='Overlap coordinate-ensemble minus adapted REI; secondary descriptive intervals, paired whole events including both shots.',records=records))
print(pd.DataFrame(records).to_string(index=False))
