"""Disaggregate existing predictions, preserving study/model/split/seed identities."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from qp_ode_simulator.strength_benchmark import summary

root=Path(sys.argv[1]).resolve();out=root/'strength_benchmark_20260909'
records=[];catalog=[]
for path in sorted(root.rglob('predictions.csv')):
    if out in path.parents:continue
    columns=list(pd.read_csv(path,nrows=0).columns)
    source=str(path.relative_to(root))
    if 'error_mm' not in columns:
        catalog.append(dict(source=source,status='Skipped: not coordinate-error predictions'));continue
    keys=[c for c in ['model','candidate','family','arm','distance','size','seed','setting','split','case'] if c in columns]
    names=list(dict.fromkeys(keys+[c for c in ['error_mm','strength_band','geometry','event_uid','event','shot'] if c in columns]))
    f=pd.read_csv(path,usecols=names)
    if 'strength_band' not in f:
        if source.startswith(('weak_information_','weak_observation_')):f['strength_band']=0
        else:catalog.append(dict(source=source,status='Skipped: no strength labels'));continue
    uid='event_uid' if 'event_uid' in f else 'event'
    group_cols=keys+['strength_band']
    for values,a in f.groupby(group_cols,dropna=False):
        info=dict(zip(group_cols,values if isinstance(values,tuple) else (values,)))
        info['strength']={0:'weak',1:'medium',2:'strong'}.get(info.pop('strength_band'),'unknown')
        for shape in ['both']+(['circular','elliptical'] if 'geometry' in a else []):
            b=a if shape=='both' else a[a.geometry==shape]
            if len(b)==0:continue
            records.append(dict(source=source,shape=shape,events=b[uid].nunique(),shots=len(b),**info,**summary(b.error_mm)))
    catalog.append(dict(source=source,status='Aggregated separately; do not pool across studies',prediction_rows=len(f)))
pd.DataFrame(records).to_csv(out/'historical_strength_metrics.csv',index=False)
pd.DataFrame(catalog).to_csv(out/'historical_catalog.csv',index=False)
print(pd.DataFrame(catalog).to_string(index=False));print('Summary rows',len(records))
