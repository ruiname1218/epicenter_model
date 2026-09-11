"""Prespecified endpoint details and geometric checks; no model changes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from qp_ode_simulator.rei_fair_models import verify
from qp_ode_simulator.rei_fair_data import read,_write_json


def main(root):
    verify(root);p=read(root/'protocol.json');rows=pd.read_csv(root/'test_rows.csv');f=pd.read_csv(root/'predictions.csv');out=[]
    metrics={'mean_mm':lambda a:a.mean(-1),'p90_mm':lambda a:np.quantile(a,.9,axis=-1),'within_1mm':lambda a:(a<=1).mean(-1),'over_3mm':lambda a:(a>3).mean(-1)}
    sub=f[(f.horizon==4)&(f.domain=='ID')]
    for name,ref in [('SVR','REI_full'),('CNN','REI_full'),('Ridge','REI_full'),('SVR','Prior')]:
        for group,band in [('all',None),('weak',0),('medium',1),('strong',2)]:
            q=sub if band is None else sub[sub.strength_band==band]
            a=q[q.model==name].sort_values(['event_uid','shot']);b=q[q.model==ref].sort_values(['event_uid','shot'])
            assert a[['event_uid','shot']].reset_index(drop=True).equals(b[['event_uid','shot']].reset_index(drop=True))
            x=a.error_mm.to_numpy().reshape(-1,2);y=b.error_mm.to_numpy().reshape(-1,2)
            ix=np.random.default_rng(20260911711).integers(0,len(x),(10000,len(x)))
            bx=x[ix].reshape(10000,-1);by=y[ix].reshape(10000,-1)
            for metric,fn in metrics.items():
                boot=fn(bx)-fn(by);out.append(dict(domain='ID',horizon=4,group=group,model=name,reference=ref,metric=metric,
                    difference=float(fn(x.reshape(-1))-fn(y.reshape(-1))),ci95=np.quantile(boot,[.025,.975]).tolist(),exploratory=not(name=='SVR' and ref=='REI_full' and group=='all' and metric=='mean_mm')))
    _write_json(root/'endpoint_intervals.json',out)
    xy=rows[['epicenter_row','epicenter_col']].to_numpy();geometric={}
    for kind in ['sites','physical']:
        inside=Delaunay(np.asarray(p[kind])).find_simplex(xy,tol=1e-10)>=0
        geometric[kind]=dict(outside_physical_events=int(rows.loc[~inside,'event_uid'].nunique()),total_physical_events=int(rows.event_uid.nunique()))
    _write_json(root/'geometry_audit.json',geometric)
    # Parameter ranges verify that the designated speed stress really is OOD.
    range_rows=[]
    for domain in rows.domain.unique():
        for law in ['ballistic','diffusive']:
            r=rows[(rows.domain==domain)&(rows.propagation_law==law)]
            key='apparent_speed_m_per_s' if law=='ballistic' else 'diffusion_coefficient_mm2_per_ms'
            range_rows.append(dict(domain=domain,law=law,parameter=key,minimum=float(r[key].min()),maximum=float(r[key].max())))
    _write_json(root/'actual_parameter_ranges.json',range_rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
