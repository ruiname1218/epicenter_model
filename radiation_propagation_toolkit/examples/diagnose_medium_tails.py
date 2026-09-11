"""Exploratory paired tail/8-to16ms checks, never used for model selection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from qp_ode_simulator.medium_inverse import read,_write_json


def main(root):
    frame=pd.read_csv(root/'predictions.csv');s=read(root/'selection.json');out=[]
    pairs=[(s['primary'],'reference'),('n864_h16_binomial','n864_h8_binomial'),
        ('n864_h16_GLS','n864_h8_GLS'),('n864_h16_GLS','n864_h16_binomial')]
    metrics={'mean_mm':lambda a:a.mean(-1),'p90_mm':lambda a:np.quantile(a,.9,axis=-1),
        'within_1mm':lambda a:(a<=1).mean(-1),'over_3mm':lambda a:(a>3).mean(-1)}
    for name,ref in pairs:
        a=frame[frame.model==name].sort_values(['event_uid','shot']);b=frame[frame.model==ref].sort_values(['event_uid','shot'])
        assert a[['event_uid','shot']].reset_index(drop=True).equals(b[['event_uid','shot']].reset_index(drop=True))
        x=a.error_mm.to_numpy().reshape(-1,2);y=b.error_mm.to_numpy().reshape(-1,2)
        idx=np.random.default_rng(20260911339).integers(0,len(x),(10000,len(x)))
        bx=x[idx].reshape(10000,-1);by=y[idx].reshape(10000,-1)
        for metric,fn in metrics.items():
            boot=fn(bx)-fn(by);out.append(dict(model=name,reference=ref,metric=metric,
                difference=float(fn(x.reshape(-1))-fn(y.reshape(-1))),ci95=np.quantile(boot,[.025,.975]).tolist(),exploratory=True))
    _write_json(root/'tail_and_8to16_exploratory.json',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
