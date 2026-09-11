"""Marginal coverage diagnostic; not a joint 2D confidence guarantee."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


root=Path(__file__).resolve().parents[2]/'localization_runs/attention_mdn_gp_20260908/confirmation'
predictions=pd.read_csv(root/'predictions.csv')
lookup=predictions.groupby(['event_uid','shot'])[['true_x_mm','true_y_mm','strength_band']].first()
scores=[]
for path in sorted(root.glob('*distribution.npz')):
    data=np.load(path)
    keys=pd.MultiIndex.from_arrays([data['event_uid'],data['shot']],names=['event_uid','shot'])
    rows=lookup.loc[keys]
    truth=rows[['true_x_mm','true_y_mm']].to_numpy()
    if 'weights' in data:
        cdf=np.sum(data['weights'][:,:,None]*ndtr((truth[:,None,:]-data['means'])/data['std']),axis=1)
    else:
        cdf=ndtr((truth-data['mean'])/data['std'])
    hit=(cdf>=.025)&(cdf<=.975)
    for band in ['all',0,1,2]:
        mask=np.ones(len(rows),bool) if band=='all' else rows.strength_band.to_numpy()==band
        scores.append(dict(file=path.name,strength_band=band,samples=int(mask.sum()),
            marginal_95_coverage=float(hit[mask].mean()),x_coverage=float(hit[mask,0].mean()),
            y_coverage=float(hit[mask,1].mean()),both_marginals_coverage=float(hit[mask].all(1).mean())))
(root/'marginal_coverage.json').write_text(json.dumps(scores,indent=2)+'\n')
print(pd.DataFrame(scores).to_string(index=False))
