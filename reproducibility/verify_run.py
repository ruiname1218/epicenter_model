"""Check a fresh run, not just the published prediction tables."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from bootstrap import ROOT

def verify(out):
    metadata=json.loads((out/'run_metadata.json').read_text());cfg=metadata['config']
    assert metadata['status']=='completed'
    if (out/'manifest.json').exists():
        for name,expected in json.loads((out/'manifest.json').read_text()).items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected,name
    pred=pd.read_csv(out/'predictions.csv');metrics=pd.read_csv(out/'metrics.csv')
    if cfg['study']=='leakage':
        np.testing.assert_allclose(pred.error,np.hypot(pred.pred_x-pred.target_x,pred.pred_y-pred.target_y),atol=1e-12)
        for _,g in pred.groupby(['study','condition','model','rep']):
            assert len(g)==2*cfg['test_events'] and g.event.nunique()==cfg['test_events']
        for _,r in metrics.iterrows():
            g=pred[(pred.study==r.study)&(pred.condition==r.condition)&(pred.model==r.model)]
            if r.group!='all':g=g[g.strength==['weak','medium','strong'].index(r.group)]
            np.testing.assert_allclose(g.error.mean(),r['mean'])
        for path in out.glob('*.npz'):
            d=np.load(path);roles=[set(d['event'][d['role']==role]) for role in ['train','validation','test']]
            assert not roles[0]&roles[1] and not roles[0]&roles[2] and not roles[1]&roles[2]
            for key in ['y','strength','onset']:np.testing.assert_array_equal(d[key][::2],d[key][1::2])
            if roles[0]:
                assert len(roles[0])==cfg['train_events']*len(cfg['training_seeds'])
                assert len(roles[1])==cfg['validation_events']*len(cfg['training_seeds'])
    else:
        fit=pd.read_csv(out/'fit_rows.csv');test=pd.read_csv(out/'test_rows.csv')
        assert not set(fit.event_uid)&set(test.event_uid)
        assert fit[fit.role=='train'].event_uid.nunique()==cfg['train_events']
        assert fit[fit.role=='validation'].event_uid.nunique()==cfg['validation_events']
        for domain,n in [('ID',cfg['id_test_events']),('noise2',cfg['id_test_events']),('slow',cfg['slow_test_events'])]:
            assert test[test.domain==domain].event_uid.nunique()==n
        keys=['event_uid','shot','domain']
        merged=pred.merge(test[keys+['epicenter_row','epicenter_col']],on=keys,validate='many_to_one')
        np.testing.assert_allclose(merged.error_mm,np.hypot(merged.pred_x-merged.epicenter_row,merged.pred_y-merged.epicenter_col),atol=1e-12)
        for _,r in metrics[metrics.group.isin(['all','weak','medium','strong'])].iterrows():
            g=pred[(pred.horizon==r.horizon)&(pred.model==r.model)&(pred.domain==r.domain)]
            if r.group!='all':g=g[g.strength_band==['weak','medium','strong'].index(r.group)]
            np.testing.assert_allclose(g.error_mm.mean(),r.mean_mm)
    result=dict(status='passed',study=cfg['study'],profile=cfg['profile'],prediction_rows=len(pred),metric_rows=len(metrics),elapsed_seconds=metadata['elapsed_seconds'],checks=['fresh generated record splits','per-shot coordinate error','aggregate errors','configured event counts'])
    (out/'reproduction_checks.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run_name');a=ap.parse_args()
    if not a.run_name.replace('-','').replace('_','').isalnum():raise ValueError('invalid run name')
    print(json.dumps(verify(ROOT/'.repro_runs'/a.run_name/'run'),indent=2))
