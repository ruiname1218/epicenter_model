"""Compare a completed paper-profile rerun with archived numerical evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from bootstrap import ROOT

def compare(name):
    if not name.replace('-','').replace('_','').isalnum():raise ValueError('invalid run name')
    run=ROOT/'.repro_runs'/name/'run';meta=json.loads((run/'run_metadata.json').read_text())
    assert meta['status']=='completed' and meta['config']['profile']=='paper'
    study=meta['config']['study']
    source=ROOT/('leakage/results/poster_matched_v3' if study=='leakage' else 'radiation/results/rei_fair_20260911')
    keys=['study','condition','model','group'] if study=='leakage' else ['horizon','model','domain','group']
    actual=pd.read_csv(run/'metrics.csv').set_index(keys).sort_index()
    expected=pd.read_csv(source/'metrics.csv').set_index(keys).sort_index()
    assert actual.index.equals(expected.index),'Metric rows differ'
    numeric=list(expected.select_dtypes(include='number').columns)
    np.testing.assert_allclose(actual[numeric],expected[numeric],rtol=1e-7,atol=1e-9,equal_nan=True)
    delta=np.nanmax(np.abs(actual[numeric].to_numpy()-expected[numeric].to_numpy()))
    pkeys=['study','condition','model','rep','event','shot'] if study=='leakage' else ['horizon','model','domain','event_uid','shot']
    a=pd.read_csv(run/'predictions.csv').set_index(pkeys).sort_index()
    e=pd.read_csv(source/'predictions.csv').set_index(pkeys).sort_index()
    assert a.index.equals(e.index),'Prediction rows differ'
    columns=['pred_x','pred_y','error' if study=='leakage' else 'error_mm']
    np.testing.assert_allclose(a[columns],e[columns],rtol=1e-7,atol=1e-9)
    result=dict(status='passed',study=study,metric_rows=len(actual),metric_fields=numeric,max_absolute_metric_difference=float(delta),prediction_rows=len(a),max_absolute_prediction_difference=float(np.abs(a[columns].to_numpy()-e[columns].to_numpy()).max()),rtol=1e-7,atol=1e-9)
    (run/'paper_comparison.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run_name');a=ap.parse_args();print(json.dumps(compare(a.run_name),indent=2))
