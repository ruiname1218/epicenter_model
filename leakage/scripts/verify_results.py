"""Verify published evidence without raw arrays, training or model loading."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for item in json.loads((ROOT/'publication_files.json').read_text()):
    p=ROOT/item['path']
    assert p.stat().st_size==item['bytes'],p
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
base=ROOT/'results/poster_matched_v3'
df=pd.read_csv(base/'predictions.csv');metrics=pd.read_csv(base/'metrics.csv')
np.testing.assert_allclose(df.error,np.hypot(df.pred_x-df.target_x,df.pred_y-df.target_y),atol=1e-12)
assert not df.duplicated(['study','condition','model','rep','event','shot']).any()
for _,g in df.groupby(['study','condition','model','rep']):
    assert len(g)==1080 and g.event.nunique()==540
    assert (g.groupby('event').size()==2).all()
for _,r in metrics.iterrows():
    g=df[(df.study==r.study)&(df.condition==r.condition)&(df.model==r.model)]
    if r.group!='all':g=g[g.strength==['weak','medium','strong'].index(r.group)]
    for actual,expected in [(g.error.mean(),r['mean']),((g.error<=1).mean(),r.within_1),(g.error.quantile(.9),r.p90)]:
        np.testing.assert_allclose(actual,expected,atol=1e-12)
coverage=pd.read_csv(ROOT/'results/leakage_research_report_complete_20261009/poster_result_coverage.csv')
assert len(coverage)==44
for _,r in coverage.iterrows():
    match=metrics[(metrics.study==r.study)&(metrics.condition==r.condition)&(metrics.model==r.model)&(metrics.group==r.group)]
    assert len(match)==1
    np.testing.assert_allclose(r['mean'],match.iloc[0]['mean'])
ref=ROOT/'results/legacy_centroid_reference_v3'
pred=pd.read_csv(ref/'predictions.csv');summary=pd.read_csv(ref/'metrics.csv')
np.testing.assert_allclose(pred.error,np.hypot(pred.pred_x-pred.target_x,pred.pred_y-pred.target_y),atol=1e-12)
for _,r in summary.iterrows():
    g=pred[pred.condition==r.condition]
    if r.group!='all':g=g[g.strength==['weak','medium','strong'].index(r.group)]
    np.testing.assert_allclose(g.error.mean(),r['mean'])
print('Publication hashes, 112 aggregate rows, 44 poster entries and centroid metrics passed.')
