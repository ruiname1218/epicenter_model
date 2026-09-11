import numpy as np
import pandas as pd
from qp_ode_simulator.strength_benchmark import masks,summary,TEST_SEEDS,SEEDS

def test_strength_masks_keep_strong_shape_holdout_distinct():
    rows=pd.DataFrame({'strength_band':[0,1,2,2],'geometry':['circular','elliptical','circular','elliptical']})
    m=masks(rows)
    assert [int(m[k].sum()) for k in ['weak','medium','strong','strong_circle','strong_ellipse']]==[1,1,2,1,1]
    assert len(set(TEST_SEEDS))==len(set(SEEDS))==3

def test_metrics_thresholds_and_tail():
    s=summary([0,1,2,4])
    assert s['mean_mm']==1.75 and s['within_1mm']==.5 and s['over_3mm']==.25
    assert s['p95_mm']>=s['p90_mm']>=s['median_mm']
