"""Summarize frozen validation selection; never select by test error."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    runs = Path(__file__).resolve().parents[2]/'localization_runs'
    out = runs/'temporal_gnn_improvements_20260908'
    selection = json.loads((out/'selected_before_test.json').read_text())
    selected = selection['arm']
    frame = pd.read_csv(out/'predictions.csv')
    previous = pd.read_csv(runs/'model_comparison_20260908/confirmation/predictions.csv')
    records = []
    for split in ['test_id', 'test_ood']:
        candidate = frame[(frame.arm==selected)&(frame.split==split)].groupby('event').error_mm.mean()
        for name in ['self_only', 'gnn', 'cnn_single', 'fixed_ridge_100.0', 'svr_17']:
            if name in ['self_only', 'gnn']:
                ref = frame[(frame.arm==name)&(frame.split==split)].groupby('event').error_mm.mean()
            else:
                ref = previous[(previous.candidate==name)&(previous.split==split)].groupby('event').error_mm.mean()
            assert candidate.index.equals(ref.index)
            delta = (candidate-ref).to_numpy()
            rng = np.random.default_rng(20260908)
            boot = [rng.choice(delta, len(delta)).mean() for _ in range(10000)]
            records.append(dict(selected=selected, reference=name, split=split,
                                difference_mm=float(delta.mean()),
                                event_ci95_mm=np.quantile(boot,[.025,.975]).tolist()))
    (out/'selected_comparisons.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Selected:', selected)
    print(pd.DataFrame(selection['validation']).groupby('arm').validation_error_mm.mean().to_string())
    print((out/'metrics.csv').read_text())
    print(json.dumps(records,indent=2))
    sub = frame[frame.arm.isin([selected,'gnn','self_only'])]
    print(sub.groupby(['arm','split','strength_band']).error_mm.mean().to_string())
    print('Training seconds:',sum(x['seconds'] for x in selection['validation']))


if __name__=='__main__':
    main()
