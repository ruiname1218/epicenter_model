"""Verify input hashes and recompute 48 published metrics; no pickle/model loading."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

hashes = json.loads((ROOT / 'paper/sources.json').read_text())
for name, expected in hashes.items():
    assert hashlib.sha256((ROOT / 'results' / name).read_bytes()).hexdigest() == expected, name
r = ROOT / 'results/rei_fair_20260911'
preds = read(r / 'predictions.csv')
metrics = read(r / 'metrics.csv')
checks = 0
for model in ['REI_full', 'SVR', 'Ridge', 'CNN']:
    for group, band in [('all', None), ('weak', '0'), ('medium', '1'), ('strong', '2')]:
        values = np.array([float(x['error_mm']) for x in preds
                           if x['horizon'] == '4' and x['domain'] == 'ID' and x['model'] == model
                           and (band is None or x['strength_band'] == band)])
        selected = [x for x in metrics if x['horizon'] == '4' and x['domain'] == 'ID'
                    and x['model'] == model and x['group'] == group]
        assert len(selected) == 1
        for key, actual in [('mean_mm', values.mean()), ('within_1mm', (values <= 1).mean()),
                            ('p90_mm', np.quantile(values, .9))]:
            assert abs(actual - float(selected[0][key])) < 1e-12, (model, group, key)
            checks += 1
print(f'{len(hashes)} source hashes and {checks} metric checks passed.')
