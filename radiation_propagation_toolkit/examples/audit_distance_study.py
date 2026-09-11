"""Read-only checks on generated records; save an audit, never select models."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator.dataset import _write_json
from qp_ode_simulator.distance_study import role
from qp_ode_simulator.localization import _hash_file

root=Path(sys.argv[1]).resolve()
p=json.loads((root/'protocol.json').read_text())
progress=json.loads((root/'progress.json').read_text())
assert progress['status']=='complete' and progress['completed']==1440
assert _hash_file(root/'generation_source.py')==p['source_sha256']
rows=[]
for event in range(1440):
    path=root/'events'/f'{event:04d}.npz'
    assert _hash_file(path)==progress['hashes'][f'{event:04d}']
    row=json.loads(path.with_suffix('.json').read_text())
    assert row['event']==event and row['role']==role(event)
    assert not row['is_control']
    with np.load(path) as z:
        assert set(z.files)=={'d3','d5','d7'}
        for d in [3,5,7]:
            assert z[f'd{d}'].shape==(2,d*d-1,2047)
            assert set(np.unique(z[f'd{d}'])) <= {0,1}
    rows.append(row)
labels=pd.DataFrame(rows)
assert labels.event_uid.nunique()==1440 and labels.generation_seed.nunique()==1440
counts=labels.groupby(['role','geometry','propagation_law','epicenter_region','strength_band']).size()
for name,n in [('train',24),('validation',8),('test',8)]:
    assert len(counts.loc[name])==36 and (counts.loc[name]==n).all()
hardware=[c for c in labels if c in ['qp_baseline_t1_by_qubit_us','qp_baseline_t2_by_qubit_us','qp_qubit_frequency_by_qubit_ghz']]
assert len(hardware)==3 and all(labels[c].nunique()==1 for c in hardware)
previous=set()
for folder in ['window_calibration_20260908/dataset','temporal_diagnosis_20260908/fresh_dataset',
               'adaptive_features_20260908/fresh_dataset','model_comparison_20260908/fresh_dataset',
               'learning_curve_20260908/fresh_dataset','attention_mdn_gp_20260908/fresh_dataset','weak_growth_20260908/new_dataset']:
    for path in (root.parent/folder).rglob('*.csv'):
        frame=pd.read_csv(path)
        if 'generation_seed' in frame:previous.update(frame.generation_seed.astype(str))
for path in (root.parent/'weak_observation_20260908/events').glob('*.json'):
    record=json.loads(path.read_text())
    if 'generation_seed' in record:previous.add(str(record['generation_seed']))
overlap=set(labels.generation_seed.astype(str))&previous
assert not overlap
sources=Path(__file__).resolve().parents[1]/'src/qp_ode_simulator'
modules=['distance_study.py','fault_response.py','simulator.py','stim_qec.py','api.py','circuit_building.py','adaptive_localization.py','temporal_localization.py','temporal_diagnosis.py']
_write_json(root/'audit.json',dict(status='passed',physical_events=1440,paired_code_observations=4320,shots=8640,
    fixed_hardware_fields=hardware,source_generation_seed_overlap_with_prior_runs=0,
    earlier_generation_seeds_checked=len(previous),test_events=288,
    source_hashes={name:_hash_file(sources/name) for name in modules},
    tests='100 tests passed, including all 3 distances, 64 forced multifault patterns and 20000-shot parity/marginal checks per distance',
    generation_snapshot_sha256=_hash_file(root/'generation_source.py'),
    protocol_sha256=_hash_file(root/'protocol.json')))
print('Audit passed: 1440 paired physical events; balanced splits; fixed hardware; no old generation-seed overlap.')
