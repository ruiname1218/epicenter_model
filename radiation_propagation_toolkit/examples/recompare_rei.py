"""Validation-selected endpoint REI adaptation on an already inspected test set.

Run from repository root: .venv/bin/python examples/recompare_rei.py
This is exploratory reanalysis, not a fresh confirmatory experiment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qp_ode_simulator.dataset import iter_shards
from qp_ode_simulator.temporal_diagnosis import rei_center
from qp_ode_simulator.temporal_localization import detector_series


def main():
    runs = Path(__file__).resolve().parents[2] / 'localization_runs'
    output = runs / 'rei_recomparison_20260908'
    output.mkdir(exist_ok=False)
    model_run = runs / 'model_comparison_20260908'
    info = json.loads((model_run / 'experiment/model.json').read_text())
    spec, plan = info['specification'], info['plan']
    sites = spec['sites_mm']
    device = spec['geometry']['circuit_physical_coords_mm']
    center = np.array(info['center'])

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    def collect(path):
        arrays, frames = [], []
        for data, labels in iter_shards(path):
            values = detector_series(data, spec)
            shots = len(values) // len(labels)
            rows = labels.loc[labels.index.repeat(shots)].copy().reset_index(drop=True)
            rows['shot'] = np.tile(np.arange(shots), len(labels))
            arrays.append(values)
            frames.append(rows)
        return np.concatenate(arrays), pd.concat(frames, ignore_index=True)

    policy = {
        'status': 'exploratory reanalysis of previously inspected test data',
        'input': 'interior detector events; one row per extraction round',
        'primary_repetitions': 1,
        'sensitivity_repetitions': 2048,
        'mapping_caveat': 'Neither interpretation is certified as author input semantics.',
        'history_grid': [16, 64, 256, 1024, 2047],
        'spatial_multiplier': 2,
        'selection': 'validation mean position error, train-center fallback for abstentions',
        'fallback': center.tolist(),
        'test_model_names': ['constant', 'cnn_single', 'cnn_ensemble',
                             'fixed_ridge_100.0', info['selected']],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write('protocol.json', policy)
    raw, rows = collect(runs / 'window_calibration_20260908/dataset')
    val = rows.event.isin(plan['validation']).to_numpy()
    assert not set(plan['validation']) & set(plan['train_pool'])
    truth = rows[['epicenter_row', 'epicenter_col']].to_numpy()[val]

    def predict(values, repetitions, history, multiplier=2):
        return rei_center(values, sites, device, circuit_repetitions=repetitions,
                          history_length=history, correlation_multiplier=multiplier)

    trials, selected = [], {}
    for repetitions in (1, 2048):
        group = []
        for history in policy['history_grid']:
            p = predict(raw[val], repetitions, history)
            valid = np.isfinite(p).all(1)
            fallback = np.where(valid[:, None], p, center)
            item = dict(repetitions=repetitions, history=history,
                        coverage=float(valid.mean()),
                        validation_error_mm=float(np.linalg.norm(fallback-truth, axis=1).mean()))
            trials.append(item)
            group.append(item)
        selected[str(repetitions)] = min(group, key=lambda x: x['validation_error_mm'])
    write('validation.json', trials)
    write('selected_before_scoring.json', selected)
    print('Validation selection:', selected, flush=True)

    test_raw, test_rows = collect(model_run / 'fresh_dataset')
    for key in ('event_uid', 'generation_seed', 'syndrome_seed'):
        assert not set(rows[key]) & set(test_rows[key])
    saved = pd.read_csv(model_run / 'confirmation/predictions.csv')
    frame = saved[saved.candidate.isin(policy['test_model_names'])].copy()
    base = saved[saved.candidate == 'constant'].sort_values(['event', 'shot']).copy()
    assert np.array_equal(base[['event', 'shot']], test_rows[['event', 'shot']])
    np.testing.assert_allclose(base[['true_x_mm', 'true_y_mm']],
                               test_rows[['epicenter_row', 'epicenter_col']])
    frames = [frame]
    settings = [('rei_legacy', None, 2047, 2)]
    for repetitions in (1, 2048):
        for history in policy['history_grid']:
            settings.append((f'rei_r{repetitions}_k{history}', repetitions, history, 2))
    settings.append(('rei_prose_gate_sensitivity', 1, selected['1']['history'], 1))
    for name, repetitions, history, multiplier in settings:
        p = predict(test_raw, repetitions, history, multiplier)
        for fallback in (False, True):
            g = base.copy()
            g['candidate'] = name + ('_fallback' if fallback else '')
            g['answered'] = np.isfinite(p).all(1)
            pred = np.where(g.answered.to_numpy()[:, None], p, center) if fallback else p
            g['predicted_x_mm'], g['predicted_y_mm'] = pred.T
            g['error_mm'] = np.linalg.norm(pred-base[['true_x_mm', 'true_y_mm']].to_numpy(), axis=1)
            frames.append(g)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics = []
    for (name, split), g in predictions.groupby(['candidate', 'split']):
        error = g.error_mm.dropna()
        metrics.append(dict(candidate=name, split=split, events=int(g.event.nunique()),
                            coverage=float(g.error_mm.notna().mean()),
                            mean_mm=float(error.mean()) if len(error) else None,
                            p90_mm=float(error.quantile(.9)) if len(error) else None))
    write('metrics.json', metrics)
    reference = f"rei_r1_k{selected['1']['history']}_fallback"
    comparisons = []
    for split in ('test_id', 'test_ood'):
        ref = predictions[(predictions.candidate == reference) & (predictions.split == split)]
        for name in policy['test_model_names']:
            g = predictions[(predictions.candidate == name) & (predictions.split == split)]
            delta = (g.groupby('event').error_mm.mean()-ref.groupby('event').error_mm.mean()).to_numpy()
            rng = np.random.default_rng(20260908)
            boot = [rng.choice(delta, len(delta)).mean() for _ in range(10000)]
            comparisons.append(dict(candidate=name, split=split, minus_selected_rei_mm=float(delta.mean()),
                                    event_bootstrap_95pct_mm=np.quantile(boot, [.025, .975]).tolist()))
    write('paired.json', comparisons)
    quiet = np.load(runs / 'window_calibration_20260908/quiet/quiet_syndromes.npz')
    quiet_raw = detector_series(dict(quiet), spec)
    write('quiet.json', [dict(candidate=name, answered=int(np.isfinite(predict(
        quiet_raw, rep, k, mult)).all(1).sum()), shots=len(quiet_raw))
        for name, rep, k, mult in settings])
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
