"""Frozen d5 feature ablations with independent calibration and fresh test events.

No query labels enter feature extraction. Two shots of an event stay in one split.
Previous test sets are neither fitting nor selection inputs.
"""
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .api import run_simulation
from .dataset import event_configuration, _seed, _write_json, _generation_lock
from .distance_growth import features, load_new, training_indices
from .distance_study import fields_to_pauli, load as old_load, ordinary, group_masks
from .fault_response import sample
from .localization import _hash_file
from .temporal_diagnosis import rei_center

SEED = 2026090917
NTEST = 720
VARIANTS = ('baseline', 'multiscale', 'whitened', 'correlation', 'combined')


def read(path):
    return json.loads(Path(path).read_text())


def pairs_for(sites):
    sites = np.asarray(sites)
    distances = np.linalg.norm(sites[:, None] - sites[None, :], axis=2)
    positive = distances[distances > 1e-9]
    return np.argwhere(np.triu((distances <= positive.min()*1.42) & (distances > 0), 1))


def counts(raw, bins):
    edges = np.linspace(0, raw.shape[2], bins+1, dtype=int)
    if np.any(np.diff(edges) == 0):
        raise ValueError('too many bins')
    return np.stack([raw[:, :, a:b].mean(2) for a, b in zip(edges[:-1], edges[1:])], 2)


def coincidences(raw, pairs):
    """Ordered short-lag pairs; self lag one; four time windows. Not independent bits."""
    pieces = []
    for lag in (0, 1):
        links = list(pairs) if lag == 0 else list(pairs) + [p[::-1] for p in pairs] + [(i, i) for i in range(raw.shape[1])]
        edges = np.linspace(0, raw.shape[2]-lag, 5, dtype=int)
        for a, b in zip(edges[:-1], edges[1:]):
            pieces.append(np.stack([(raw[:, i, a:b] * raw[:, j, a+lag:b+lag]).mean(1) for i, j in links], 1))
    return np.concatenate(pieces, 1)


def fit_calibration(quiet, sites):
    pairs = pairs_for(sites)
    calibration = {'pairs': pairs, 'mean': {}, 'white': {}}
    for bins in (4, 16):
        rates = counts(quiet, bins)
        mean = rates.mean(0)
        centered = (rates-mean).transpose(0, 2, 1).reshape(-1, quiet.shape[1])
        cov = centered.T @ centered / (len(centered)-1)
        cov = .8*cov + .2*np.diag(np.diag(cov)) + np.eye(cov.shape[0])*1e-8
        values, vectors = np.linalg.eigh(cov)
        calibration['mean'][bins] = mean
        calibration['white'][bins] = (vectors / np.sqrt(values)) @ vectors.T
    joint = np.concatenate([coincidences(quiet[i:i+32], pairs) for i in range(0, len(quiet), 32)])
    calibration['joint_mean'] = joint.mean(0)
    calibration['joint_std'] = np.maximum(joint.std(0), 1e-4)
    return calibration


def feature_sets(raw, preprocessing, calibration):
    outputs = {v: [] for v in VARIANTS}
    for i in range(0, len(raw), 64):
        batch = raw[i:i+64]
        base = features(batch, preprocessing)
        multi, white = [], []
        for bins in (4, 16):
            centered = counts(batch, bins)-calibration['mean'][bins]
            multi.append(centered.reshape(len(batch), -1))
            white.append(np.einsum('ij,njk->nik', calibration['white'][bins], centered).reshape(len(batch), -1))
        multi = np.concatenate(multi, 1)
        white = np.concatenate(white, 1)
        joint = (coincidences(batch, calibration['pairs'])-calibration['joint_mean']) / calibration['joint_std']
        blocks = {'baseline': base, 'multiscale': np.c_[base, multi], 'whitened': np.c_[base, white],
                  'correlation': np.c_[base, joint], 'combined': np.c_[base, white, joint]}
        for name, block in blocks.items():
            outputs[name].append(block.astype(np.float32))
    return {k: np.concatenate(v) for k, v in outputs.items()}


def prepare(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    parent = root.parent/'distance_growth_20260909'
    p = read(parent/'protocol.json')
    protocol = dict(seed=SEED, test_events=NTEST, parent=str(parent.resolve()), parent_protocol_hash=_hash_file(parent/'protocol.json'),
        parent_data_hash=_hash_file(parent/'progress.json'), base=p['base'], sampling=p['sampling'], circuit=p['circuit'],
        geometry=p['geometry'], response=p['response'], preprocessing=p['preprocessing'], calibration_shots=512,
        training_events=2880, validation_events=360, variants=list(VARIANTS),
        primary='Ordinary validation winner (including frozen parent); compare fresh ordinary test to frozen parent blend.',
        weak_secondary='Lowest weak validation error subject to ordinary validation error <= parent blend + 0.01 mm; compare weak test to parent and fixed center.',
        search='Each variant: Ridge alpha 100/1000, SVR C10/100 gamma .057/nfeatures, ET leaf8/16; ET leaf16 with weak sample weight 4; SVR/ET 50:50; selected-model/prior signal gates.',
        controls='Same d5 hardware, duration, noise, training events. 512 independent quiet calibration shots additional resource. Strong ellipses excluded from training/validation.',
        inference='One shot of binary detector events [24,2047]; no event truth input; no averaging independent shots at inference.',
        bootstrap='10000 paired event resamples; descriptive 95% intervals, no multiplicity correction.',
        source_hash=_hash_file(Path(__file__)))
    _write_json(root/'protocol.json', protocol)


def make_calibration(root):
    p = read(root/'protocol.json')
    union = np.asarray(p['geometry']['union_mm'])
    indices = np.asarray(p['geometry']['distances']['5']['union_indices'])
    config, _ = event_configuration(p['base'], p['sampling'], 0, SEED+10, 123)
    simulation = run_simulation(config, coords_mm=union)
    xyz = fields_to_pauli(simulation, p['circuit'], indices, quiet=True)
    quiet = []
    for i in range(32):
        bits, _ = sample(p['response'], p['circuit'], *xyz, seed=_seed(SEED+11, i, 5) % (2**63-1), shots=16)
        quiet.append(bits)
    quiet = np.concatenate(quiet)
    np.savez_compressed(root/'calibration_bits.npz', d5=quiet)
    joblib.dump(fit_calibration(quiet, p['geometry']['distances']['5']['sites_mm']), root/'calibration.joblib')


def cache(root):
    p = read(root/'protocol.json'); parent = Path(p['parent'])
    assert _hash_file(parent/'progress.json') == p['parent_data_hash']
    if not (root/'calibration.joblib').exists():
        make_calibration(root)
    old, orows = old_load(Path(read(parent/'protocol.json')['parent']), 5, {'train'}); orows['origin'] = 'old'
    new, nrows = load_new(parent, {'train', 'validation'})
    rows = pd.concat([orows, nrows], ignore_index=True)
    raw = np.concatenate([old, new]); del old, new
    idx = training_indices(rows, 2880)
    val = np.flatnonzero((rows.role == 'validation') & ordinary(rows))
    keep = np.r_[idx, val]; rows = rows.iloc[keep].reset_index(drop=True); raw = raw[keep]
    assert len(idx) == 5760 and len(val) == 720
    assert not set(rows.loc[rows.role == 'train', 'event_uid']) & set(rows.loc[rows.role == 'validation', 'event_uid'])
    xs = feature_sets(raw, p['preprocessing'], joblib.load(root/'calibration.joblib'))
    for name, x in xs.items():
        np.save(root/f'features_{name}.npy', x)
    rows.to_csv(root/'rows.csv', index=False)
    # Observable gate score, not true event strength.
    q = np.asarray(p['preprocessing']['quiet'])
    signal = np.maximum((raw.mean(2)-q).mean(1), 0)
    np.save(root/'signal.npy', signal)
    _write_json(root/'cache.json', dict(training_events=2880, validation_events=360,
        feature_dimensions={k: v.shape[1] for k, v in xs.items()}, calibration_hash=_hash_file(root/'calibration.joblib'),
        hashes={x.name:_hash_file(x) for x in root.iterdir() if x.suffix in ('.npy', '.csv', '.npz')}))
    print('Cached', {k: v.shape for k, v in xs.items()}, flush=True)


def fit_variant(root, variant):
    root = Path(root); out = root/variant; out.mkdir(exist_ok=True)
    if (out/'frozen.json').exists():
        return variant
    rows = pd.read_csv(root/'rows.csv'); x = np.load(root/f'features_{variant}.npy')
    y = rows[['epicenter_row', 'epicenter_col']].to_numpy()
    train = np.flatnonzero(rows.role == 'train'); val = np.flatnonzero(rows.role == 'validation')
    weak = rows.iloc[val].strength_band.to_numpy() == 0
    weights = np.where(rows.iloc[train].strength_band.to_numpy() == 0, 4., 1.)
    score = lambda pred: dict(ordinary=float(np.linalg.norm(pred-y[val], axis=1).mean()), weak=float(np.linalg.norm(pred[weak]-y[val][weak], axis=1).mean()))
    logs, preds = {}, {}
    configs = [('Ridge'+str(a), 'Ridge', dict(alpha=a)) for a in (100, 1000)]
    configs += [('SVR'+str(c), 'SVR', dict(C=c, gamma=.057/x.shape[1])) for c in (10, 100)]
    configs += [('ET'+str(l), 'ET', dict(min_samples_leaf=l)) for l in (8, 16)]
    configs += [('ET_weak', 'ET', dict(min_samples_leaf=16))]
    for name, family, config in configs:
        start = time.monotonic()
        if family == 'Ridge': model = make_pipeline(StandardScaler(), Ridge(**config))
        elif family == 'SVR': model = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(**config, epsilon=.1)))
        else: model = ExtraTreesRegressor(n_estimators=256, random_state=41, n_jobs=2, **config)
        model.fit(x[train], y[train], **({'sample_weight': weights} if name == 'ET_weak' else {}))
        pred = model.predict(x[val]); preds[name] = pred
        joblib.dump(model, out/f'{name}.joblib')
        logs[name] = dict(**score(pred), seconds=time.monotonic()-start, config=config)
        print(variant, name, logs[name], flush=True)
    svr = min(['SVR10', 'SVR100'], key=lambda n: logs[n]['ordinary'])
    et = min(['ET8', 'ET16'], key=lambda n: logs[n]['ordinary'])
    preds['blend'] = .5*(preds[svr]+preds[et]); logs['blend'] = score(preds['blend'])
    np.savez_compressed(out/'validation.npz', **preds)
    _write_json(out/'frozen.json', dict(logs=logs, blend=[svr, et], hashes={f.name:_hash_file(f) for f in out.glob('*.joblib')}))
    return variant


def train(root, workers):
    if not (root/'cache.json').exists(): cache(root)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_variant, str(root), v) for v in VARIANTS]
        for f in as_completed(futures): print('Finished', f.result(), flush=True)


def freeze(root):
    if (root/'selection.json').exists(): raise ValueError('already frozen')
    p = read(root/'protocol.json'); parent = Path(p['parent'])/'n2880'
    rows = pd.read_csv(root/'rows.csv'); val = np.flatnonzero(rows.role == 'validation')
    y = rows.iloc[val][['epicenter_row', 'epicenter_col']].to_numpy(); weak = rows.iloc[val].strength_band.to_numpy() == 0
    x = np.load(root/'features_baseline.npy')[val]
    preds = {'parent': .5*(joblib.load(parent/'SVR.joblib').predict(x)+joblib.load(parent/'ExtraTrees.joblib').predict(x))}
    center = np.asarray(read(parent/'frozen.json')['center'])
    preds['prior'] = np.broadcast_to(center, y.shape)
    for variant in VARIANTS:
        with np.load(root/variant/'validation.npz') as z:
            preds.update({variant+'/'+k: z[k] for k in z.files})
    signal = np.load(root/'signal.npy')[val]
    # Fix a small gate grid from training-only observable quantiles; validate its mixture weight.
    train_signal = np.load(root/'signal.npy')[rows.role == 'train']
    thresholds = np.unique(np.quantile(train_signal, [.25, .5, .75]))
    gates = {}
    bases = ['parent'] + [v+'/blend' for v in VARIANTS] + [v+'/ET_weak' for v in VARIANTS]
    for base in bases:
        for i, threshold in enumerate(thresholds):
            if threshold <= 0: continue
            name = base+f'@gate{i}'
            weight = signal**2/(signal**2+threshold**2)
            preds[name] = center+weight[:, None]*(preds[base]-center)
            gates[name] = dict(base=base, threshold=float(threshold))
    def score(pred):
        e = np.linalg.norm(pred-y, axis=1)
        return dict(ordinary=float(e.mean()), weak=float(e[weak].mean()))
    scores = {k: score(v) for k, v in preds.items()}
    primary = min(scores, key=lambda k: scores[k]['ordinary'])
    eligible = [k for k in scores if scores[k]['ordinary'] <= scores['parent']['ordinary']+.01]
    specialist = min(eligible, key=lambda k: scores[k]['weak'])
    by_variant = {v: min([k for k in scores if k.startswith(v+'/')], key=lambda k: scores[k]['ordinary']) for v in VARIANTS}
    selection = dict(primary=primary, weak_specialist=specialist, by_variant=by_variant, scores=scores, gates=gates,
        center=center.tolist(), status='Frozen before fresh test generation', parent_hashes=read(parent/'frozen.json')['hashes'],
        source_hash=_hash_file(Path(__file__)), protocol_hash=_hash_file(root/'protocol.json'),
        model_hashes={str(f.relative_to(root)):_hash_file(f) for v in VARIANTS for f in (root/v).glob('*.joblib')},
        calibration_hash=_hash_file(root/'calibration.joblib'))
    _write_json(root/'selection.json', selection)
    pd.DataFrame(scores).T.sort_values('ordinary').to_csv(root/'validation_scores.csv')
    print('FROZEN', primary, scores[primary], 'weak', specialist, scores[specialist], 'parent', scores['parent'], flush=True)


_WORKER = None


def initialize(root):
    global _WORKER
    _WORKER = Path(root), read(Path(root)/'protocol.json')


def test_job(i):
    root, p = _WORKER
    config, labels = event_configuration(p['base'], p['sampling'], i, SEED, 123)
    simulation = run_simulation(config, coords_mm=np.asarray(p['geometry']['union_mm']))
    xyz = fields_to_pauli(simulation, p['circuit'], np.asarray(p['geometry']['distances']['5']['union_indices']))
    bits, _ = sample(p['response'], p['circuit'], *xyz, seed=_seed(SEED+1, i, 5) % (2**63-1), shots=2)
    row = simulation.parameters.iloc[0].to_dict()
    row = {k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for k, v in row.items()}
    row.update(labels, event=i, event_uid=f'd5-features-{SEED}:{i}', role='test')
    path = root/'events'/f'{i:04d}.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream: np.savez_compressed(stream, d5=bits)
    path.with_suffix('.npz.tmp').replace(path); _write_json(path.with_suffix('.json'), row)
    return i


def generate(root, workers):
    selection = read(root/'selection.json')
    assert selection['source_hash'] == _hash_file(Path(__file__))
    (root/'events').mkdir(exist_ok=True)
    with _generation_lock(root):
        done = [i for i in range(NTEST) if (root/'events'/f'{i:04d}.npz').exists() and (root/'events'/f'{i:04d}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(str(root),)) as pool:
            futures = [pool.submit(test_job, i) for i in range(NTEST) if i not in done]
            for future in as_completed(futures):
                done.append(future.result())
                if len(done) % 36 == 0: print('Fresh test', len(done), '/', NTEST, flush=True)
        _write_json(root/'test_manifest.json', dict(events=NTEST, selection_hash=_hash_file(root/'selection.json'),
            hashes={f.name:_hash_file(f) for f in (root/'events').iterdir()}))


def evaluate(root):
    p = read(root/'protocol.json'); s = read(root/'selection.json'); manifest = read(root/'test_manifest.json')
    assert manifest['selection_hash'] == _hash_file(root/'selection.json')
    assert s['source_hash'] == _hash_file(Path(__file__))
    assert s['calibration_hash'] == _hash_file(root/'calibration.joblib')
    for name, digest in s['model_hashes'].items(): assert _hash_file(root/name) == digest
    raw, rows = [], []
    for i in range(NTEST):
        path = root/'events'/f'{i:04d}.npz'
        for suffix in ('.npz', '.json'): assert _hash_file(path.with_suffix(suffix)) == manifest['hashes'][f'{i:04d}{suffix}']
        with np.load(path) as z: bits = z['d5']
        assert bits.shape == (2, 24, 2047) and np.isin(bits, [0, 1]).all()
        raw.append(bits); row = read(path.with_suffix('.json'))
        rows.extend([dict(row, shot=j) for j in range(2)])
    raw = np.concatenate(raw); rows = pd.DataFrame(rows)
    previous = pd.read_csv(root/'rows.csv')
    assert not set(rows.generation_seed) & set(previous.generation_seed)
    assert rows.event_uid.nunique() == NTEST
    xs = feature_sets(raw, p['preprocessing'], joblib.load(root/'calibration.joblib'))
    y = rows[['epicenter_row', 'epicenter_col']].to_numpy(); center = np.asarray(s['center'])
    parent = Path(p['parent'])/'n2880'
    for name, digest in s['parent_hashes'].items(): assert _hash_file(parent/name) == digest
    preds = {'parent': .5*(joblib.load(parent/'SVR.joblib').predict(xs['baseline'])+joblib.load(parent/'ExtraTrees.joblib').predict(xs['baseline'])), 'prior': np.broadcast_to(center, y.shape)}
    signal = np.maximum((raw.mean(2)-np.asarray(p['preprocessing']['quiet'])).mean(1), 0)
    def predict(name):
        if name in preds: return preds[name]
        if name in s['gates']:
            gate = s['gates'][name]; weight = signal**2/(signal**2+gate['threshold']**2)
            value = center+weight[:, None]*(predict(gate['base'])-center)
        else:
            variant, model = name.split('/')
            if model == 'blend': value = np.mean([predict(variant+'/'+m) for m in read(root/variant/'frozen.json')['blend']], axis=0)
            else: value = joblib.load(root/variant/(model+'.joblib')).predict(xs[variant])
        preds[name] = value
        return value
    targets = sorted(set(['parent', 'prior', s['primary'], s['weak_specialist']] + list(s['by_variant'].values())))
    for name in targets: predict(name)
    geo = p['geometry']['distances']['5']
    history = read(Path(p['parent'])/'selected_before_test.json')['rei']['history']
    rei = rei_center(raw, geo['sites_mm'], geo['physical_mm'], history_length=history, circuit_repetitions=1)
    answered = np.isfinite(rei).all(1); preds['REI'] = np.where(answered[:, None], rei, center); targets += ['REI']
    frames = []
    for name in targets:
        f = rows.copy(); f['model'] = name; f['error_mm'] = np.linalg.norm(preds[name]-y, axis=1)
        f['pred_x'], f['pred_y'] = preds[name].T; frames.append(f)
    frame = pd.concat(frames, ignore_index=True); frame.to_csv(root/'predictions.csv', index=False)
    metrics, paired = [], []
    for group, mask in group_masks(frame).items():
        sub = frame[mask]
        for name, f in sub.groupby('model'):
            e = f.error_mm
            metrics.append(dict(group=group, model=name, events=f.event_uid.nunique(), mean_mm=float(e.mean()), p90_mm=float(e.quantile(.9)), within_1mm=float((e <= 1).mean())))
        for name in sorted(set([s['primary'], s['weak_specialist']])):
            for reference in ('parent', 'prior', 'REI'):
                a = sub[sub.model == name].groupby('event_uid').error_mm.mean()
                b = sub[sub.model == reference].groupby('event_uid').error_mm.mean()
                assert a.index.equals(b.index)
                delta = (a-b).to_numpy(); rng = np.random.default_rng(SEED+20)
                boot = rng.choice(delta, (10000, len(delta))).mean(1)
                paired.append(dict(group=group, model=name, reference=reference, difference_mm=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist()))
    pd.DataFrame(metrics).to_csv(root/'metrics.csv', index=False); _write_json(root/'paired.json', paired)
    _write_json(root/'audit.json', dict(status='passed', test_events=NTEST, binary_shape=[2,24,2047], seed_overlap=0,
        frozen_selection_hash=_hash_file(root/'selection.json'), model_checksums='passed', data_checksums='passed', rei_answer_rate=float(answered.mean())))
    print(pd.DataFrame(metrics).query("group in ['ordinary','weak','strong_circle','strong_ellipse']").to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare','cache','train','freeze','generate','evaluate'])
    parser.add_argument('root', type=Path); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage in ('train', 'generate'): globals()[args.stage](args.root, args.workers)
    else: globals()[args.stage](args.root)
