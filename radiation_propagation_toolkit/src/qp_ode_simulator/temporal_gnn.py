"""Small temporal CNN + shared-data-qubit message passing ablation."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import stim
import torch
from torch import nn

from .dataset import iter_shards, _write_json
from .localization import _hash_file
from .temporal_localization import TemporalCNN, detector_series


def graph_from_spec(spec):
    """Extract actual CX supports from the fixed d3 rotated-memory-Z circuit."""
    circuit = stim.Circuit.generated('surface_code:rotated_memory_z', distance=3, rounds=2)
    coords = circuit.get_final_qubit_coordinates()
    lookup = {tuple(v): k for k, v in coords.items()}
    sites = [lookup[tuple(x)] for x in spec['sites_grid']]
    if len(sites) != 8:
        raise ValueError('this experiment supports only the fixed distance-3 layout')
    supports = {q: set() for q in sites}
    for instruction in circuit.flattened():
        if instruction.name == 'CX':
            targets = [t.value for t in instruction.targets_copy()]
            for a, b in zip(targets[::2], targets[1::2]):
                if a in supports and b not in supports:
                    supports[a].add(b)
                if b in supports and a not in supports:
                    supports[b].add(a)
    adjacency = np.array([[i != j and bool(supports[a] & supports[b])
                           for j, b in enumerate(sites)] for i, a in enumerate(sites)], dtype=np.float32)
    if np.any(adjacency.sum(1) == 0):
        raise ValueError('isolated check')
    roles = {tuple(x): role for x, role in zip(spec['geometry']['circuit_grid_coords'],
                                              spec['geometry']['circuit_qubit_roles'])}
    xy = np.array(spec['sites_mm'], dtype=np.float32)
    scale = max(float(np.ptp(xy, axis=0).max()/2), 1e-6)
    features = np.column_stack(((xy-xy.mean(0))/scale,
                                [roles[tuple(x)] == 1 for x in spec['sites_grid']],
                                [roles[tuple(x)] == 2 for x in spec['sites_grid']])).astype(np.float32)
    return adjacency, features, {str(q): sorted(supports[q]) for q in sites}


class TemporalGNN(nn.Module):
    def __init__(self, adjacency, node_features, exchange=True):
        super().__init__()
        sites = len(adjacency)
        # Matched encoder and parameter count between the two ablation arms.
        self.temporal = TemporalCNN(sites).temporal
        a = np.asarray(adjacency, dtype=np.float32) if exchange else np.eye(sites, dtype=np.float32)
        self.register_buffer('adjacency', torch.tensor(a/a.sum(1, keepdims=True)))
        self.register_buffer('node_features', torch.tensor(node_features))
        self.project = nn.Linear(133, 32)
        self.self_layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(2)])
        self.message_layers = nn.ModuleList([nn.Linear(32, 32, bias=False) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(sites*32, 64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64, 2))

    def forward(self, values):
        b, s, t = values.shape
        encoded = self.temporal(values.reshape(b*s, 1, t)*10).reshape(b, s, 128)
        h = torch.relu(self.project(torch.cat((encoded, values.mean(2, keepdim=True)*10,
                                              self.node_features.expand(b, -1, -1)), dim=2)))
        for local, message in zip(self.self_layers, self.message_layers):
            h = h + torch.relu(local(h) + message(torch.matmul(self.adjacency, h)))
        return self.head(h.flatten(1))


def infer(model, raw, center, scale):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(np.array(raw[i:i+64], dtype=np.float32))).numpy()
                               for i in range(0, len(raw), 64)])*scale+center


def preprocess(values, quiet, representation):
    """Observed-only calibration; optional fixed 16-round averaging."""
    if representation == 'raw':
        return values
    x = np.asarray(values, dtype=np.float32) - np.asarray(quiet, dtype=np.float32)[None, :, None]
    if representation == 'centered':
        return x
    if representation == 'binned':
        # Include the final incomplete bin without padding bias or lost rounds.
        return np.stack([x[..., i:i+16].mean(-1) for i in range(0, x.shape[-1], 16)], axis=-1)
    raise ValueError('unknown representation')


def run(runs, improve=False):
    runs = Path(runs)
    out = runs/('temporal_gnn_improvements_20260908' if improve else 'temporal_gnn_20260908')
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    info = json.loads((runs/'model_comparison_20260908/experiment/model.json').read_text())
    spec, plan = info['specification'], info['plan']
    adjacency, static, supports = graph_from_spec(spec)
    configs = {arm: dict(exchange=arm=='gnn', representation='raw', loss='mse')
               for arm in ['self_only', 'gnn']}
    if improve:
        for representation, loss in [('centered', 'mse'), ('binned', 'mse'), ('raw', 'huber')]:
            for graph in ['self_only', 'gnn']:
                configs[f'{representation}_{loss}_{graph}'] = dict(
                    exchange=graph=='gnn', representation=representation, loss=loss)
    policy = dict(status='exploratory; previously inspected test set', seeds=[41,42,43],
                  arms=list(configs), configs=configs, quiet=info['quiet'],
                  epochs=40, patience=8, batch_size=64,
                  selection='validation mean of single-model errors across three seeds',
                  adjacency=adjacency.tolist(), supports=supports, specification=spec,
                  source_sha256=_hash_file(Path(__file__)))
    _write_json(out/'protocol.json', policy)

    def collect(path):
        arrays, frames = [], []
        for data, labels in iter_shards(path):
            a = detector_series(data, spec)
            shots = len(a)//len(labels)
            rows = labels.iloc[np.repeat(np.arange(len(labels)), shots)].reset_index(drop=True)
            rows['shot'] = np.tile(np.arange(shots), len(labels))
            arrays.append(a)
            frames.append(rows)
        return np.concatenate(arrays), pd.concat(frames, ignore_index=True)

    raw, rows = collect(runs/'window_calibration_20260908/dataset')
    train = np.flatnonzero(rows.event.isin(plan['training_sets']['360']))
    val = np.flatnonzero(rows.event.isin(plan['validation']))
    assert not set(rows.iloc[train].event) & set(rows.iloc[val].event)
    y = rows[['epicenter_row','epicenter_col']].to_numpy()
    center = y[train].mean(0)
    scale = max(float(np.ptp(spec['geometry']['circuit_physical_coords_mm'], axis=0).max()/2), 1e-3)
    fitted, logs = [], []
    for arm in policy['arms']:
        config = configs[arm]
        input_values = preprocess(raw, info['quiet'], config['representation'])
        for seed in policy['seeds']:
            torch.manual_seed(seed)
            model = TemporalGNN(adjacency, static, exchange=config['exchange'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            rng = np.random.default_rng(seed)
            best, best_error, chosen, history = None, np.inf, 0, []
            start_time = time.perf_counter()
            for epoch in range(1, 41):
                model.train()
                order = rng.permutation(train)
                for start in range(0, len(order), 64):
                    idx = order[start:start+64]
                    optimizer.zero_grad(set_to_none=True)
                    p = model(torch.tensor(input_values[idx], dtype=torch.float32))
                    target = torch.tensor((y[idx]-center)/scale, dtype=torch.float32)
                    loss = (nn.functional.mse_loss(p, target) if config['loss']=='mse' else
                            nn.functional.smooth_l1_loss(p, target, beta=.5))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                error = float(np.linalg.norm(infer(model, input_values[val], center, scale)-y[val], axis=1).mean())
                history.append(dict(epoch=epoch, error_mm=error))
                if error < best_error:
                    best, best_error, chosen = copy.deepcopy(model.state_dict()), error, epoch
                if epoch-chosen >= 8:
                    break
            model.load_state_dict(best)
            folder = out/f'{arm}_seed{seed}'
            folder.mkdir()
            torch.save(best, folder/'weights.pt')
            metadata = dict(arm=arm, seed=seed, config=config, validation_error_mm=best_error, epoch=chosen,
                            history=history, center=center.tolist(), scale=scale,
                            seconds=time.perf_counter()-start_time,
                            parameters=sum(p.numel() for p in model.parameters()),
                            weights_sha256=_hash_file(folder/'weights.pt'))
            _write_json(folder/'model.json', metadata)
            logs.append(metadata)
            fitted.append((arm, seed, model))
            print(arm, seed, 'validation', best_error, 'epoch', chosen, flush=True)
    selected = min(policy['arms'], key=lambda arm: np.mean([m['validation_error_mm'] for m in logs if m['arm']==arm]))
    _write_json(out/'selected_before_test.json', dict(arm=selected, validation=logs))
    test, test_rows = collect(runs/'model_comparison_20260908/fresh_dataset')
    for field in ['event_uid','generation_seed','syndrome_seed']:
        assert not set(rows[field]) & set(test_rows[field])
    frames = []
    for arm, seed, model in fitted:
        config = configs[arm]
        test_input = preprocess(test, info['quiet'], config['representation'])
        model.eval()
        # Check checkpoint-only reconstruction before scoring.
        restored = TemporalGNN(adjacency, static, exchange=config['exchange'])
        restored.load_state_dict(torch.load(out/f'{arm}_seed{seed}'/'weights.pt', weights_only=True))
        np.testing.assert_allclose(infer(restored, test_input[:2], center, scale), infer(model, test_input[:2], center, scale))
        p = infer(restored, test_input, center, scale)
        f = test_rows[['event','shot','event_uid','strength_band','geometry','epicenter_region']].copy()
        f['split'] = np.where((f.geometry=='elliptical') & (f.strength_band==2),'test_ood','test_id')
        f['arm'], f['seed'] = arm, seed
        f['predicted_x_mm'], f['predicted_y_mm'] = p.T
        f['error_mm'] = np.linalg.norm(p-test_rows[['epicenter_row','epicenter_col']].to_numpy(), axis=1)
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out/'predictions.csv', index=False)
    metrics = frame.groupby(['arm','split']).error_mm.agg(['mean','median',lambda x:x.quantile(.9)])
    metrics.columns=['mean_mm','median_mm','p90_mm']
    metrics.to_csv(out/'metrics.csv')
    paired=[]
    for split in ['test_id','test_ood']:
        g=frame[frame.split==split]
        delta=(g[g.arm=='gnn'].groupby('event').error_mm.mean()-g[g.arm=='self_only'].groupby('event').error_mm.mean()).to_numpy()
        rng=np.random.default_rng(20260908)
        boot=[rng.choice(delta,len(delta)).mean() for _ in range(10000)]
        paired.append(dict(split=split,gnn_minus_self_mm=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    _write_json(out/'paired.json',paired)
    print(metrics.to_string(), flush=True)
    print('selected:',selected, 'paired:',paired, flush=True)


if __name__ == '__main__':
    import sys
    run(sys.argv[1], improve='--improve' in sys.argv[2:])
