"""Fast checks for source pins and archived paper settings; no downloads or training."""
from pathlib import Path
import json,re

ROOT=Path(__file__).resolve().parents[1]

def test_sources_are_immutable():
    sources=json.loads((ROOT/'reproducibility/sources.json').read_text())
    for spec in sources.values():assert re.fullmatch('[0-9a-f]{40}',spec['commit'])

def test_paper_profiles_match_archives():
    rad=json.loads((ROOT/'reproducibility/configs/radiation-paper.json').read_text())
    specs=json.loads((ROOT/rad['specs']).read_text())
    for role,domain,n in [('train','ID',864),('validation','ID',180),('test','ID',540),('test','slow',216)]:
        assert sum(s['role']==role and s['domain']==domain for s in specs)==n
    leak=json.loads((ROOT/'reproducibility/configs/leakage-paper.json').read_text())
    p=json.loads((ROOT/leak['protocol']).read_text())
    assert leak['train_events']==p['train_events_per_rep']
    assert leak['validation_events']==p['validation_events_per_rep']
    assert leak['test_events']==p['test_events']
    assert leak['training_seeds']==p['training_seeds']

def test_lockfile_has_fixed_versions():
    lines=(ROOT/'reproducibility/requirements-linux-py310.lock').read_text().splitlines()
    assert len(lines)>30
    assert all('==' in line for line in lines if line and not line.startswith(('#','--')))

def test_smoke_reduces_training_not_physics():
    for study in ['radiation','leakage']:
        def read(profile):return json.loads((ROOT/f'reproducibility/configs/{study}-{profile}.json').read_text())
        smoke,paper=read('smoke'),read('paper')
        assert smoke['protocol']==paper['protocol']
        assert smoke['train_events']<paper['train_events'] and smoke['cnn_epochs']<paper['cnn_epochs']
