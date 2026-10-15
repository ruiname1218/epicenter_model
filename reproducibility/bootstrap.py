"""Fetch immutable source revisions into an ignored cache; never vendor upstream source."""
from pathlib import Path
import hashlib,json,subprocess,shutil

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.repro_sources'

def call(*args,cwd=None):
    subprocess.run(args,cwd=cwd,check=True)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def bootstrap():
    sources=json.loads((ROOT/'reproducibility/sources.json').read_text())
    CACHE.mkdir(exist_ok=True)
    marker=CACHE/'ready.json'
    if marker.exists():
        state=json.loads(marker.read_text())
        assert state['sources']==sources,'Source pins changed; use a fresh checkout/cache'
        assert state['patch_hash']==digest(ROOT/'reproducibility/leakage/events.patch')
        for name,h in state['files'].items():assert digest(CACHE/name)==h,name
        return CACHE
    for name,spec in sources.items():
        dest=CACHE/name
        if not dest.exists():
            call('git','init',str(dest))
            call('git','remote','add','origin',spec['url'],cwd=dest)
        remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=dest,text=True).strip()
        assert remote==spec['url']
        call('git','fetch','--depth=1','--filter=blob:none','origin',spec['commit'],cwd=dest)
        call('git','sparse-checkout','init','--cone',cwd=dest)
        call('git','sparse-checkout','set',spec['sparse_path'],cwd=dest)
        call('git','checkout','--detach',spec['commit'],cwd=dest)
    # Minimal package avoids importing the upstream reservoir/classification stack.
    package=CACHE/'leakage_runtime/reserveq';package.mkdir(parents=True,exist_ok=True)
    (package/'__init__.py').write_text('"""Runtime subset fetched from pinned ReserveQ source."""\n')
    for name in ['codes.py','events.py','library_codes.py']:
        shutil.copy2(CACHE/'leakage/reserveq'/name,package/name)
    # Instrument a runtime copy, leaving the fetched upstream checkout unchanged.
    patch=str(ROOT/'reproducibility/leakage/events.patch')
    call('git','apply','--check','--unsafe-paths',f'--directory={package.parent}',patch,cwd=CACHE)
    call('git','apply','--unsafe-paths',f'--directory={package.parent}',patch,cwd=CACHE)
    # Expose the epoch budget without altering the paper profile (80 epochs).
    models=CACHE/'radiation/radiation_propagation_toolkit/src/qp_ode_simulator/rei_fair_models.py'
    text=models.read_text();needle='for epoch in range(80):'
    assert text.count(needle)==1
    models.write_text(text.replace(needle,'for epoch in range(CNN_EPOCHS):')+'\nCNN_EPOCHS = 80\n')
    files={str(p.relative_to(CACHE)):digest(p) for folder in [CACHE/'radiation/radiation_propagation_toolkit/src',package] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    marker.write_text(json.dumps(dict(sources=sources,patch_hash=digest(ROOT/'reproducibility/leakage/events.patch'),files=files),indent=2))
    return CACHE

if __name__=='__main__':print(bootstrap())
