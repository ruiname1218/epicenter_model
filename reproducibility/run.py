"""One-command generation, feature extraction, training and evaluation."""
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
from bootstrap import ROOT,bootstrap

def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',choices=['radiation','leakage']);ap.add_argument('--profile',choices=['smoke','paper'],default='smoke');ap.add_argument('--run-name');args=ap.parse_args()
    config_path=ROOT/'reproducibility/configs'/f'{args.study}-{args.profile}.json'
    cfg=json.loads(config_path.read_text());name=args.run_name or f'{args.study}-{args.profile}'
    if not name.replace('-','').replace('_','').isalnum():raise ValueError('Run name must contain only letters, digits, hyphens or underscores')
    parent=ROOT/'.repro_runs'/name
    out=parent/'run'
    if parent.exists():raise FileExistsError(f'{parent}: choose a new --run-name; results are never overwritten')
    cache=bootstrap()
    sys.path[:0]=[str(cache/'radiation/radiation_propagation_toolkit/src'),str(cache/'leakage_runtime')]
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
    out.mkdir(parents=True)
    started=time.time()
    metadata=dict(config=cfg,python=platform.python_version(),packages={n:version(n) for n in ['numpy','pandas','scipy','matplotlib','scikit-learn','torch','stim','pymatching','joblib']},sources=json.loads((cache/'ready.json').read_text()),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    metadata['runner_sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'reproducibility').rglob('*') if p.is_file() and p.suffix in ['.py','.json','.patch','.lock','.txt']}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    if args.study=='radiation':radiation(out,cfg)
    else:leakage(out,cfg)
    metadata.update(status='completed',elapsed_seconds=time.time()-started)
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    refresh_manifest(out)
    print('Completed:',out,flush=True)

def refresh_manifest(out):
    """Finalize the metadata hash after recording completion and elapsed time."""
    path=out/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        manifest['run_metadata.json']=hashlib.sha256((out/'run_metadata.json').read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest,indent=2))

def radiation(out,cfg):
    from qp_ode_simulator import rei_fair_data as data,rei_fair_models as models
    data.HORIZONS=models.HORIZONS=tuple(cfg['horizons']);models.SEEDS=tuple(cfg['cnn_seeds']);models.CNN_EPOCHS=cfg['cnn_epochs']
    protocol=json.loads((ROOT/cfg['protocol']).read_text())
    protocol['execution_profile']=cfg
    # Fails if simulator dependencies differ from the original archived experiment.
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));data.check(out)
    all_specs=json.loads((ROOT/cfg['specs']).read_text());specs=[]
    for role,domain,n in [('train','ID',cfg['train_events']),('validation','ID',cfg['validation_events']),('test','ID',cfg['id_test_events']),('test','slow',cfg['slow_test_events'])]:
        selected=[s for s in all_specs if s['role']==role and s['domain']==domain][:n]
        assert len(selected)==n
        specs.extend(selected)
    (out/'specs.json').write_text(json.dumps(specs,indent=2));(out/'events').mkdir()
    for test in [False,True]:
        selection_hash=data._hash_file(out/'selection.json') if test else None
        selected=[s for s in specs if (s['role']=='test')==test]
        with ProcessPoolExecutor(max_workers=cfg['workers'],initializer=data.initialize,initargs=(str(out),)) as pool:
            for i,_ in enumerate(pool.map(data.event_job,selected),1):
                if i%12==0:print('Generated',i,'/',len(selected),'test=',test,flush=True)
        manifest=dict(selection_hash=selection_hash,hashes={f'{s["event"]:04d}{ext}':data._hash_file(out/'events'/f'{s["event"]:04d}{ext}') for s in selected for ext in ['.npz','.json']})
        data._write_json(out/('test_manifest.json' if test else 'fit_manifest.json'),manifest)
        data.cache(out,test)
        if not test:models.train(out)
        else:
            assert selection_hash==data._hash_file(out/'selection.json')
            # The historical evaluator scans sibling runs for old seeds. Isolate its
            # working directory under a unique parent instead of changing that check.
            models.evaluate(out)

def leakage(out,cfg):
    import importlib.util
    path=ROOT/'reproducibility/leakage/benchmark.py'
    spec=importlib.util.spec_from_file_location('leakage_benchmark',path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.OUT=out;b.SEEDS=tuple(cfg['training_seeds']);b.NTRAIN=cfg['train_events'];b.NVAL=cfg['validation_events'];b.NTEST=cfg['test_events'];b.CNN_EPOCHS=cfg['cnn_epochs']
    b.main()

if __name__=='__main__':main()
