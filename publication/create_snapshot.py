"""Copy a research working tree without heavy artifacts; never modifies the source."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
             '.ruff_cache', '.cache', 'build', 'dist', 'node_modules'}
HEAVY = {'.npz', '.npy', '.joblib', '.pkl', '.pickle', '.pt', '.pth', '.ckpt',
         '.safetensors', '.h5', '.hdf5', '.onnx'}
LIMIT = 25 * 1024**2
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}'
                    rb'|AKIA[0-9A-Z]{16}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----'
                    rb'|sk-[A-Za-z0-9]{40,})')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.destination.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Use a separate destination outside the source tree')
    included, excluded, directories = [], [], []
    candidates = []
    for folder, names, files in os.walk(source, followlinks=False):
        for name in list(names):
            p = Path(folder) / name
            if name in SKIP_DIRS or name.endswith('.egg-info') or p.is_symlink():
                names.remove(name)
                directories.append(dict(path=p.relative_to(source).as_posix(),
                                        reason='environment, VCS, cache, build or symlink'))
        for name in files:
            p = Path(folder) / name
            rel = p.relative_to(source).as_posix()
            size = p.lstat().st_size
            reason = ('symlink' if p.is_symlink() else
                      'secret-like filename' if name == '.env' or name.startswith('.env.') or
                      p.suffix.lower() in {'.pem', '.key'} else
                      'raw array, feature cache or model checkpoint' if p.suffix.lower() in HEAVY else
                      'file exceeds 25 MiB' if size > LIMIT else None)
            if reason:
                excluded.append(dict(path=rel, bytes=size, reason=reason))
                continue
            # Report filenames only, never a matched secret value.
            data = p.read_bytes()
            if SECRET.search(data):
                raise ValueError(f'Possible credential: {rel}; inspect before publishing')
            target = dest / rel
            if target.exists():
                raise ValueError(f'Destination already contains source path: {rel}')
            candidates.append((p, target, dict(path=rel, bytes=size,
                               sha256=hashlib.sha256(data).hexdigest())))
    # Copy only after the complete retained-file scan has succeeded.
    for p, target, record in candidates:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Copy changed: {record["path"]}')
        included.append(record)
    pub = dest / 'publication'
    pub.mkdir(exist_ok=True)
    for name, records, fields in [
        ('included_files.csv', included, ['path', 'bytes', 'sha256']),
        ('excluded_files.csv', excluded, ['path', 'bytes', 'reason']),
        ('excluded_directories.csv', directories, ['path', 'reason'])]:
        with (pub / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda x:x['path']))
    summary = dict(included_files=len(included), included_bytes=sum(x['bytes'] for x in included),
                   excluded_files=len(excluded), excluded_bytes=sum(x['bytes'] for x in excluded),
                   excluded_directories=len(directories), per_file_limit_bytes=LIMIT,
                   source_unchanged=True, git_lfs_used=False,
                   note='Excluded-directory contents are not counted in excluded-file totals. '
                        'Credential pattern scan is a precaution, not an exhaustive security audit.')
    (pub / 'snapshot_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.copy2(Path(__file__), pub / 'create_snapshot.py')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
