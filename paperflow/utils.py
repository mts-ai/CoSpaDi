"""Minimal, auditable IO helpers."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

def dump_json(path: str | Path, value: Any):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=p.parent,delete=False) as f:
        json.dump(value,f,indent=2,sort_keys=True,default=str,allow_nan=False)
        f.write('\n')
        tmp=f.name
    os.replace(tmp,p)

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def fingerprint(obj):
    data=json.dumps(obj,sort_keys=True,default=str).encode()
    return hashlib.sha256(data).hexdigest()

def run_checked(command, *, cwd, dry_run=False, env=None, log_path=None):
    """Argv-only subprocess launch, streaming output, bounded memory."""
    import collections
    command=list(map(str,command))
    if dry_run:
        print('DRY RUN:',subprocess.list2cmdline(command))
        return ''
    if log_path is not None:
        log_path=Path(log_path)
        log_path.parent.mkdir(parents=True,exist_ok=True)
    from contextlib import nullcontext
    with (open(log_path,'w',encoding='utf-8') if log_path else nullcontext()) as log:
        proc=subprocess.Popen(command,cwd=str(cwd),text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,env=env,bufsize=1)
        tail=collections.deque(maxlen=100)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line,end='',flush=True)
            if log is not None:log.write(line)
            tail.append(line.rstrip('\n'))
        code=proc.wait()
        if code!=0:
            raise subprocess.CalledProcessError(code,command,output='\n'.join(tail))
    return '\n'.join(tail)

def discover_checkpoint(root, stdout):
    """Use the program's emitted last path first, then search for HF config.json."""
    for line in reversed(stdout.splitlines()):
        path=Path(line.strip())
        if path.is_dir() and (path/'config.json').is_file():
            return path.resolve()
    root=Path(root)
    candidates=[p.parent for p in root.rglob('config.json')
                if p.parent.name != 'runs' and not p.parent.name.endswith('_separate')]
    if not candidates:
        raise FileNotFoundError(f'No Hugging Face checkpoint produced under {root}')
    return max(candidates,key=lambda p:(p/'config.json').stat().st_mtime).resolve()

def torch_versions():
    import torch
    return {'python':sys.version.split()[0],'torch':torch.__version__,
            'cuda':torch.version.cuda,'gpu_count':torch.cuda.device_count()}
