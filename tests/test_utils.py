import pytest
from paperflow.utils import fingerprint,dump_json,load_json,discover_checkpoint,run_checked
def test_fingerprint_order():assert fingerprint({'a':1,'b':2})==fingerprint({'b':2,'a':1})
def test_atomic_json(tmp_path):
    p=tmp_path/'x.json';dump_json(p,{'foo':[1,2]});assert load_json(p)['foo']==[1,2]
def test_discovery_stdout(tmp_path):
    p=tmp_path/'ckpt';p.mkdir();(p/'config.json').write_text('{}')
    assert discover_checkpoint(tmp_path,'log\n'+str(p)+'\n')==p.resolve()
def test_discovery_failure(tmp_path):
    with pytest.raises(FileNotFoundError):discover_checkpoint(tmp_path,'no output')
def test_dryrun_no_execution(tmp_path):
    assert run_checked(['nonexistent-executable'],cwd=tmp_path,dry_run=True)==''
def test_subprocess_failure(tmp_path):
    import subprocess,sys
    with pytest.raises(subprocess.CalledProcessError):
        run_checked([sys.executable,'-c','raise RuntimeError(42)'],cwd=tmp_path)
