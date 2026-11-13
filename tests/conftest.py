import pytest
from pathlib import Path
from types import SimpleNamespace
from paperflow.methods import MethodSpec
@pytest.fixture
def fake_context(tmp_path):
    cfg=SimpleNamespace(model=SimpleNamespace(name='Qwen/Qwen3-0.6B'),compression_ratio=0.2)
    return dict(cfg=cfg,method=MethodSpec(name='ksvd',iterations=60),
                repo_root=tmp_path,run_dir=tmp_path/'run',
                checkpoint_root=tmp_path/'outputs',calib_root=tmp_path/'calib',dry_run=True)
