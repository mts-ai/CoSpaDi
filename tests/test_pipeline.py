from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from paperflow.pipeline import Pipeline
from paperflow.methods import MethodSpec
class Worker:
    def __init__(self,result):self.calls=0;self.result=result
    def run(self,ctx):self.calls+=1;return dict(self.result)
class Reporter:
    def __init__(self):self.calls=0
    def report(self,ctx,payload):self.calls+=1
def config(tmp,force=False):
    return OmegaConf.create({'project':'unit','repo_root':str(tmp),'run_dir':str(tmp/'run'),
      'checkpoint_root':str(tmp/'ckpt'),'calib_root':str(tmp/'calib'),
      'checkpoint_path':None,'dry_run':False,'force':force,
      'stages':{'collect':True,'compress':False,'evaluate':True,'report':True},
      'components':{'statistics':{'x':1},'compression':{'x':2},'evaluation':{'x':3}}})
def test_pipeline_caches_completed_stages(tmp_path):
    a=Worker({'calib_root':str(tmp_path/'calib')});b=Worker({});c=Worker({'quality':0.7});r=Reporter()
    p=Pipeline(a,b,c,r,MethodSpec('ksvd',60))
    p.run(config(tmp_path))
    (tmp_path/'calib').mkdir(exist_ok=True)
    (tmp_path/'calib/calibration_manifest.json').write_text('{}')
    p.run(config(tmp_path))
    assert (a.calls,b.calls,c.calls,r.calls)==(1,0,1,2)
    p.run(config(tmp_path,force=True))
    assert a.calls==2 and c.calls==2
def test_dryrun_no_files(tmp_path):
    a=Worker({});p=Pipeline(a,Worker({}),Worker({}),Reporter(),
                            MethodSpec('ksvd',60))
    cfg=config(tmp_path);cfg.dry_run=True
    p.run(cfg)
    assert not (tmp_path/'run').exists()
