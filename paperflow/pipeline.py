"""Re-entrant pipeline with config fingerprints and verified stage resumes."""
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
from .utils import dump_json,load_json,fingerprint,torch_versions

class Pipeline:
    def __init__(self,statistics,compression,evaluation,reporter,method):
        self.statistics=statistics;self.compression=compression
        self.evaluation=evaluation;self.reporter=reporter
        self.method=method

    def run(self,cfg):
        root=Path(os.path.abspath(str(cfg.repo_root)))
        run_dir=Path(os.path.abspath(str(cfg.run_dir)))
        ckpt_root=Path(os.path.abspath(str(cfg.checkpoint_root)))
        calib_root=Path(os.path.abspath(str(cfg.calib_root)))
        if not cfg.dry_run:run_dir.mkdir(parents=True,exist_ok=True)
        ctx={'cfg':cfg,'repo_root':root,'run_dir':run_dir,
             'checkpoint_root':ckpt_root,'calib_root':calib_root,
             'method':self.method,'dry_run':bool(cfg.dry_run)}
        if cfg.checkpoint_path:
            ctx['model_path']=str(Path(os.path.abspath(str(cfg.checkpoint_path))))
        cfg_data=OmegaConf.to_container(cfg,resolve=True)
        stage_cfg={'collect':cfg.components.statistics,'compress':cfg.components.compression,
                   'evaluate':cfg.components.evaluation}
        stages={'collect':self.statistics,'compress':self.compression,'evaluate':self.evaluation}
        results={}
        for stage,worker in stages.items():
            if not bool(getattr(cfg.stages,stage)):
                results[stage]={'status':'disabled'}
                continue
            key=fingerprint({'stage':stage,'stage_cfg':OmegaConf.to_container(stage_cfg[stage],resolve=True),
                             'cfg':cfg_data,'model_path':ctx.get('model_path')})
            receipt=run_dir/f'{stage}.receipt.json'
            cached=None
            if not cfg.force and not cfg.dry_run and receipt.is_file():
                previous=load_json(receipt)
                if previous.get('fingerprint')==key:
                    previous_result=previous['result']
                    check=(Path(previous_result['model_path'])/'config.json' if stage=='compress' and
                           previous_result.get('model_path') else None)
                    valid_stats=(stage!='collect' or (Path(previous_result.get('calib_root','/absent'))/'calibration_manifest.json').is_file())
                    if valid_stats and (stage!='compress' or (check and check.is_file())):
                        cached=previous_result
            if cached is not None:
                result=dict(cached,status='cached')
            else:
                result=worker.run(ctx)
                result.setdefault('status','planned' if cfg.dry_run else 'completed')
                if not cfg.dry_run:dump_json(receipt,{'fingerprint':key,'result':result})
            results[stage]=result
            if result.get('model_path'):ctx['model_path']=result['model_path']
            if result.get('calib_root'):ctx['calib_root']=Path(result['calib_root'])
        from .evaluation import collect_eval_metrics
        metrics=collect_eval_metrics(results.get('evaluate',{}).get('results_root',''))
        payload={'project':str(cfg.project),'run_dir':str(run_dir),'model_path':ctx.get('model_path'),
                 'configuration':cfg_data,'environment':torch_versions(),'stages':results,
                 'metrics':metrics,'created_unix':int(time.time())}
        if not cfg.dry_run:
            # JSON first, so subsequent reporters can attach its artifact.
            dump_json(run_dir/'results.json',payload)
            if cfg.stages.report:self.reporter.report(ctx,payload)
        return payload
