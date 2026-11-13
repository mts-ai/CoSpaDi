"""Keep logging separate from algorithms. ClearML/TB use lazy imports."""
import json
from pathlib import Path
from collections.abc import Mapping
from .utils import dump_json

class JsonReporter:
    def __init__(self,filename='results.json'):
        self.filename=filename
    def report(self,ctx,payload):
        dump_json(ctx['run_dir']/self.filename,payload)

class TensorBoardReporter:
    def __init__(self,log_dir='tensorboard'):
        self.log_dir=log_dir
    def report(self,ctx,payload):
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(ctx['run_dir']/self.log_dir)) as writer:
            for stage,obj in payload.get('stages',{}).items():
                writer.add_text(f'pipeline/{stage}',json.dumps(obj,default=str))
                for key,value in obj.items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        writer.add_scalar(f'pipeline/{stage}/{key}',value)
            for key,value in payload.get('metrics',{}).items():
                if isinstance(value,(int,float)):writer.add_scalar(key,value)

class ClearMLReporter:
    def __init__(self,project='CoSpaDi',task_name='run',enabled=False,
                 reuse_last_task_id=False):
        self.project=project;self.task_name=task_name;self.enabled=enabled
        self.reuse_last_task_id=reuse_last_task_id
    def report(self,ctx,payload):
        if not self.enabled:return
        from clearml import Task
        task=Task.init(project_name=self.project,task_name=self.task_name,
                       reuse_last_task_id=self.reuse_last_task_id)
        try:
            task.connect(payload['configuration'],name='resolved_hydra_config')
            for key,value in payload.get('metrics',{}).items():
                if isinstance(value,(int,float)):
                    task.get_logger().report_scalar(title='evaluation',series=key,
                                                    value=float(value),iteration=0)
            task.upload_artifact('pipeline_summary',str(ctx['run_dir']/'results.json'))
        finally:task.close()

class CompositeReporter:
    """Dispatch independently configured reporters in deterministic config order.

    Accepts either a Hydra-instantiated mapping (e.g. ``json`` and ``clearml``)
    or a list, to preserve the original pipeline integration interface.
    """
    def __init__(self, reporters):
        self.reporters = list(reporters.values()) if isinstance(reporters, Mapping) else list(reporters)
        for reporter in self.reporters:
            if not callable(getattr(reporter, 'report', None)):
                raise TypeError(f'Invalid reporter instance: {reporter!r}')
    def report(self,ctx,payload):
        errors=[]
        for reporter in self.reporters:
            try: reporter.report(ctx,payload)
            except Exception as exc:errors.append(f'{type(reporter).__name__}: {exc}')
        if errors:raise RuntimeError('Reporting failed: '+'; '.join(errors))
