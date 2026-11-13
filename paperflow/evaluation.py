"""Evaluation of CoSpaDi compressed models, including 14-bit BF16 truncation."""
import sys
from pathlib import Path
from .utils import run_checked

class CoSpaDiEvaluation:
    def __init__(self, quant_coeffs=14):
        self.quant_coeffs=quant_coeffs
    def run(self,ctx):
        model_path=ctx.get('model_path')
        if not model_path and not ctx['dry_run']:
            raise ValueError('Set checkpoint_path for evaluate-only runs')
        cmd=[sys.executable,'-m','paperflow.impl.evaluate','--ckpt_path',str(model_path or '<CHECKPOINT_PATH>'),
             '--results_dir',str(ctx['run_dir']/'evaluation')]
        if self.quant_coeffs is not None:cmd+=['--quant_coeffs',str(self.quant_coeffs)]
        import os
        env=os.environ.copy()
        env['COSPADI_QUANT_OUTPUT_ROOT']=str(ctx['run_dir']/'quantized')
        run_checked(cmd,cwd=ctx['repo_root'],dry_run=ctx['dry_run'],env=env,log_path=ctx['run_dir']/'evaluation.log')
        return {'backend':'coefficient-quantization','quant_coeffs':self.quant_coeffs,
                'checkpoint':str(model_path),'command':cmd,'results_root':str(ctx['run_dir']/'evaluation')}

class LMEvalHarness:
    def __init__(self,tasks=None,batch_size='auto',num_fewshot=None,device='cuda:0',
                 dtype='bfloat16', trust_remote_code=True, output_dir='eval',limit=None):
        self.tasks=tasks or ['mmlu','arc_easy','arc_challenge','hellaswag','piqa',
                             'lambada_openai','race','wikitext','sciq','openbookqa',
                             'winogrande','mathqa']
        self.batch_size=batch_size;self.num_fewshot=num_fewshot;self.device=device
        self.dtype=dtype;self.trust_remote_code=trust_remote_code;self.output_dir=output_dir
        self.limit=limit
    def build_command(self,ctx):
        model=ctx.get('model_path') or '<CHECKPOINT_PATH>'
        model_args=f'pretrained={model},dtype={self.dtype},trust_remote_code={str(self.trust_remote_code)}'
        cmd=[sys.executable,'-m','lm_eval','--model','hf','--model_args',model_args,
            '--tasks',','.join(self.tasks),'--batch_size',str(self.batch_size),
            '--device',self.device,
            '--output_path',str(ctx['run_dir']/self.output_dir)]
        if self.num_fewshot is not None:cmd+=['--num_fewshot',str(self.num_fewshot)]
        if self.limit is not None:cmd+=['--limit',str(self.limit)]
        return cmd
    def run(self,ctx):
        if not ctx.get('model_path') and not ctx['dry_run']:
            raise ValueError('No saved model; set checkpoint_path to evaluate separately')
        cmd=self.build_command(ctx)
        run_checked(cmd,cwd=ctx['repo_root'],dry_run=ctx['dry_run'],log_path=ctx['run_dir']/'evaluation.log')
        return {'backend':'lm-eval','command':cmd,'results_root':str(ctx['run_dir']/self.output_dir)}


def collect_eval_metrics(results_root):
    """Extract numeric lm-eval task metrics, preserving task and metric names."""
    import json
    from pathlib import Path
    root=Path(results_root)
    if not root.exists():return {}
    if root.is_file():files=[root]
    else:files=sorted((p for p in root.rglob('*.json') if p.name.startswith('results')),
                     key=lambda p:p.stat().st_mtime,reverse=True)
    for path in files:
        try:
            payload=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        data=payload.get('results',payload)
        if not isinstance(data,dict):continue
        metrics={}
        for task,items in data.items():
            if not isinstance(items,dict):continue
            for key,value in items.items():
                if isinstance(value,(float,int)) and not isinstance(value,bool):
                    metrics[f'{task}/{key}']=float(value)
        if metrics:return metrics
    return {}
