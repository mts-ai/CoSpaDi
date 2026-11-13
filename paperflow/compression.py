"""Run the CoSpaDi K-SVD compression worker without changing its solver."""
import sys
from pathlib import Path
from .utils import run_checked


def deployment_subdir(bits=14, dtype='float16', scaling='minimum'):
    suffix = '_balanced' if scaling == 'balanced' else ''
    return f'transformers_fp{bits}_{dtype}{suffix}'


def _extras(args):
    result = []
    for key, value in args.items():
        flag = '--' + key
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            if isinstance(value, list):
                result.extend([flag, ','.join(map(str, value))])
            else:
                result.extend([flag, str(value)])
    return result


class CoSpaDiBackend:
    """Execute the paper K-SVD implementation using a separate worker process."""

    def build_command(self, ctx):
        cfg, method = ctx['cfg'], ctx['method']
        if method.name != 'ksvd':
            raise ValueError('CoSpaDi backend only supports ksvd.')
        return [sys.executable, "-m", "paperflow.impl.compress",
                '--method', 'ksvd', '--compression-ratio', str(cfg.compression_ratio),
                '--model-name', str(cfg.model.name), '--calib-data', str(ctx['calib_root']),
                '--save-path', str(ctx['checkpoint_root']),
                '--ks-ratio', str(method.ks_ratio),
                '--ksvd-iterations', str(method.iterations),
                '--power-iterations', str(method.power_iterations)] + _extras(method.extra_args)

    def factor_path(self, ctx):
        """Match the worker's auto_experiment_name without guessing from sibling runs."""
        cfg, method = ctx['cfg'], ctx['method']
        model_id = str(cfg.model.name).replace('/', '_')
        folder = (f'{model_id}_cr_{cfg.compression_ratio}_{method.name}_'
                  f'ks_{method.ks_ratio}_ksvd_it_{method.iterations}_'
                  f'power_it_{method.power_iterations}')
        return Path(ctx['checkpoint_root']) / folder

    def checkpoint_path(self, ctx):
        path = self.factor_path(ctx)
        extra = ctx['method'].extra_args
        if str(extra.get('checkpoint-format', 'deployable_v1')) == 'deployable_v1':
            return path / deployment_subdir(int(extra.get('coeff-bits', 14)),
                str(extra.get('storage-dtype', 'float16')), str(extra.get('factor-scaling', 'minimum')))
        return path

    def run(self, ctx):
        cmd = self.build_command(ctx)
        factors_only = bool(ctx['method'].extra_args.get('artifacts-only', False))
        if factors_only and getattr(getattr(ctx['cfg'], 'stages', None), 'evaluate', False):
            raise ValueError('Per-layer factor runs require stages.evaluate=false.')
        if not 0.0 <= float(ctx['cfg'].compression_ratio) < 1.0:
            raise ValueError('compression_ratio must be in [0, 1)')
        if not ctx['dry_run'] and not Path(ctx['calib_root']).is_dir():
            raise FileNotFoundError(
                f"Calibration directory not found: {ctx['calib_root']}. "
                'Set COSPADI_CALIB_ROOT or collect statistics with stages.collect=true.')
        output = run_checked(cmd, cwd=ctx['repo_root'], dry_run=ctx['dry_run'],
                             log_path=ctx['run_dir']/'compress.log')
        if ctx['dry_run']:
            return {'command': cmd, 'model_path': None, 'backend': 'cospadi-ksvd', 'status': 'planned'}
        if factors_only:
            path = self.factor_path(ctx)
            if not any(path.glob('model.layers.*_u.pth')):
                raise FileNotFoundError(f'No layer factors produced: {path}')
            return {'command': cmd, 'model_path': None, 'factor_path': str(path),
                    'backend': 'cospadi-ksvd', 'status': 'factors-only'}
        path = self.checkpoint_path(ctx)
        if not (path / 'config.json').is_file():
            raise FileNotFoundError(f'CoSpaDi did not produce the expected checkpoint: {path}')
        return {'command': cmd, 'model_path': str(path), 'backend': 'cospadi-ksvd'}
