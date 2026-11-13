"""Hydra CLI. All runtime components are instantiated from config."""
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

_package_root = Path(__file__).resolve().parent
_config_root = _package_root / 'configs'
if not _config_root.is_dir():
    _config_root = _package_root.parent / 'configs'

@hydra.main(version_base=None,config_path=str(_config_root),config_name='config')
def main(cfg: DictConfig):
    runner=instantiate(cfg.pipeline)
    result=runner.run(cfg)
    print('PIPELINE_RESULT:',result['model_path'] or '(no checkpoint)')

if __name__=='__main__':main()
