from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def test_reporters_are_independent_configurations():
    cfg = OmegaConf.load(ROOT/'configs/config.yaml')
    assert {'reporter': ['json']} in cfg.defaults
    assert not (ROOT/'configs/reporting/default.yaml').exists()
    for name in ('json','tensorboard','clearml'):
        config_file=ROOT/'configs/reporter'/f'{name}.yaml'
        assert config_file.is_file()
        assert f'# @package reporters.{name}' in config_file.read_text()
        node=OmegaConf.load(config_file)
        assert node._target_.startswith('paperflow.reporting.')


def test_clearml_is_selected_not_merely_enabled_by_index():
    config=OmegaConf.load(ROOT/'configs/reporter/clearml.yaml')
    assert config.enabled is True
    assert config.reuse_last_task_id is False
