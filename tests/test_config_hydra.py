from pathlib import Path
import pytest
pytest.importorskip('hydra')
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model', ['qwen3_06b', 'qwen3_8b', 'llama32_1b', 'llama3_8b'])
def test_all_models_compose(model):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs')):
        c = compose(config_name='config', overrides=[f'model={model}', 'dry_run=true'])
        p = instantiate(c.pipeline)
        assert p.method.name == 'ksvd'
        assert c.model.name


@pytest.mark.parametrize('selection,expected', [
    ('reporter=json', ['JsonReporter']),
    ('reporter=tensorboard', ['TensorBoardReporter']),
    ('reporter=clearml', ['ClearMLReporter']),
    ('reporter=[json,tensorboard]', ['JsonReporter', 'TensorBoardReporter']),
    ('reporter=[json,clearml,tensorboard]', ['JsonReporter','ClearMLReporter','TensorBoardReporter']),
    ('reporter=[]', []),
])
def test_selectable_reporters(selection, expected):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs')):
        c = compose(config_name='config', overrides=[selection, 'dry_run=true'])
        p = instantiate(c.pipeline)
        assert [type(rep).__name__ for rep in p.reporter.reporters] == expected


@pytest.mark.parametrize('method', ['ksvd', 'ksvd_legacy', 'ksvd_opp'])
def test_reject_non_cospadi_groups(method):
    from hydra.errors import MissingConfigException
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs')):
        with pytest.raises(MissingConfigException):
            compose(config_name='config', overrides=[f'method={method}'])


@pytest.mark.parametrize('model', ['qwen3_06b', 'qwen3_8b', 'llama32_1b', 'llama3_8b'])
@pytest.mark.parametrize('suffix,ratio', [('cr02', 0.2), ('cr03', 0.3), ('cr04', 0.4)])
def test_experiment_selects_complete_paper_recipe(model, suffix, ratio):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs')):
        cfg = compose(config_name='config', overrides=[f'experiments={model}_cospadi_{suffix}', 'dry_run=true'], return_hydra_config=True)
        assert cfg.hydra.runtime.choices.method == 'ksvd_deployable'
        pipeline = instantiate(cfg.pipeline)
        assert cfg.model.slug == model
        assert cfg.compression_ratio == pytest.approx(ratio)
        assert cfg.method.name == 'ksvd'
        assert cfg.method.iterations == 60
        assert cfg.method.power_iterations == 8
        assert cfg.method.ks_ratio == (1.6 if model == 'llama32_1b' else 2.0)
        assert cfg.evaluation.quant_coeffs == 14
        assert cfg.method.extra_args['checkpoint-format'] == 'deployable_v1'
        assert cfg.method.extra_args['storage-dtype'] == 'float16'
        assert cfg.method.extra_args['coeff-bits'] == 14
        assert cfg.method.extra_args['factor-scaling'] == ('balanced' if model == 'qwen3_8b' else 'minimum')
        assert not cfg.stages.collect and cfg.stages.compress and cfg.stages.evaluate
        assert pipeline.method.name == 'ksvd'


def test_experiment_switch_and_reporter_list():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs')):
        cfg = compose(config_name='config', overrides=['experiments=llama3_8b_cospadi_cr02',
                    'reporter=[json,tensorboard]', 'dry_run=true'])
        pipeline = instantiate(cfg.pipeline)
        assert cfg.model.slug == 'llama3_8b'
        assert [type(r).__name__ for r in pipeline.reporter.reporters] == [
            'JsonReporter', 'TensorBoardReporter']
