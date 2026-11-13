import pytest
from paperflow.compression import CoSpaDiBackend
from paperflow.evaluation import CoSpaDiEvaluation, LMEvalHarness
from paperflow.methods import MethodSpec


def test_old_method_commands(fake_context):
    args = CoSpaDiBackend().build_command(fake_context)
    assert args[args.index('--ksvd-iterations') + 1] == '60'
    assert args[args.index('--power-iterations') + 1] == '8'
    assert '--dynamic' not in args
    assert '--method' in args and args[args.index('--method') + 1] == 'ksvd'
    assert '--model-name' in args
    assert args[1:4] == ['-m', 'paperflow.impl.compress', '--method']


def test_unsupported_method_rejected(fake_context):
    class InvalidMethod:
        name = 'ksvd_opp'
    fake_context['method'] = InvalidMethod()
    with pytest.raises(ValueError, match='only supports ksvd'):
        CoSpaDiBackend().build_command(fake_context)


def test_coefficient_quantization_eval(fake_context):
    result = CoSpaDiEvaluation(quant_coeffs=14).run(fake_context)
    assert result['command'][-2:] == ['--quant_coeffs', '14']
    assert result['results_root'] == str(fake_context['run_dir']/'evaluation')


def test_lm_eval_dedicated_output(fake_context):
    cmd = LMEvalHarness(tasks=['wikitext']).build_command(fake_context)
    assert 'wikitext' in cmd and '--output_path' in cmd


def test_checkpoint_path_is_specific_to_model_and_settings(fake_context):
    backend = CoSpaDiBackend()
    path = backend.checkpoint_path(fake_context)
    assert path.name == 'transformers_fp14_float16'
    assert path.parent.name == 'Qwen_Qwen3-0.6B_cr_0.2_ksvd_ks_2.0_ksvd_it_60_power_it_8'
    assert path.parent.parent == fake_context['checkpoint_root']


def test_missing_calibration_fails_early(fake_context):
    fake_context['dry_run'] = False
    with pytest.raises(FileNotFoundError, match='Calibration directory'):
        CoSpaDiBackend().run(fake_context)


def test_invalid_compression_ratio_rejected(fake_context):
    fake_context['cfg'].compression_ratio = 1.1
    with pytest.raises(ValueError, match='compression_ratio'):
        CoSpaDiBackend().run(fake_context)
