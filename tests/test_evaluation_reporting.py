import json
from pathlib import Path
from paperflow.evaluation import collect_eval_metrics,LMEvalHarness
from paperflow.reporting import JsonReporter,CompositeReporter

def test_extract_legacy_results(tmp_path):
    data={'wikitext':{'word_perplexity,none':9.3},'piqa':{'acc,none':.63}}
    (tmp_path/'results.json').write_text(json.dumps(data))
    metrics=collect_eval_metrics(tmp_path)
    assert metrics['wikitext/word_perplexity,none']==9.3

def test_extract_harness_results(tmp_path):
    folder=tmp_path/'lm_eval';folder.mkdir()
    (folder/'results_2026.json').write_text(json.dumps({'results':{'mmlu':{'acc,none':.4}}}))
    assert collect_eval_metrics(folder)['mmlu/acc,none']==.4

def test_unknown_results(tmp_path):assert collect_eval_metrics(tmp_path)=={}

def test_no_forced_fewshot(fake_context):
    cmd=LMEvalHarness().build_command(fake_context)
    assert '--num_fewshot' not in cmd
    cmd=LMEvalHarness(num_fewshot=0).build_command(fake_context)
    assert cmd[cmd.index('--num_fewshot')+1]=='0'

def test_json_reporter(fake_context):
    fake_context['run_dir'].mkdir()
    CompositeReporter([JsonReporter()]).report(fake_context,{'metrics':{'x':1}})
    assert json.loads((fake_context['run_dir']/'results.json').read_text())['metrics']['x']==1


def test_composite_accepts_named_reporters(tmp_path):
    import json
    from paperflow.reporting import CompositeReporter, JsonReporter
    ctx={'run_dir': tmp_path}
    CompositeReporter({'json': JsonReporter(filename='named.json')}).report(ctx, {'metrics': {'accuracy': .75}})
    assert json.loads((tmp_path/'named.json').read_text())['metrics']['accuracy'] == .75


def test_composite_rejects_bad_reporter():
    import pytest
    from paperflow.reporting import CompositeReporter
    with pytest.raises(TypeError, match='Invalid reporter'):
        CompositeReporter({'bad': object()})
