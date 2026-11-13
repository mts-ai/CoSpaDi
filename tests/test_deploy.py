import json
import torch
import pytest
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from paperflow.deploy import convert_checkpoint, TensorReader, PROJECTION
from paperflow.impl.evaluate import bf16_truncate_frac_bits

@pytest.mark.parametrize('family', ['llama', 'qwen3'])
@pytest.mark.parametrize('dtype,scaling', [('float16','minimum'), ('bfloat16','minimum'), ('float16','balanced')])
def test_packed_reload_and_generation(tmp_path, family, dtype, scaling):
    cls, model_cls = (LlamaConfig, LlamaForCausalLM) if family == 'llama' else (Qwen3Config, Qwen3ForCausalLM)
    torch.manual_seed(42)
    cfg = cls(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
              num_attention_heads=2, num_key_value_heads=1, head_dim=8,
              max_position_embeddings=128, tie_word_embeddings=True)
    model = model_cls(cfg).to(torch.bfloat16)
    source = tmp_path / 'source'
    model.save_pretrained(source)
    factors = tmp_path / 'factors'
    factors.mkdir()
    expected = {}
    for key, t in model.state_dict().items():
        if not PROJECTION.fullmatch(key): continue
        name = key[:-7]
        out_features, in_features = t.shape
        u = torch.randn(in_features, 2).double() * .1
        v = torch.randn(2, out_features).double() * .1
        v[0] = 0
        torch.save(u, factors / (name + '_u.pth'))
        torch.save(v, factors / (name + '_v.pth'))
        a=u.T.to(torch.bfloat16).contiguous()
        b=bf16_truncate_frac_bits(v.T.to(torch.bfloat16).contiguous(),2)
        if scaling=='balanced':
            from paperflow.deploy import balance_fp16_range
            a,b,_=balance_fp16_range(a,b,policy=scaling)
        expected[name] = a.to(getattr(torch,dtype)), b.to(getattr(torch,dtype))
    output = tmp_path / 'cr_0.2' / 'transformers'
    report = convert_checkpoint(source, output, .2, factors, storage_dtype=dtype, factor_scaling=scaling,
        state_dict=model.state_dict() if family == 'llama' and dtype == 'float16' else None)
    assert report['achieved_core_cr'] > 0
    assert report['factorized_projections'] == 7
    reader = TensorReader(output)
    assert not any(PROJECTION.fullmatch(k) for k in reader.keys)
    assert all(not reader.get(k).is_floating_point() or str(reader.get(k).dtype) == 'torch.' + dtype for k in reader.keys)
    loaded, info = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True,
        torch_dtype=getattr(torch, dtype), output_loading_info=True, attn_implementation='eager')
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    for name, (u, v) in expected.items():
        up, vp = loaded.get_submodule(name + '_u'), loaded.get_submodule(name + '_v')
        assert torch.equal(up.weight, u)
        assert vp.values.dtype == getattr(torch, dtype), (name, vp.values.dtype, dtype)
        assert torch.equal(vp.weight, v)
    x = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        assert torch.isfinite(loaded(x).logits).all()
        assert loaded.generate(x, max_new_tokens=2, do_sample=False).shape[1] == 5
    assert callable(loaded.optimize)
    reloaded = tmp_path / 'reload'
    loaded.save_pretrained(reloaded)
    again = AutoModelForCausalLM.from_pretrained(reloaded, trust_remote_code=True,
        torch_dtype=getattr(torch, dtype), attn_implementation='eager')
    for k, t in loaded.state_dict().items():
        assert torch.equal(t, again.state_dict()[k]), k

def test_incomplete_factors_rejected(tmp_path):
    source = tmp_path / 'source'
    LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).save_pretrained(source)
    factors = tmp_path / 'factors'
    factors.mkdir()
    torch.save(torch.ones(16, 2), factors / 'model.layers.0.self_attn.q_proj_u.pth')
    torch.save(torch.ones(2, 16), factors / 'model.layers.0.self_attn.q_proj_v.pth')
    with pytest.raises(ValueError, match='Require complete'):
        convert_checkpoint(source, tmp_path / 'output', .2, factors)

def test_storage_overflow_rejected(tmp_path):
    source = tmp_path / 'source'
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    model.save_pretrained(source)
    factors = tmp_path / 'factors'
    factors.mkdir()
    for key, tensor in model.state_dict().items():
        if not PROJECTION.fullmatch(key): continue
        torch.save(torch.full((tensor.shape[1], 2), 1e8), factors / (key[:-7] + '_u.pth'))
        torch.save(torch.full((2, tensor.shape[0]), 1e8), factors / (key[:-7] + '_v.pth'))
    with pytest.raises(ValueError, match='No power-of-two'):
        convert_checkpoint(source, tmp_path / 'output', .2, factors)

def test_binary_range_rebalance_is_reversible_and_preserves_product():
    from paperflow.deploy import balance_fp16_range
    u=torch.tensor([[.001, -.002, .003], [100000., 200000., -100000.]],dtype=torch.bfloat16)
    v=torch.tensor([[200000., .002],[-100000., .004],[0.,-.008]],dtype=torch.bfloat16)
    original=u.double(),v.double()
    a,b,meta=balance_fp16_range(u,v)
    assert torch.isfinite(a.half()).all() and torch.isfinite(b.half()).all()
    assert torch.equal(original[1]@original[0],b.double()@a.double())
    assert meta['atom_indices']==[0,1]
    shifts=torch.tensor(meta['dictionary_exponent_shifts'])
    assert torch.equal(torch.ldexp(a,-shifts[:,None]),u)
    assert torch.equal(torch.ldexp(b,shifts[None,:]),v)
    a,b,_=balance_fp16_range(torch.zeros_like(u),v)
    assert torch.isfinite(a.half()).all() and torch.isfinite(b.half()).all()

def test_backend_deployable_path_and_flags(fake_context):
    from paperflow.compression import CoSpaDiBackend
    fake_context['method'].extra_args.update({'checkpoint-format': 'deployable_v1', 'coeff-bits': 14, 'storage-dtype': 'float16'})
    cmd = CoSpaDiBackend().build_command(fake_context)
    assert cmd[cmd.index('--checkpoint-format') + 1] == 'deployable_v1'
    assert CoSpaDiBackend().checkpoint_path(fake_context).name == 'transformers_fp14_float16'


def test_balanced_scaling_reduces_underflow_without_changing_product():
    from paperflow.deploy import balance_fp16_range
    u=torch.tensor([[2.**-30, -2.**-28], [0., 0.]],dtype=torch.bfloat16)
    v=torch.tensor([[2.**8, 2.**4],[-2.**9, 0.]],dtype=torch.bfloat16)
    a,b,meta=balance_fp16_range(u,v,policy='balanced')
    assert (u.half()==0).sum() > (a.half()==0).sum()
    assert torch.equal(b.double()@a.double(),v.double()@u.double())
    assert torch.equal(b.half().double()@a.half().double(),v.double()@u.double())
    assert meta['policy']=='balanced' and meta['atom_indices']==[0]
    assert torch.equal(b[:,1],v[:,1])  # Unused dictionary atoms keep their coefficients.
    shifts=torch.zeros(u.shape[0],dtype=torch.int32)
    shifts[meta['atom_indices']]=torch.tensor(meta['dictionary_exponent_shifts'],dtype=torch.int32)
    assert torch.equal(torch.ldexp(a,-shifts[:,None]),u)
    assert torch.equal(torch.ldexp(b,shifts[None,:]),v)
