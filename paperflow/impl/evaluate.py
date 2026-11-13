"""Paper evaluation: coefficient BF16 mantissa truncation followed by lm-eval."""
import argparse
import json
import os
from pathlib import Path
from os.path import join
import re

import torch
from safetensors import safe_open

def load_safetensors_tensors(folder_path):
    """
    Load all tensors from safetensors files in the specified folder.
    Handles both:
    - Single file: model.safetensors
    - Sharded files: model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc.
    
    Args:
        folder_path (str): Path to the folder containing safetensors files
        
    Returns:
        dict: Dictionary mapping tensor names to tensor objects
    """
    tensors = {}
    
    # Get all .safetensors files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.safetensors')]
    
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {folder_path}")
    
    # Check for sharded pattern: model-XXXXX-of-YYYYY.safetensors
    sharded_pattern = re.compile(r'model-(\d+)-of-(\d+)\.safetensors$')
    sharded_files = []
    single_model_file = None
    
    for file in files:
        match = sharded_pattern.match(file)
        if match:
            sharded_files.append((file, int(match.group(1)), int(match.group(2))))
        elif file == "model.safetensors":
            single_model_file = file
    
    # Sort sharded files by shard number
    sharded_files.sort(key=lambda x: x[1])
    
    if single_model_file and sharded_files:
        raise ValueError(f"Found both single model.safetensors and sharded files in {folder_path}. Please use one format only.")
    
    if single_model_file:
        # Load single file
        file_path = join(folder_path, single_model_file)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        print(f"Loaded tensors from single file: {single_model_file}")
        
    elif sharded_files:
        # Validate sharded files
        total_shards = sharded_files[0][2]  # Y from model-X-of-Y
        expected_shard_numbers = set(range(1, total_shards + 1))
        actual_shard_numbers = set(shard[1] for shard in sharded_files)
        
        if actual_shard_numbers != expected_shard_numbers:
            missing = expected_shard_numbers - actual_shard_numbers
            extra = actual_shard_numbers - expected_shard_numbers
            error_msg = f"Incomplete sharded model in {folder_path}. "
            if missing:
                error_msg += f"Missing shards: {sorted(missing)}. "
            if extra:
                error_msg += f"Unexpected shards: {sorted(extra)}. "
            raise ValueError(error_msg)
        
        # Load all shards
        for file_name, shard_num, total in sharded_files:
            file_path = join(folder_path, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    # Check for duplicate keys across shards (shouldn't happen in properly sharded models)
                    if k in tensors:
                        print(f"Warning: Duplicate tensor key '{k}' found in {file_name}. Overwriting previous value.")
                    tensors[k] = f.get_tensor(k)
            print(f"Loaded shard {shard_num}/{total} from {file_name}")
            
    else:
        # No recognized pattern - try to load any .safetensors file
        print("Warning: No standard model.safetensors or sharded files found. Attempting to load all .safetensors files...")
        for file in files:
            file_path = join(folder_path, file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in tensors:
                        print(f"Warning: Duplicate tensor key '{k}' found in {file}. Overwriting previous value.")
                    tensors[k] = f.get_tensor(k)
            print(f"Loaded tensors from {file}")
    
    print(f"Total tensors loaded: {len(tensors)}")
    return tensors

def bf16_truncate_frac_bits(x: torch.Tensor, n: int = 1) -> torch.Tensor:
    """
    Truncate (zero) the lowest n mantissa bits of a bfloat16 tensor.
    This is truncation (not rounding). NaNs/±Inf are preserved unchanged.
    n ∈ [1, 7].
    """
    assert x.dtype == torch.bfloat16, "x must be torch.bfloat16"
    assert 1 <= n <= 7, "bfloat16 mantissa has 7 bits"

    i16 = x.view(torch.int16)
    i32 = i16.to(torch.int32)

    bits = i32 & 0xFFFF
    exp  = (bits >> 7) & 0xFF
    is_special = exp.eq(0xFF)  # NaN/Inf

    mask_clear = (~((1 << n) - 1)) & 0xFFFF
    bits_trunc = torch.where(is_special, bits, bits & mask_clear)

    return bits_trunc.to(torch.int16).view(torch.bfloat16)

def fnv1a_32(data: str) -> int:
    """32-bit FNV-1a hash, deterministic, no imports."""
    h = 0x811c9dc5
    for c in data:
        h ^= ord(c)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def _float32_logits(model_call):
    """Promote logits for FP32 likelihood scoring without changing model compute."""
    def call(*args, **kwargs):
        return model_call(*args, **kwargs).to(torch.float32)
    return call

def evaluate(model_path: str, results_dir: str):
    # Run the zero-shot benchmark suite.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = ["mmlu", "arc_easy", "arc_challenge", "hellaswag", "piqa",
             "lambada_openai", "race", "wikitext", "sciq"]
    hflm = HFLM(pretrained=model_path, tokenizer=model_path, batch_size=1,
                max_batch_size=8, use_fast_tokenizer=False, trust_remote_code=True)
    config = json.loads((Path(model_path) / "config.json").read_text())
    packed = config.get("cospadi_checkpoint_format", config.get("compot_checkpoint_format")) == "packed_mask_v1"
    # Long FP16 WikiText
    # token-score sums can overflow even when every token log-probability is
    # finite. The public harness option computes softmax/scores in FP32.
    first_tasks = [task for task in tasks if task != "wikitext"] if packed else tasks
    results = lm_eval.simple_evaluate(
        hflm, tasks=first_tasks, num_fewshot=0, batch_size=1,
        max_batch_size=8, device="cuda", confirm_run_unsafe_code=True)
    if packed:
        hflm._model_call = _float32_logits(hflm._model_call)
        wiki = lm_eval.simple_evaluate(
            hflm, tasks=["wikitext"], num_fewshot=0, batch_size=1,
            max_batch_size=8, device="cuda", confirm_run_unsafe_code=True)
        results["results"].update(wiki["results"])
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    if packed:
        (output / "scoring_protocol.json").write_text(json.dumps({
            "model_compute_dtype": "checkpoint dtype; unchanged",
            "accuracy_scoring": "harness defaults",
            "wikitext_softmax_dtype": "float32",
            "reason": "Prevent long-document FP16 score reduction overflow; model weights and forward arithmetic unchanged"
        }, indent=2))
    (output / "results.json").write_text(
        json.dumps(results["results"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(results["results"])


def quantize_coefficients(checkpoint: str, bits: int, output_root: str) -> str:
    """Reconstruct fused weights from the separately stored D/C factors.

    This uses the exact original operation:
    v_quant = bf16_truncate_frac_bits(v, 16 - bits)
    new_weight = (v_quant @ u).cpu()
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not 9 <= bits <= 15:
        raise ValueError("Coefficient precision must be between 9 and 15 bits")
    checkpoint = str(Path(checkpoint).resolve())
    if checkpoint.endswith("_separate"):
        checkpoint = checkpoint[:-len("_separate")]
    separate = checkpoint + "_separate"
    if not Path(separate).is_dir():
        raise FileNotFoundError(f"Separate factor checkpoint missing: {separate}")
    hash_input = f"{separate}_{bits}"
    hash_suffix = f"{fnv1a_32(hash_input):08x}"
    out_path = str(Path(output_root) / (Path(checkpoint).name + f"_c_trunc_fp{bits}_{hash_suffix}"))
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cpu")
    model_state_dict = model.state_dict()
    num_layers = len(model.model.layers)
    tensors_separate = load_safetensors_tensors(separate)
    tensors = load_safetensors_tensors(checkpoint)
    keys_used = []
    for layer_idx in range(num_layers):
        for layer_type in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate_proj"]:
            full_name_weight = f"model.layers.{layer_idx}.{layer_type}.weight"
            full_name_u = f"model.layers.{layer_idx}.{layer_type}_u.weight"
            full_name_v = f"model.layers.{layer_idx}.{layer_type}_v.weight"
            keys_used.append(full_name_weight)
            original_weight = tensors[full_name_weight].to(torch.bfloat16)
            u = tensors_separate[full_name_u].to(torch.bfloat16).cuda()
            v = tensors_separate[full_name_v].to(torch.bfloat16).cuda()
            v_quant = bf16_truncate_frac_bits(v, 16 - bits)
            assert not torch.any(torch.isnan(v_quant)), "NaNs in v_quant!"
            new_weight = (v_quant @ u).cpu()
            assert not torch.any(torch.isnan(new_weight)), "NaNs in new_weight!"
            assert original_weight.shape == new_weight.shape, (
                f"Shape mismatch: {original_weight.shape} != {new_weight.shape}")
            tensors_separate[full_name_weight] = new_weight
            tensors_separate.pop(full_name_u)
            tensors_separate.pop(full_name_v)
    for key in model_state_dict.keys():
        if key not in keys_used:
            tensors_separate[key] = model_state_dict[key]
    model.load_state_dict(tensors_separate, strict=True)
    model = model.to(torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CoSpaDi paper evaluation")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--quant_coeffs", type=int, default=14)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    import numpy as np, random
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    model_path = args.ckpt_path
    config = json.loads((Path(model_path) / 'config.json').read_text())
    packed = config.get('cospadi_checkpoint_format', config.get('compot_checkpoint_format')) == 'packed_mask_v1'
    if packed and args.quant_coeffs is not None:
        if config.get('cospadi_coefficient_bits') != args.quant_coeffs:
            raise ValueError('Packed artifact coefficient precision differs from evaluation request')
    if args.quant_coeffs is not None and not packed:
        output_root = os.environ.get("COSPADI_QUANT_OUTPUT_ROOT", str(Path(args.results_dir) / "quantized"))
        model_path = quantize_coefficients(args.ckpt_path, args.quant_coeffs, output_root)
    evaluate(model_path, args.results_dir)


if __name__ == "__main__":
    main()
