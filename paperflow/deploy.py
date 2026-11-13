"""Convert original CoSpaDi factors into a deployable packed Transformers model.

Supports legacy ``*_separate`` HF checkpoints and per-projection ``*_u.pth`` /
``*_v.pth`` factors beside a fused HF checkpoint. No factorization is inferred
from fused weights: original factors must be present.
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards
from .modelings.cospadi_sparse import pack_weight
from .impl.evaluate import bf16_truncate_frac_bits

PROJECTION = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|gate|down)_proj))\.weight$"
)
FACTOR = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|gate|down)_proj))_([uv])\.weight$"
)


class TensorReader:
    def __init__(self, root):
        self.root = Path(root)
        self.paths = {}
        self.bins = {}
        for file in sorted(self.root.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in self.paths:
                        raise ValueError(f"Duplicate tensor {key}")
                    self.paths[key] = file
        if not self.paths:
            for file in self.root.glob("pytorch_model*.bin"):
                data = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
                self.bins.update(data)
        self.keys = set(self.paths) | set(self.bins)
        if not self.keys:
            raise ValueError("No Hugging Face tensor checkpoint found")

    def get(self, key):
        if key in self.bins:
            return self.bins[key]
        with safe_open(self.paths[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def balance_fp16_range(u, v, policy="minimum"):
    """Reversibly shift binary exponents between corresponding factor atoms.

    Applied after coefficient truncation. Powers of two preserve each BF16
    product; no rank, sparsity pattern, or solver output file is changed.
    """
    limit = float(torch.finfo(torch.float16).max)
    peak_u = u.abs().amax(dim=1).double()
    peak_v = v.abs().amax(dim=0).double()
    tiny = 1e-300  # Avoid overflow in limit / peak for an unused zero atom.
    lower = torch.ceil(torch.log2(peak_v.clamp_min(tiny) / limit)).to(torch.int32)
    upper = torch.floor(torch.log2(limit / peak_u.clamp_min(tiny))).to(torch.int32)
    if torch.any(lower > upper):
        raise ValueError("No power-of-two factor scaling can fit both factors in FP16")
    if policy not in ("minimum", "balanced"):
        raise ValueError(f"Unknown factor scaling policy: {policy}")
    shifts = torch.zeros_like(lower)
    if policy == "balanced":
        active = (peak_u > 0) & (peak_v > 0)
        ideal = torch.round((torch.log2(peak_v.clamp_min(tiny)) -
                             torch.log2(peak_u.clamp_min(tiny))) / 2).to(torch.int32)
        shifts = torch.where(active, ideal, shifts)
    shifts = torch.minimum(torch.maximum(shifts, lower), upper)
    changed = torch.nonzero(shifts, as_tuple=True)[0]
    metadata = dict(policy=policy, atom_indices=changed.cpu().tolist(),
                    dictionary_exponent_shifts=shifts[changed].cpu().tolist(),
                    applied_after_coefficient_truncation=True,
                    operation="U[row] *= 2**shift; V[column] *= 2**(-shift)")
    if changed.numel():
        u = torch.ldexp(u, shifts[:, None]).contiguous()
        v = torch.ldexp(v, -shifts[None, :]).contiguous()
    return u, v, metadata


def convert_checkpoint(source, output, target_cr, factors=None, state_dict=None,
                       coeff_bits=14, storage_dtype="float16", factor_scaling="minimum"):
    if not 9 <= int(coeff_bits) <= 15:
        raise ValueError("Coefficient precision must be between 9 and 15 bits")
    if storage_dtype not in ("float16", "bfloat16"):
        raise ValueError("Storage dtype must be float16 or bfloat16")
    if factor_scaling not in ("minimum", "balanced"):
        raise ValueError("Factor scaling must be minimum or balanced")
    if factor_scaling == "balanced" and storage_dtype != "float16":
        raise ValueError("Balanced scaling is an FP16 export option")
    dtype = getattr(torch, storage_dtype)

    def cast(tensor):
        result = tensor.to(dtype).contiguous()
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite values after storage cast; do not export")
        return result
    source, output = Path(source), Path(output)
    factors = Path(factors) if factors else source
    if output.resolve() == source.resolve() or output.resolve() == factors.resolve():
        raise ValueError("Output must be separate from the source checkpoints")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    config = json.loads((source / "config.json").read_text())
    family = config["model_type"]
    if family not in ("llama", "qwen3"):
        raise ValueError(f"Unsupported architecture: {family}")
    if state_dict is None:
        reader = TensorReader(source)
    else:

        class StateReader:
            keys = set(state_dict)

            def get(self, key):
                return state_dict[key].detach().cpu()

        reader = StateReader()
    pairs = {}
    for key in reader.keys:
        match = FACTOR.fullmatch(key)
        if match:
            pairs.setdefault(match[1], {})[match[2]] = ("hf", key)
    for file in factors.glob("model.layers.*_u.pth"):
        name = file.name[: -len("_u.pth")]
        if not PROJECTION.fullmatch(name + ".weight"):
            continue
        pairs[name] = {"u": ("pth", file), "v": ("pth", file.with_name(name + "_v.pth"))}
    if not pairs:
        raise ValueError("Original U/V factors are required; fused weights cannot recover them")
    for name, pair in pairs.items():
        if set(pair) != {"u", "v"}:
            raise ValueError(f"Incomplete factor pair: {name}")
    dense_keys = {key for key in reader.keys if PROJECTION.fullmatch(key)}
    projection_names = {key[:-7] for key in dense_keys} | set(pairs)
    expected_count = int(config["num_hidden_layers"]) * 7
    if len(projection_names) != expected_count or set(pairs) != projection_names:
        raise ValueError(
            f"Require complete original factors for all {expected_count} projections; found {len(pairs)}"
        )
    state, specs, rows = {}, {}, []
    original_projection_bytes = packed_projection_bytes = 0
    for name in sorted(projection_names):
        key = name + ".weight"
        pair = pairs[name]
        values = {}
        for side in ("u", "v"):
            kind, location = pair[side]
            tensor = (
                reader.get(location)
                if kind == "hf"
                else torch.load(location, map_location="cpu", weights_only=True, mmap=True).T
            )
            values[side] = tensor.to(torch.bfloat16).contiguous()
        u, v = values["u"], values["v"]
        v = bf16_truncate_frac_bits(v, 16 - int(coeff_bits))
        if u.ndim != 2 or v.ndim != 2 or v.shape[1] != u.shape[0]:
            raise ValueError(f"Invalid factor shapes for {name}: {u.shape}, {v.shape}")
        if not torch.isfinite(u).all() or not torch.isfinite(v).all():
            raise ValueError(f"Non-finite factor values for {name}")
        k, in_features = u.shape
        out_features = v.shape[0]
        rescaling = None
        if storage_dtype == "float16":
            u, v, rescaling = balance_fp16_range(u, v, policy=factor_scaling)
        mask, nonzeros = pack_weight(v)
        u, nonzeros = cast(u), cast(nonzeros)
        dense_bytes = in_features * out_features * 2
        stored_bytes = tensor_bytes(u) + tensor_bytes(mask) + tensor_bytes(nonzeros)
        original_projection_bytes += dense_bytes
        state[name + "_u.weight"] = u
        state[name + "_v.packed_mask"] = mask
        state[name + "_v.values"] = nonzeros
        specs[name] = dict(
            k=k,
            in_features=in_features,
            out_features=out_features,
            nnz=nonzeros.numel(),
            storage="packed_mask_v1",
        )
        packed_projection_bytes += stored_bytes
        rows.append(
            dict(
                path=name,
                storage="packed_mask_v1",
                dense_bytes=dense_bytes,
                stored_bytes=stored_bytes,
                rank=k,
                nnz=nonzeros.numel(),
                fp16_range_rescaling=rescaling,
            )
        )
    for key in sorted(reader.keys):
        if PROJECTION.fullmatch(key) or FACTOR.fullmatch(key):
            continue
        renamed = key
        if key.endswith(".bias") and key[:-5] in specs:
            renamed = key[:-5] + "_v.bias"
        tensor = reader.get(key)
        state[renamed] = (
            cast(tensor)
            if tensor.is_floating_point()
            else tensor.contiguous()
        )
    if (
        config.get("tie_word_embeddings")
        and "lm_head.weight" in state
        and "model.embed_tokens.weight" in state
    ):
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError("Config requests tied embeddings, but lm_head differs from embeddings")
        del state["lm_head.weight"]
    unchanged_core_bytes = sum(
        tensor_bytes(t)
        for key, t in state.items()
        if key.startswith("model.layers.")
        and not any(
            key.startswith(name + "_") or key == name + ".weight" for name in projection_names
        )
    )
    dense_core_bytes = original_projection_bytes + unchanged_core_bytes
    packed_core_bytes = packed_projection_bytes + unchanged_core_bytes
    achieved = 1 - packed_core_bytes / dense_core_bytes
    if not 0 < achieved < 1:
        raise ValueError(f"No actual decoder-core compression: {achieved:.6%}")
    config.pop("compot_checkpoint_format", None)
    config.update(
        cospadi_checkpoint_format="packed_mask_v1",
        factor_shapes=specs,
        torch_dtype=storage_dtype,
        cospadi_coefficient_bits=int(coeff_bits),
        cospadi_fake_quantization="BF16 low mantissa bits truncated, then cast to storage dtype",
        cospadi_storage_dtype=storage_dtype,
        cospadi_factor_rescaling=f"Reversible power-of-two exponent shifts, policy={factor_scaling}; applied after coefficient truncation; see manifest",
        auto_map={
            "AutoModelForCausalLM": f"modeling_{family}_svd."
            + ("Llama" if family == "llama" else "Qwen3")
            + "ForCausalLM"
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    plan = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size="2GB"
    )
    for filename, keys in plan.filename_to_tensors.items():
        save_file({key: state[key] for key in keys}, output / filename, metadata={"format": "pt"})
    if plan.is_sharded:
        (output / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}, indent=2)
        )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for file in source.iterdir():
        if file.is_file() and (
            file.name.startswith(
                (
                    "tokenizer",
                    "special_tokens",
                    "added_tokens",
                    "vocab",
                    "merges",
                    "generation_config",
                    "chat_template",
                )
            )
            or file.name == "tokenizer.model"
        ):
            shutil.copy2(file, output / file.name)
    modeling_root = Path(__file__).parent / "modelings"
    for name in (f"modeling_{family}_svd.py", "cospadi_sparse.py"):
        shutil.copy2(modeling_root / name, output / name)
    report = dict(
        format="packed_mask_v1",
        source=str(source),
        factors=str(factors),
        nominal_compression_ratio=target_cr,
        coefficient_bits=int(coeff_bits),
        storage_dtype=storage_dtype,
        factor_scaling=factor_scaling,
        meets_nominal_core_cr=achieved >= float(target_cr),
        compression_note="Legacy CoSpaDi allocation ignores mask bytes; fake-quantized coefficients occupy 16 bits in this deployable format. Actual byte savings are reported without changing ranks.",
        achieved_core_cr=achieved,
        dense_core_bytes=dense_core_bytes,
        packed_core_bytes=packed_core_bytes,
        projection_cr=1 - packed_projection_bytes / original_projection_bytes,
        total_tensor_payload_bytes=sum(tensor_bytes(t) for t in state.values()),
        actual_safetensors_file_bytes=sum(f.stat().st_size for f in output.glob("*.safetensors")),
        scope="all decoder-layer parameters; embeddings and lm_head excluded from core CR; norms/biases included",
        runtime="Factors and packed sparse buffers remain persistent; one dense coefficient temporary per portable forward; no persistent dense or fused cache",
        factorized_projections=len(specs),
        projections=rows,
    )
    (output / "compression_manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--factors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-cr", type=float, required=True)
    parser.add_argument("--coeff-bits", type=int, default=14)
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--factor-scaling", choices=("minimum", "balanced"), default="minimum")
    args = parser.parse_args()
    report = convert_checkpoint(args.source, args.output, args.target_cr, args.factors, coeff_bits=args.coeff_bits, storage_dtype=args.storage_dtype, factor_scaling=args.factor_scaling)
    print(json.dumps({k: v for k, v in report.items() if k != "projections"}, indent=2))


if __name__ == "__main__":
    main()
