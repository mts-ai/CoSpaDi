# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
modeling_qwen3_svd.py – CoSpaDi checkpoint loader with MACKO sparse runtime.

Checkpoint compatibility:
  Loads ``*_u.weight`` / ``*_v.weight`` tensors for all Qwen3 linear
  projections: q/k/v/o and gate/up/down.  Portable inference uses ordinary
  dense U/V products and is mathematically equivalent to the fused checkpoint.
  ``model.optimize()`` remains available for the MACKO-compatible up+gate-only
  MLP case (factorized up/gate with dense down_proj).
"""

from typing import Any, Callable, Optional, Union
import asyncio
import torch
from torch import nn
import torch.nn.functional as F

class PackedSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, nnz, bias=False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer('packed_mask', torch.zeros((in_features*out_features+7)//8, dtype=torch.uint8))
        self.register_buffer('values', torch.zeros(int(nnz), dtype=torch.get_default_dtype()))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def weight(self):
        shifts = torch.arange(8, dtype=torch.uint8, device=self.packed_mask.device)
        mask = ((self.packed_mask[:, None] >> shifts) & 1).reshape(-1)
        mask = mask[:self.in_features*self.out_features].bool()
        dense = self.values.new_zeros(self.in_features*self.out_features)
        return dense.masked_scatter_(mask, self.values).view(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, nnz={self.values.numel()}, packed_mask_bytes={self.packed_mask.numel()}'

try:
    from macko_spmv import LinearSparseGEMV
except:
    print("MACKO was not imported")

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

logger = logging.get_logger(__name__)


# =============================================================================
# ASYNC UTILITY
# =============================================================================

async def run_inference_async(
    func: Callable,
    *args,
    device: Optional[torch.device] = None,
    **kwargs,
):
    """Offload one complete forward call to a thread pool executor."""
    loop = asyncio.get_running_loop()

    def _call():
        with torch.inference_mode():
            return func(*args, **kwargs)

    result = await loop.run_in_executor(None, _call)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return result


# =============================================================================
# CHECKPOINT LOADING UTILITIES (unchanged logic, needed for from_pretrained)
# =============================================================================

def get_k_and_sparsity(cr, d1, d2, L, ks_ratio):
    """Legacy fallback for rank inference from compression ratio."""
    vanilla = d1 * d2 * L
    k = (1 - cr) * d1 * d2 * L / (d1 + d2 * L / ks_ratio)
    sparsity = k / ks_ratio
    return int(k), 1 - int(sparsity) / (int(k) + 1e-8)


_MLP_PROJECTIONS = frozenset({"gate_proj", "up_proj", "down_proj"})
_ATTN_PROJECTIONS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


def _projection_path_candidates(layer_idx: int, projection: str) -> tuple[str, ...]:
    """Return config-metadata keys that can identify a projection.

    CoSpaDi serializes exact factor ranks using the real module path, e.g.
    ``model.layers.0.mlp.up_proj`` or
    ``model.layers.0.self_attn.q_proj``.  The old loader hard-coded ``mlp``
    here, which made all attention factors invisible to ``from_pretrained``.
    """
    if projection in _MLP_PROJECTIONS:
        block = "mlp"
    elif projection in _ATTN_PROJECTIONS:
        block = "self_attn"
    else:
        raise ValueError(f"Unsupported Qwen3 projection {projection!r}")

    suffix = f"layers.{layer_idx}.{block}.{projection}"
    return (
        f"model.{suffix}",
        suffix,
        f"model.model.{suffix}",
        f"transformer.{suffix}",
    )


def _lookup_mapping_value(config, attribute: str, candidates: tuple[str, ...]):
    mapping = getattr(config, attribute, None)
    if not isinstance(mapping, dict):
        return None
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _as_int_rank(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("k", "rank", "svd_rank", "allocation_k"):
            if key in value:
                return int(value[key])
        return None
    return int(value)


def _resolve_ks_ratio(config, candidates: tuple[str, ...]) -> float:
    # Realized serialized ratio is authoritative. Requested overrides are the
    # next fallback, followed by historical config formats.
    for attribute in (
        "allocation_ks_ratio",
        "ks_ratio_overrides",
        "svd_ks_ratio",
        "ks_ratio",
    ):
        value = _lookup_mapping_value(config, attribute, candidates)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("ks_ratio", value.get("value"))
        if value is not None:
            return float(value)

    value = getattr(config, "ks_ratio_default", None)
    if value is None:
        value = getattr(config, "ks_ratio", 2.0)
    if isinstance(value, dict):
        return 2.0
    return float(value)

def _resolve_factor_spec(
    config,
    layer_idx: int,
    projection: str,
    in_features: int,
    out_features: int,
) -> Optional[dict[str, Any]]:
    """Resolve exact factor rank for checkpoint-compatible parameter creation."""
    candidates = _projection_path_candidates(layer_idx, projection)

    factor_shapes = getattr(config, "factor_shapes", None)
    if getattr(config, "cospadi_checkpoint_format", getattr(config, "compot_checkpoint_format", None)) == "packed_mask_v1" and not any(path in (factor_shapes or {}) for path in candidates):
        return None
    if isinstance(factor_shapes, dict):
        for path in candidates:
            raw = factor_shapes.get(path)
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raw = {"k": raw}
            k = _as_int_rank(raw)
            if k is None:
                raise ValueError(f"factor_shapes[{path!r}] missing k/rank: {raw}")
            if k <= 0:
                raise ValueError(f"Invalid factor rank k={k} for {path}")
            spec = dict(raw)
            stored_in = spec.get("in_features")
            stored_out = spec.get("out_features")
            if stored_in is not None and int(stored_in) != int(in_features):
                raise ValueError(
                    f"factor_shapes[{path!r}] has in_features={stored_in}, "
                    f"but the Qwen3 architecture expects {in_features}"
                )
            if stored_out is not None and int(stored_out) != int(out_features):
                raise ValueError(
                    f"factor_shapes[{path!r}] has out_features={stored_out}, "
                    f"but the Qwen3 architecture expects {out_features}"
                )
            spec.update(path=path, k=k, in_features=in_features, out_features=out_features)
            return spec

    for attribute in ("svd_rank", "rank_allocation", "allocation_k"):
        raw = _lookup_mapping_value(config, attribute, candidates)
        k = _as_int_rank(raw)
        if k is not None:
            if k <= 0:
                raise ValueError(f"Invalid {attribute}={k} for {candidates[0]}")
            return {"path": candidates[0], "k": k, "in_features": in_features,
                    "out_features": out_features, "source": attribute}

    raw_cr = _lookup_mapping_value(config, "allocation_cr", candidates)
    if raw_cr is not None:
        if isinstance(raw_cr, dict):
            raw_cr = raw_cr.get("cr", raw_cr.get("compression_ratio"))
        if raw_cr is not None and float(raw_cr) > 0.0:
            ks_ratio = _resolve_ks_ratio(config, candidates)
            k, _ = get_k_and_sparsity(float(raw_cr), in_features, out_features, 1, ks_ratio)
            return {"path": candidates[0], "k": k, "in_features": in_features,
                    "out_features": out_features, "ks_ratio": ks_ratio, "source": "allocation_cr"}

    return None


def _create_dense_or_factorized_projection(
    module: nn.Module,
    *,
    config,
    layer_idx: Optional[int],
    projection: str,
    in_features: int,
    out_features: int,
    bias: bool,
) -> bool:
    """Create checkpoint-compatible parameters for one linear projection.

    Returns ``True`` when ``<projection>_u``/``<projection>_v`` were created
    and ``False`` when a normal dense ``<projection>`` was created.

    CoSpaDi stores U as [in, k] and V as [k, out].  ``nn.Linear`` stores
    weights transposed, so the checkpoint contains ``U.T`` in the first
    module and ``V.T`` in the second one.  Consequently ``v(u(x))`` is
    exactly ``x @ U @ V``.  Any original bias belongs on the second factor.
    """
    spec = None
    if layer_idx is not None:
        spec = _resolve_factor_spec(
            config, layer_idx, projection, in_features, out_features
        )

    if spec is None:
        setattr(
            module,
            projection,
            nn.Linear(in_features, out_features, bias=bias),
        )
        return False

    k = int(spec["k"])
    setattr(
        module,
        f"{projection}_u",
        nn.Linear(in_features, k, bias=False),
    )
    setattr(
        module,
        f"{projection}_v",
        PackedSparseLinear(k, out_features, spec['nnz'], bias=bias)
        if spec.get('storage') == 'packed_mask_v1' else nn.Linear(k, out_features, bias=bias),
    )
    return True


def _apply_dense_or_factorized_projection(
    module: nn.Module, projection: str, x: torch.Tensor
) -> torch.Tensor:
    """Apply either the dense layer or its checkpoint U/V factors."""
    factorized = getattr(module, "factorized_projections", set())
    if projection in factorized:
        u = getattr(module, f"{projection}_u")
        v = getattr(module, f"{projection}_v")
        return v(u(x))
    return getattr(module, projection)(x)


# =============================================================================
# LEAN RUNTIME MLP (post-optimize, mirrors File 2 structure exactly)
# =============================================================================

class SparseQwen3MLP(nn.Module):
    """
    Fused-dense + direct-sparse MLP.

    Structure (per forward):
      1 dense GEMM:  x @ combined_U.T          -> [B*S, k_up+k_gate]
      1 chunk:       split into up_r, gate_r
      2 sparse GEMV: up_r @ up_V.T, gate_r @ gate_V.T
      1 dense GEMM:  (act(gate_out)*up_out) @ down_proj.T

    Total kernel launches: 4 (same as File 2).
    No wrapper classes, no metadata lookups, no reshape gymnastics.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, act_fn, config):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        self.config = config

        # These are populated by from_factorized(); declared here for clarity.
        self.combined_U: nn.Linear = None  # type: ignore[assignment]
        self.up_V: LinearSparseGEMV = None  # type: ignore[assignment]
        self.gate_V: LinearSparseGEMV = None  # type: ignore[assignment]
        self.down_proj: nn.Linear = None  # type: ignore[assignment]
        self.k_up: int = 0
        self.k_gate: int = 0

    @classmethod
    def from_factorized(cls, factorized_mlp: "Qwen3MLP", module_path: str) -> "SparseQwen3MLP":
        """
        Build the lean runtime from a checkpoint-loaded Qwen3MLP that has
        up_proj_u/v and gate_proj_u/v parameters.
        """
        config = factorized_mlp.config
        sparse = cls(config.hidden_size, config.intermediate_size,
                     ACT2FN[config.hidden_act], config)

        up_u: nn.Linear = factorized_mlp.up_proj_u
        gate_u: nn.Linear = factorized_mlp.gate_proj_u
        up_v: nn.Linear = factorized_mlp.up_proj_v
        gate_v: nn.Linear = factorized_mlp.gate_proj_v

        sparse.k_up = up_u.weight.shape[0]
        sparse.k_gate = gate_u.weight.shape[0]
        k_total = sparse.k_up + sparse.k_gate

        dtype = up_u.weight.dtype
        device = up_u.weight.device

        # --- Fused dense U: single [hidden -> k_up+k_gate] GEMM ---
        sparse.combined_U = nn.Linear(
            config.hidden_size, k_total, bias=False, device=device, dtype=dtype
        )
        with torch.no_grad():
            sparse.combined_U.weight.copy_(
                torch.cat([up_u.weight.detach(), gate_u.weight.detach()], dim=0)
            )

        # --- Direct sparse V: no wrapper, just the kernel ---
        sparse.up_V = LinearSparseGEMV(sparse.k_up, config.intermediate_size, bias=False)
        sparse.up_V.compress_weight(up_v.weight.detach().contiguous())

        sparse.gate_V = LinearSparseGEMV(sparse.k_gate, config.intermediate_size, bias=False)
        sparse.gate_V.compress_weight(gate_v.weight.detach().contiguous())

        # --- Down projection: reuse the already-loaded dense layer ---
        sparse.down_proj = factorized_mlp.down_proj

        return sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single dense projection (1 kernel launch)
        combined = self.combined_U(x)  # [..., k_up + k_gate]

        # Split (no kernel launch, just a view)
        up_r, gate_r = combined.split([self.k_up, self.k_gate], dim=-1)

        # Two sparse GEMV calls (2 kernel launches)
        up_out = self.up_V(up_r)
        gate_out = self.gate_V(gate_r)

        # Gated activation + down projection (1 kernel launch for down_proj)
        return self.down_proj(self.act_fn(gate_out) * up_out)

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
                f"k_up={self.k_up}, k_gate={self.k_gate}")


# =============================================================================
# STANDARD COMPONENTS (unchanged from upstream)
# =============================================================================

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            self.rope_type = rope_parameters.get("rope_type", "default")
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            else:
                self.rope_type = "default"

        self.rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            if self.rope_type not in ROPE_INIT_FUNCTIONS:
                raise ValueError(f"Unsupported RoPE type {self.rope_type!r}.")
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        del seq_len
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        else:
            base = getattr(config, "rope_theta", 10000.0)
        dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# CHECKPOINT-COMPATIBLE MLP (used during from_pretrained loading)
# =============================================================================

class Qwen3MLP(nn.Module):
    """Checkpoint-compatible Qwen3 MLP for dense or separate CoSpaDi factors.

    Each of ``gate_proj``, ``up_proj`` and ``down_proj`` is created either as
    its original dense ``nn.Linear`` or as ``*_u`` + ``*_v`` according to the
    exact per-projection rank metadata stored in ``config``.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.factorized_projections: set[str] = set()

        shapes = {
            "gate_proj": (self.hidden_size, self.intermediate_size),
            "up_proj": (self.hidden_size, self.intermediate_size),
            "down_proj": (self.intermediate_size, self.hidden_size),
        }

        for projection, (in_features, out_features) in shapes.items():
            if _create_dense_or_factorized_projection(
                self,
                config=config,
                layer_idx=layer_idx,
                projection=projection,
                in_features=in_features,
                out_features=out_features,
                bias=False,
            ):
                self.factorized_projections.add(projection)

    def is_factorized(self, projection: str) -> bool:
        return projection in self.factorized_projections

    def has_separate_factors(self) -> bool:
        return bool(self.factorized_projections)

    def is_macko_compatible(self) -> bool:
        """Current MACKO runtime supports factorized up+gate and dense down."""
        return (
            {"up_proj", "gate_proj"}.issubset(self.factorized_projections)
            and "down_proj" not in self.factorized_projections
        )

    def forward(self, x):
        gate = _apply_dense_or_factorized_projection(self, "gate_proj", x)
        up = _apply_dense_or_factorized_projection(self, "up_proj", x)
        hidden = self.act_fn(gate) * up
        return _apply_dense_or_factorized_projection(self, "down_proj", hidden)


# =============================================================================
# ATTENTION
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: Optional[torch.Tensor], scaling: float, dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Qwen3 attention with checkpoint-compatible dense or U/V projections."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.factorized_projections: set[str] = set()

        q_out = config.num_attention_heads * self.head_dim
        kv_out = config.num_key_value_heads * self.head_dim
        shapes = {
            "q_proj": (config.hidden_size, q_out),
            "k_proj": (config.hidden_size, kv_out),
            "v_proj": (config.hidden_size, kv_out),
            "o_proj": (q_out, config.hidden_size),
        }

        for projection, (in_features, out_features) in shapes.items():
            if _create_dense_or_factorized_projection(
                self,
                config=config,
                layer_idx=layer_idx,
                projection=projection,
                in_features=in_features,
                out_features=out_features,
                bias=bool(config.attention_bias),
            ):
                self.factorized_projections.add(projection)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            getattr(config, "sliding_window", None)
            if self.layer_type == "sliding_attention"
            else None
        )

    def is_factorized(self, projection: str) -> bool:
        return projection in self.factorized_projections

    def _project(self, projection: str, x: torch.Tensor) -> torch.Tensor:
        return _apply_dense_or_factorized_projection(self, projection, x)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor], past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None, **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self._project("q_proj", hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self._project("k_proj", hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self._project("v_proj", hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self._project("o_proj", attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False, cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =============================================================================
# MODEL CLASSES
# =============================================================================

@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Qwen3DecoderLayer, "attentions": Qwen3Attention}

    def optimize(self, verbose: bool = True) -> "Qwen3PreTrainedModel":
        """Replace factorized Qwen3MLP modules with lean SparseQwen3MLP."""
        return optimize(self, verbose=verbose)


def optimize(model: "Qwen3PreTrainedModel", verbose: bool = True) -> "Qwen3PreTrainedModel":
    """
    Convert checkpoint-loaded separate CoSpaDi factors into the lean
    fused-dense + direct-sparse runtime.

    After this call, each MLP forward is exactly 4 kernel launches:
      1× dense GEMM (combined_U), 2× sparse GEMV (up_V, gate_V), 1× dense GEMM (down_proj).
    """
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen3MLP) and module.is_macko_compatible()
    ]

    replaced = 0
    for name, module in targets:
        parent_name, _, module_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        sparse_mlp = SparseQwen3MLP.from_factorized(module, module_path=name)
        setattr(parent, module_name, sparse_mlp)
        replaced += 1

        if verbose:
            logger.info(
                "Converted %s -> SparseQwen3MLP (k_up=%d, k_gate=%d)",
                name, sparse_mlp.k_up, sparse_mlp.k_gate,
            )

    if verbose:
        if replaced:
            logger.info("MACKO optimization complete: %d MLP layers converted.", replaced)
        else:
            logger.warning(
                "No MACKO-compatible separate MLP factors found. Current optimized runtime "
                "requires factorized up_proj + gate_proj and a dense down_proj. "
                "Portable U/V loading still works for all Qwen3 projections."
            )
    return model


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        layer_types = getattr(self.config, "layer_types", [])
        self.has_sliding_layers = "sliding_attention" in layer_types
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    async def forward_async(self, **kwargs) -> BaseModelOutputWithPast:
        device = next(self.parameters()).device
        return await run_inference_async(self.forward, device=device, **kwargs)


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def set_decoder(self, decoder): self.model = decoder
    def get_decoder(self): return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, cache_position=cache_position, **kwargs
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    async def forward_async(self, **kwargs) -> CausalLMOutputWithPast:
        device = next(self.parameters()).device
        return await run_inference_async(self.forward, device=device, **kwargs)

    async def generate_async(self, **kwargs) -> torch.LongTensor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel): ...
class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"
class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel): ...


__all__ = [
    "Qwen3ForCausalLM", "Qwen3Model", "Qwen3PreTrainedModel",
    "Qwen3ForSequenceClassification", "Qwen3ForQuestionAnswering", "Qwen3ForTokenClassification",
    "Qwen3MLP", "SparseQwen3MLP", "optimize", "run_inference_async",
]

# Preserve the modeling code and its relative imports on save_pretrained.
Qwen3ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
