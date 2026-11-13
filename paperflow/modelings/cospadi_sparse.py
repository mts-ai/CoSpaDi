"""Lossless BF16 sparse storage: dense dictionary, nonzeros and one-bit mask.

Portable execution materializes one coefficient matrix at a time. No dense
coefficient cache or fused projection is retained in the model. The original
MACKO integration can access ``weight`` to initialize its own sparse kernels.
"""

import torch
from torch import nn
from torch.nn import functional as F


def pack_weight(weight):
    import numpy as np

    weight = weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    flat = weight.flatten()
    selected = flat != 0
    mask = torch.from_numpy(np.packbits(selected.numpy(), bitorder="little").copy())
    return mask, flat[selected].contiguous()


class PackedSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, nnz, bias=False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer(
            "packed_mask", torch.zeros((in_features * out_features + 7) // 8, dtype=torch.uint8)
        )
        self.register_buffer("values", torch.zeros(int(nnz), dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def weight(self):
        shifts = torch.arange(8, dtype=torch.uint8, device=self.packed_mask.device)
        mask = ((self.packed_mask[:, None] >> shifts) & 1).reshape(-1)
        mask = mask[: self.in_features * self.out_features].bool()
        dense = self.values.new_zeros(self.in_features * self.out_features)
        return dense.masked_scatter_(mask, self.values).view(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, nnz={self.values.numel()}, packed_mask_bytes={self.packed_mask.numel()}"
