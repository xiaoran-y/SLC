from __future__ import annotations

import copy

import torch
from torch import nn


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(int(n))])


def pos_encode(seq_len: int) -> torch.Tensor:
    return torch.arange(seq_len, dtype=torch.long)


def ut_mask(*, seq_len: int) -> torch.Tensor:
    # Upper-triangular causal mask for attention (True/1 means masked).
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)


def transformer_FFN(d_model: int, dropout: float) -> nn.Module:
    d_model = int(d_model)
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(d_model, d_model),
    )

