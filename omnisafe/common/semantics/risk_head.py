"""Risk head predicting discounted future cost based on semantic embedding only.

Simple MLP: input embedding -> hidden -> scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SemanticRiskHead(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, D)
        return self.net(x).squeeze(-1)
