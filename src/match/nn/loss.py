from __future__ import annotations

from match import Tensor
from .module import Module


class MSELoss(Module):
    """loss = (1/N) * Σ (yhati - yi)^2"""
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Returns a new Tensor
        return ((target - prediction) ** 2).mean()


class MultiClassCrossEntropyLoss(Module):
    """
    loss = - (1/N) * Σ Σ yi * log(yhati)
    """

    def forward(self, prediction: Tensor, target: Tensor) -> float:

        return -(target * prediction.log()).mean()