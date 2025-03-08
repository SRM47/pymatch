"""Pytorch Remake
"""

from math import prod
from random import gauss
from .tensor import Tensor, use_numpy
import numpy as np

if use_numpy:
    from .tensordata_numpy import TensorData
else:
    from .tensordata import TensorData

from match.tensorbase import TensorBase


def cat(tensors: list[Tensor], dim=0) -> Tensor:
    """_summary_

    Args:
        tensors (list[Tensor]): _description_
        dim (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: _description_
    """
    # Store the underlying TensorData objects.
    tensordata_objects = [tensor.data for tensor in tensors]
    # Concatenate the TensorData objects into a single object.
    concatenated_tensordata_objects = TensorData.concatenate(
        tensordatas=tensordata_objects, dim=dim
    )
    return Tensor(data=concatenated_tensordata_objects)


def randn(*shape, generator=lambda: gauss(0, 1)) -> Tensor:
    if shape != () and isinstance(shape[0], tuple):
        shape = shape[0]

    t = TensorBase(shape)
    t.randn_(0, 1)
    return Tensor(data=t)
