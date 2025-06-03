from .config import BackendOption, backend_option
from random import gauss
from .tensor import Tensor
from match.tensorbase import TensorBase


def cat(tensors: list[Tensor], dim=0) -> Tensor:
    """_summary_

    Args:
        tensors (list[Tensor]): _description_
        dim (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: _description_
    """
    # Store the underlying TensorBase objects.
    TensorBase_objects = [tensor.data for tensor in tensors]
    # Concatenate the TensorBase objects into a single object.
    concatenated_TensorBase_objects = TensorBase.concatenate(
        TensorBases=TensorBase_objects, dim=dim
    )
    return Tensor(data=concatenated_TensorBase_objects)


def randn(*shape, generator=lambda: gauss(0, 1)) -> Tensor:
    if shape != () and isinstance(shape[0], tuple):
        shape = shape[0]

    t = TensorBase(shape)
    t.randn_(0, 1)
    return Tensor(data=t)
