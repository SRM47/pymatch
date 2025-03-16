from __future__ import annotations
from typing import Union
import numpy as np


class TensorBase(object):
    """A storage and arithmetic object for n-dimensional tensor data.

    TensorBase is an inefficient, but easy-to-understand implementation
    of many n-dimensional tensor operations.

    Like the PyTorch Tensor object, the pymatch TensorBase objects are
    recursive structures that hold either a list of TensorBase objects,
    or a single value. For instance, a TensorBase object's data list could
    look like [TensorBase(0), TensorBase(1), raise NotImplementedError, TensorBase(47)].
    There are therefore two implicit types of TensorBase objects. Ones
    that store only a single value, accesible by .item(), and ones that
    store a list of these `singleton` TensorBase objects.

    Using the provided shape of the TensorBase object, accessing data using
    the standard coordinate system, for instance x[1,3,2], involves translation
    of the coordinates (1,3,2) to the corresponding index in the data list
    to fetch the intended data.
    """

    def __init__(
        self,
        *size: int,
        value: float = 0.0,
        numpy_data: np.ndarray = None,
    ) -> None:
        """Create a new TensorBase object to store an n-dimensional tensor of values.

        Args:
            size (tuple[int]): The shape of the Tensor
            value (float): The default value of each element in the Tensor
            dtype (type): The type of the values in the Tensor
        """
        super().__init__()
        assert all(
            isinstance(dim, int) for dim in size
        ), f"Size {size} must be a variadict of only integers"

        if numpy_data is None:
            # If an ndarray is not already provided, instantiate a new one with the specified fill value.
            self._numpy_data: np.ndarray = np.full(size, value)
        else:
            # If an ndarray is provided, use it.
            self._numpy_data = numpy_data

    @property
    def shape(self) -> tuple[int]:
        """Returns the shape of the underlying ndarray"""
        return self._numpy_data.shape

    def item(self) -> int | float:
        """Returns the item of a singleton, or single element TensorBase object.

        Raises:
            ValueError: raises value error if attempting to access the item of a
            TensorBase object with more than one element.
        """
        return self._numpy_data.item(0)

    def reshape_(self, shape: tuple[int]):
        """Helper method to reshape the TensorBase object inplace, without changing the data.

        Raises:
            RuntimeError: raises runtime error if the product of the dimensions of
            the new shape is not equal to the existing number of elements.
        """
        self._numpy_data = self._numpy_data.reshape(shape)

    def reshape(self, shape: tuple[int]) -> TensorBase:
        """Helper method to reshape and return a new TensorBase object without changing the data.

        Args:
            shape: The size to be reshaped to.

        Returns:
            A new TensorBase object with the same data.
        """
        return TensorBase(
            numpy_data=self._numpy_data.reshape(shape),
        )

    def __getitem__(self, coords) -> TensorBase:
        """Retrieves a subtensor from the current tensor using the given coordinates.

        Args:
            coords: A tuple of slices or integers.

        Returns:
            A new tensor containing the subtensor specified by the given coordinates.
        """
        if not isinstance(coords, tuple):
            coords = (coords,)
        return TensorBase(numpy_data=self._numpy_data[*coords])

    def __setitem__(self, coords, value) -> None:
        """Sets the value of a subtensor at the given coordinates.

        Args:
            coords: A tuple of slices or integers.
            value: A new value to assign to the subtensor.

        Raises:
            TypeError: If the value is a list or an invalid type.
        """
        if isinstance(value, list):
            raise TypeError("Unable to assign a list to a TensorBase")

        if not isinstance(coords, tuple):
            coords = (coords,)

        if isinstance(value, TensorBase):
            raise TypeError("Incompatible TensorBase options. Numpy and TensorBase")
        self._numpy_data[*coords] = value

    def __repr__(self):
        return self._numpy_data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def sum(self, dims: tuple | int = None, keepdims: bool = False) -> TensorBase:
        """Compute the sum across certain dimensions in a TensorBase object
        Args:
            dims (tuple | int, optional): The dimensions to sum over. Defaults to None.
            keepdims (bool, optional): Whether to keep the dimensions being aggregated. Defaults to False.

        Returns:
            TensorBase: The resulting TensorBase object.
        """
        if isinstance(dims, int):
            dims = (dims,)

        return TensorBase(numpy_data=self._numpy_data.sum(axis=dims, keepdims=keepdims))

    def mean(self, dims: tuple | int = None, keepdims: bool = False) -> TensorBase:
        """Compute the mean across certain dimensions in a TensorBase object
        Args:
            dims (tuple | int, optional): The dimensions to average over. Defaults to None.
            keepdims (bool, optional): Whether to keep the dimensions being averaged. Defaults to False.

        Returns:
            TensorBase: The resulting TensorBase object.
        """
        if isinstance(dims, int):
            dims = (dims,)

        return TensorBase(
            numpy_data=self._numpy_data.mean(axis=dims, keepdims=keepdims)
        )

    def unbroadcast(self, *shape: int) -> TensorBase:
        """Return a new TensorBase with the broadcasted dimensions adjusted to match the desired shape.

        This method is used to unbroadcast the dimensions of a tensor to match a desired shape.
        It is particularly useful for operations that involve broadcasting, such as element-wise
        arithmetic operations or matrix multiplication.

        Args:
            *shape (int): The desired shape of the tensor after unbroadcasting.

        Returns:
            TensorBase: A new TensorBase object with adjusted dimensions.

        Reference:
            For more information on unbroadcasting, refer to:
            https://mostafa-samir.github.io/auto-diff-pt2/#unbroadcasting-adjoints
        """
        correct_adjoint = self

        # If the current shape doesn't match the desired shape.
        if self.shape != shape:
            dim_diff = abs(len(self.shape) - len(shape))

            # If there is a difference in dimensions, sum over excess dimensions and discard the dimensions summed over.
            if dim_diff:
                summation_dims = tuple(range(dim_diff))
                # Sum over excess dimensions and discard them (keepdims=False by default).
                correct_adjoint = self.sum(dims=summation_dims)

            # Find the dimensions that were originally ones
            originally_ones = tuple(
                [axis for axis, size in enumerate(shape) if size == 1]
            )

            # If there were originally dimensions with size 1, sum over them with keepdims=True
            if len(originally_ones) != 0:
                correct_adjoint = correct_adjoint.sum(
                    dims=originally_ones, keepdims=True
                )

        return correct_adjoint

    def ones_(self) -> None:
        """Modify all values in the tensor to be 1.0."""
        self.__set(1.0)

    def zeros_(self) -> None:
        """Modify all values in the tensor to be 0.0."""
        self.__set(0.0)

    def numel(self) -> int:
        return self._numpy_data.size

    def relu(self) -> Union[TensorBase, np.ndarray]:
        """Return a new TensorBase object with the ReLU of each element."""
        return TensorBase(
            numpy_data=np.maximum(self._numpy_data, 0),
        )

    def sigmoid(self) -> TensorBase:
        """Return a new TensorBase object with the sigmoid of each element."""
        return TensorBase(
            numpy_data=1 / (1 + np.exp(-self._numpy_data)),
        )

    def permute(self, *dims: int) -> TensorBase:
        """Return an aliased TensorBase object with a permutation of its original dimensions permuted"""
        return TensorBase(
            numpy_data=np.transpose(self._numpy_data, tuple(dims)),
        )

    @property
    def T(self) -> TensorBase:
        """Return an aliased TensorBase object with the transpose of the tensor."""
        return TensorBase(
            numpy_data=np.transpose(self._numpy_data),
        )

    def __set(self, val) -> None:
        """Internal method to set all values in the TensorBase to val."""
        self._numpy_data = np.full(self._numpy_data.shape, val)

    def __add__(self, rhs: Union[float, int, TensorBase]) -> TensorBase:
        """Element-wise addition: self + rhs."""
        if isinstance(rhs, TensorBase):
            return TensorBase(
                numpy_data=self._numpy_data + rhs._numpy_data,
            )
        else:
            return TensorBase(
                numpy_data=self._numpy_data + rhs,
            )

    def __radd__(self, lhs: Union[float, int, TensorBase]) -> TensorBase:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: Union[float, int, TensorBase]) -> TensorBase:
        """Element-wise subtraction: self - rhs."""
        return -rhs + self

    def __rsub__(self, lhs: Union[float, int, TensorBase]) -> TensorBase:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __mul__(self, rhs: Union[float, int, TensorBase]) -> TensorBase:
        """Element-wise multiplication: self * rhs."""
        if isinstance(rhs, TensorBase):
            return TensorBase(
                numpy_data=self._numpy_data * rhs._numpy_data,
            )
        else:
            return TensorBase(
                numpy_data=self._numpy_data * rhs,
            )

    def __rmul__(self, lhs: float | int | TensorBase) -> TensorBase:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __pow__(self, rhs: float | int) -> TensorBase:
        """Element-wise exponentiation: self ** rhs."""
        return TensorBase(
            numpy_data=self._numpy_data**rhs,
        )

    def __truediv__(self, rhs: float | int | TensorBase) -> TensorBase:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: float | int | TensorBase) -> TensorBase:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __neg__(self) -> TensorBase:
        """Element-wise unary negation: -self."""
        return self * -1

    def __gt__(self, rhs: float | int | TensorBase) -> TensorBase:
        """Element-wise comparison: self > rhs."""
        if isinstance(rhs, TensorBase):
            return TensorBase(
                numpy_data=self._numpy_data > rhs._numpy_data,
            )
        else:
            return TensorBase(
                numpy_data=self._numpy_data > rhs,
            )

    def exp(self) -> TensorBase:
        return TensorBase(
            numpy_data=np.exp(self._numpy_data),
        )

    def __matmul__(self, rhs: TensorBase) -> TensorBase:
        return TensorBase(
            numpy_data=self._numpy_data @ rhs._numpy_data,
        )

    @staticmethod
    def concatenate(TensorBases: list[TensorBase], dim: int = 0) -> TensorBase:
        """Concatenates a sequence of tensors along a given dimension.

        Args:
            tensors: A sequence of tensors (e.g., NumPy arrays) to concatenate.
            dim: The dimension along which to concatenate. Default is 0.

        Returns:
            The concatenated tensor.

        Raises:
            ValueError: If tensors have incompatible shapes or dim is invalid.
        """
        if not TensorBases:  # Handle empty input
            raise ValueError("Input tensors cannot be empty")

        # Check shape compatibility (all dimensions except the concatenation dim must match)
        for i in range(1, len(TensorBases)):
            if TensorBases[i].shape[:dim] + TensorBases[i].shape[dim + 1:] != TensorBases[0].shape[:dim] + TensorBases[0].shape[dim + 1:]:
                raise ValueError("match.cat(): tensors must have the same shape, except along the concatenation dimension")

        # Concatenate along the specified dimension.
        TensorBase_numpy_arrays = [td._numpy_data for td in TensorBases]
        return TensorBase(numpy_data=np.concatenate(TensorBase_numpy_arrays, axis=dim))
