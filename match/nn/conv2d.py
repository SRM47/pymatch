from __future__ import annotations
from typing import Optional

import numpy as np
import match

from math import prod
from match import Tensor, TensorData, use_numpy
from .module import Module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple | int,
        stride: tuple | int = 1,
        padding: tuple | int = 0,
        dilation: tuple | int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        if self.in_channels < 0:
            raise RuntimeError("in_channels must be non negative")

        self.out_channels: int = out_channels
        if self.out_channels < 0:
            raise RuntimeError("out_channels must be non negative")

        self.__initialize_kernels(kernel_size)

        self.stride: tuple | int = self.__initialize_position_variable(stride)
        if any(s <= 0 for s in self.stride):
            raise RuntimeError(f"stride must be greater than 0, but got {self.stride}")

        self.padding: tuple | int = self.__initialize_position_variable(padding)
        if any(p < 0 for p in self.padding):
            raise RuntimeError(f"padding must be non negative, but got {self.padding}")

        self.dilation: tuple | int = self.__initialize_position_variable(dilation)
        if any(d < 1 for d in self.dilation):
            raise RuntimeError(
                f"dilation must be greater than 0, but got {self.dilation}"
            )

        self.groups: int = groups
        self.padding_mode = padding_mode
        self.__initialize_bias(bias)

    def __initialize_position_variable(self, val: tuple | int):
        val = val if isinstance(val, tuple) else (val, val)
        if len(val) != 2:
            raise RuntimeError(
                "stride, padding, dilation should be a tuple of two ints. the first int is used for the height dimension, and the second int for the width dimension."
            )
        return val

    def __initialize_kernels(self, kernel_size: tuple | int) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if len(kernel_size) != 2:
            raise RuntimeError(
                "kernel_size should be a tuple of two ints. the first int is used for the height dimension, and the second int for the width dimension."
            )

        if any(kernel_dim <= 0 for kernel_dim in kernel_size):
            raise RuntimeError(
                f"kernel size should be greater than zero, but got shape {kernel_size}"
            )

        self._single_kernel_shape = (self.in_channels,) + kernel_size

        # Store all kernels as a 2D matrix for efficient memory access instead of
        # another data structure that stores each 3D kernel individually.
        # Each column represents a single kernel, resulting in out_channels columns.
        # The number of rows corresponds to the total number of elements in each kernel.
        self._trainable_kernels: Tensor = Tensor.randn(
            prod(self._single_kernel_shape), self.out_channels
        )

    def __initialize_bias(self, bias: bool) -> None:
        self.bias: bool = bias
        if bias:
            self._trainable_bias = match.randn(self.out_channels)

    def get_expected_output_dimensions(self, x: Tensor) -> tuple[int]:
        """Calculates rhe expected dimensions of the output tensor after the convolution.

        Args:
            x (Tensor): The input tensor.

        Returns:
            tuple[int]: The expected output shape of the tensor.
        """
        N = None
        if len(x.shape) == 4:
            N, _, height_in, width_in = x.shape
        elif len(x.shape) == 3:
            _, height_in, width_in = x.shape
        else:
            raise ValueError(
                "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
            )

        height_out = int(
            (
                height_in
                + 2 * self.padding[0]
                - self.dilation[0] * (self._single_kernel_shape[1] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )

        width_out = int(
            (
                width_in
                + 2 * self.padding[1]
                - self.dilation[1] * (self._single_kernel_shape[2] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )

        if N:
            return N, self.out_channels, height_out, width_out
        else:
            return self.out_channels, height_out, width_out

    def __get_kernel_position_slices(
        self,
        height_in: int,
        width_in: int,
        N: int = None,
    ) -> tuple[slice]:
        """Calculates the slice positions for applying the kernels to the input tensor.
        The returned tuple is a sequence of slices that tracks how a kernel glides across the input tensor.

        Args:
            height_in (int): The height of each instance of the input tensor
            width_in (int): The width of each instance of the input tensor
            N (int, optional): Batch size, if batch provided. Defaults to None.

        Returns:
            tuple[slice]: An ordered tuple of many slice objects defining the positions a kernel will be applied within the input tensor.
        """

        # Unpack kernel dimensions and convolution parameters into individual parameters.
        kernel_channels, kernel_height, kernel_width = self._single_kernel_shape
        stride_height, stride_width = self.stride
        dilation_height, dilation_width = self.dilation
        padding_height, padding_width = self.padding

        # Calculate effective kernel size with dilation.
        dilated_kernel_height = (kernel_height - 1) * dilation_height + 1
        dilated_kernel_width = (kernel_width - 1) * dilation_width + 1

        # Calculate starting and ending positions for kernel placement.
        starting_height = -padding_height
        starting_width = -padding_width
        ending_height = height_in + padding_height - dilated_kernel_height + 1
        ending_width = width_in + padding_width - dilated_kernel_width + 1

        # Build kernel position slices with padding and dilation.
        instance_kernel_positions = []
        for h in range(starting_height, ending_height, stride_height):
            for w in range(starting_width, ending_width, stride_width):
                instance_kernel_positions.append(
                    (
                        slice(0, kernel_channels),  # Channel slice
                        slice(
                            h, h + dilated_kernel_height, dilation_height
                        ),  # Height slice with dilation
                        slice(
                            w, w + dilated_kernel_width, dilation_width
                        ),  # Width slice with dilation
                    )
                )

        if N:
            # If N is not None, the tensor is 4D.
            instance_kernel_positions = [
                (n,) + position
                for n in range(N)
                for position in instance_kernel_positions
            ]

        # Calculate actual output dimensions for verification.
        height_out = len(range(starting_height, ending_height, stride_height))
        width_out = len(range(starting_width, ending_width, stride_width))

        return tuple(instance_kernel_positions), height_out, width_out

    def __prepare_input_for_kernels(self, x: Tensor) -> Tensor:
        """
        Prepares input tensor for efficient application of the convolution kernels:

        1. Calculate _Kernel Positions_, which are the slices within the input tensor where a kernel will be applied.
        2. Extract the subtensors from the input tensor corresponding to the cacluated kernel positions.
        3. Concatenate and reshape the subtensors into a matrix for efficient kernel application. Each row in this tensor is a single kernel position.

        Args:
            x:  The input Tensor.
                Expected shape: (N, in_channels, H, W) for batched input, or (in_channels, H, W) for single input.
                                N:            Batch Size
                                in_channels:  The number of input channels
                                H:            Height of the input tensor
                                W:            Width of the input tensor

        Returns:
            A Tensor ready for convolution kernel application (multiply by self._trainable_kernels)
        """
        # N: Batch Size (if provided with batch input).
        # height_in: The height of each instance in the input tensor.
        # width_in: The width of each instance in the input tensor.
        N = None
        if x.dim() == 4:
            N, _, height_in, width_in = x.shape
        elif x.dim() == 3:
            _, height_in, width_in = x.shape
        else:
            raise ValueError(
                "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
            )

        # Calculate the list of positions (slice objects) of a kernel over the input tensor.
        kernel_positions, _, _ = self.__get_kernel_position_slices(
            height_in, width_in, N
        )

        single_kernel_size = prod(self._single_kernel_shape)

        # Prepare input data for kernel application.
        # This step extracts the smaller sections from the input tensor, corresponding to the kernel positions calculated above.
        # Each section is flattened into a 1D array, containing the values the kernel will multiply at that position.
        flattened_input_features = []
        for kernel_position in kernel_positions:
            # Grab subtensor corresponding to the current kernel position.
            subtensor = x.data[kernel_position]
            # Flatten subtensor into single dimension.
            flattened_subtensor = subtensor.reshape((single_kernel_size,))
            # Add flattened subtensor into array.
            flattened_input_features.append(flattened_subtensor)

        # Concatenate all of the subtensors into a single matrix. Each row is a single kernel position.
        flattened_input_features = TensorData.concatenate(
            tensordatas=flattened_input_features, dim=0
        )
        if len(x.shape) == 4:
            flattened_input_features.reshape_(
                (
                    N,
                    len(kernel_positions) // N,
                    prod(self._single_kernel_shape),
                )  # Divide by N because kernel positions includes those for all N instances in the batch.
            )
        else:
            flattened_input_features.reshape_(
                (
                    len(kernel_positions),
                    prod(self._single_kernel_shape),
                )  # Only single instance so N=1.
            )

        return Tensor(data=flattened_input_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution over the input tensor.

        Args:
            x:  The input Tensor.
                Expected shape: (N, in_channels, H, W) for batched input, or (in_channels, H, W) for single input.
                                N:            Batch Size
                                in_channels:  The number of input channels
                                H:            Height of the input tensor
                                W:            Width of the input tensor
        """
        # Calculate the expected dimensions of the tensor after applying the convolution operator.
        expected_output_dimensions = self.get_expected_output_dimensions(x)
        height_out, width_out = expected_output_dimensions[-2:]

        # Prepare the input tensor for efficient kernel application.
        conv_input_matrix = self.__prepare_input_for_kernels(x)

        # Apply the kernels (perform the convolution operation) on the prepared input.
        convolution_tensor = conv_input_matrix @ self._trainable_kernels
        if self.bias:
            convolution_tensor = convolution_tensor + self._trainable_bias

        # Transpose the last two dimensions, then reshape them to match the expected output shape after convolution.
        if len(convolution_tensor.shape) == 3:
            N = x.shape[0]  # Batch Size.
            convolution_tensor = convolution_tensor.permute(0, 2, 1)
            convolution_tensor = convolution_tensor.reshape(
                N, self.out_channels, height_out, width_out
            )
        else:
            convolution_tensor = convolution_tensor.permute(1, 0)
            convolution_tensor = convolution_tensor.reshape(
                self.out_channels, height_out, width_out
            )

        return convolution_tensor
