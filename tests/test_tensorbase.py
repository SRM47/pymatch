import unittest
import torch
from .base import BaseUnitTest
import itertools
from match import prod
from match.tensorbase import TensorBase
import operator
from typing import Callable
from random import gauss


class TestTensorBase(BaseUnitTest):
    def to_tensor(self, match_tensorbase) -> torch.Tensor:
        """
        Overrides BaseUnitTest.to_tensor
        """
        torch_tensor = None
        if match_tensorbase.ndim == 0:
            torch_tensor = torch.tensor(match_tensorbase.item()).float()
        else:
            torch_tensor = (
                torch.Tensor(
                    match_tensorbase._raw_data
                )  # Gets the raw 1D array storing the data of the TensorBase object.
                .float()
                .reshape(tuple(match_tensorbase.size))
            )
        torch_tensor.requires_grad = False
        return torch_tensor

    def generate_tensor_pair(self, shape: tuple[int] = None, fill_value=0):
        """Generates a random TensorBase and its PyTorch equivalent.

        Args:
            shape (tuple[int], optional): The shape of the random tensor pair. Defaults to None. If None, a singleton is produced

        Returns:
            match.TensorBase, torch.Tensor: The random tensor tuple
        """
        mat = TensorBase(shape)
        mat.fill_(fill_value)
        ten = self.to_tensor(mat)
        return mat, ten

    def almost_equal(self, match_tensorbase, torch_tensor, equal_nan=False) -> bool:
        print(match_tensorbase)
        print(torch_tensor)
        self.assertTrue(
            torch.allclose(
                self.to_tensor(match_tensorbase),
                torch_tensor.float(),
                rtol=1e-02,
                atol=1e-05,
                equal_nan=equal_nan,
            )
        )
        self.assertEqual(match_tensorbase.size, torch_tensor.size())
        self.assertEqual(match_tensorbase.ndim, torch_tensor.ndim)
        self.assertEqual(match_tensorbase.numel, torch_tensor.numel())
        # Strides differ because PyTorch uses views, while PyMatch doesn't, so we skip this comparison.

    def test_invalid_non_tuple(self):
        """Test TensorBase initialization with non-integer input."""
        with self.assertRaises(ValueError) as context:
            TensorBase(2, 3, 4)
        self.assertIn("Expected tuple.", str(context.exception))

    def test_invalid_non_integer(self):
        """Test TensorBase initialization with non-integer input."""
        with self.assertRaises(ValueError) as _:
            TensorBase((2, "3", 4))

    def test_invalid_list_instead_of_tuple(self):
        """Test TensorBase initialization with a list instead of a tuple."""
        with self.assertRaises(ValueError) as _:
            TensorBase([2, 3, 4])

    def test_invalid_keyword_arguments(self):
        """Test TensorBase initialization with keyword arguments."""
        with self.assertRaises(TypeError) as _:
            TensorBase((2, 3), kw=1)

    def test_tensorbase_creation(self):
        with self.subTest(msg="valid_nd"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                shape=(3, 4, 5, 6), fill_value=3
            )
            self.almost_equal(match_tensorbase, torch_tensor)

        with self.subTest(msg="valid_1d"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                shape=(3,), fill_value=1.5
            )
            self.almost_equal(match_tensorbase, torch_tensor)

        with self.subTest(msg="nd_no_elem"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                shape=(3, 3, 0), fill_value=0
            )
            self.almost_equal(match_tensorbase, torch_tensor)

        with self.subTest(msg="0d_singleton"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                shape=(), fill_value=7
            )
            self.almost_equal(match_tensorbase, torch_tensor)

        with self.subTest(msg="1d_zero_shape"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                shape=(0,), fill_value=0
            )
            self.almost_equal(match_tensorbase, torch_tensor)

    def test_tensorbase_item(self):
        with self.subTest(msg="singleton"):
            match_tensorbase = TensorBase(())
            match_tensorbase.fill_(5)
            self.assertEqual(match_tensorbase.item(), 5)

        with self.subTest(msg="nd_single_element"):
            match_tensorbase = TensorBase((1, 1, 1, 1, 1))
            match_tensorbase.fill_(47)
            self.assertEqual(match_tensorbase.item(), 47)

        with self.subTest(msg="invalid_2_elements"):
            match_tensorbase = TensorBase((2, 1))
            match_tensorbase.fill_(0)
            self.assertRaises(RuntimeError, lambda: match_tensorbase.item())

        with self.subTest(msg="invalid_0_elements"):
            match_tensorbase = TensorBase((2, 0))
            self.assertRaises(RuntimeError, lambda: match_tensorbase.item())

    def test_getitem_partial_index(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair(
                (3, 3, 3), fill_value=2
            )
            self.almost_equal(match_tensor[1:], torch_tensor[1:])

        with self.subTest(msg="extreme"):
            match_tensor, torch_tensor = self.generate_tensor_pair((5, 4), fill_value=2)
            self.almost_equal(match_tensor[4:], torch_tensor[4:])

    def test_getitem_slice(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((2, 4), fill_value=2)
            print(match_tensor, torch_tensor)
            self.almost_equal(match_tensor[:, 1], torch_tensor[:, 1])
            self.almost_equal(match_tensor[:, 1::2], torch_tensor[:, 1::2])

        with self.subTest(msg="slice_zero_failure"):
            match_tensor, _ = self.generate_tensor_pair((2, 4))
            self.assertRaises(ValueError, lambda: match_tensor[:, 1::0])

    def test_getitem_reference(self):
        self.assertTrue(True)

    def test_getitem_single(self):
        match_tensorbase, _ = self.generate_tensor_pair((2, 3, 4, 5), fill_value=0)
        with self.subTest(msg="normal"):
            self.assertEqual(match_tensorbase[0, 1, 2, 3].item(), 0)

        with self.subTest(msg="extreme"):
            self.assertEqual(match_tensorbase[1, 2, 3, 4].item(), 0)
            self.assertEqual(match_tensorbase[0, 0, 0, 0].item(), 0)

        with self.subTest(msg="oob_failure"):
            self.assertRaises(IndexError, lambda: match_tensorbase[2, 0, 0, 0])

    def test_setitem_single_value_index(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 3, 3), fill_value=2)
        torch_tensor[2:] = 0
        match_tensor[2:] = 0
        self.almost_equal(match_tensor, torch_tensor)

    def test_setitem_partial_index(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair(
                (3, 3, 3), fill_value=3
            )
            torch_tensor[2:] = torch.zeros((1, 3, 3))
            match_tensor[2:] = TensorBase((1, 3, 3))
            self.almost_equal(match_tensor, torch_tensor)

        with self.subTest(msg="shape_mismatch_failure"):
            match_tensor, _ = self.generate_tensor_pair((3, 3, 3), fill_value=2)

            def setitem_helper():
                match_tensor[2:] = TensorBase((3, 3, 1))

            self.assertRaises(RuntimeError, setitem_helper)

    def test_setitem_slice(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((2, 4), fill_value=3)
            torch_tensor[:, 1::2] = torch.zeros((2, 2))
            match_tensor[:, 1::2] = TensorBase((2, 2))
            self.almost_equal(match_tensor, torch_tensor)

        with self.subTest(msg="shape_mismatch_failure"):
            match_tensor, _ = self.generate_tensor_pair((2, 4), fill_value=2)

            def setitem_helper():
                match_tensor[:, 1::2] = TensorBase((2, 3))

            self.assertRaises(RuntimeError, setitem_helper)

    def test_setitem_single_number(self):
        with self.subTest(msg="normal"):
            tensor = TensorBase((2, 3, 4))
            tensor.fill_(0)
            tensor[0, 0, 3] = 47.0
            self.assertEqual(
                tensor._raw_data,
                [
                    0,
                    0,
                    0,
                    47,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            )

        with self.subTest(msg="extreme"):
            tensor = TensorBase((2, 3, 4))
            tensor.fill_(0)
            tensor[0, 0, 0] = 47.0
            tensor[1, 2, 3] = 47.0
            self.assertEqual(
                tensor._raw_data,
                [
                    47,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    47,
                ],
            )

        with self.subTest(msg="oob_failure"):
            tensor = TensorBase((2, 3, 4, 5))

            def setitem_helper():
                tensor[1, 2, 3, 5] = 47.0

            self.assertRaises(IndexError, setitem_helper)

    def test_setitem_single_tensorbase(self):
        tensor = TensorBase((2, 3, 4))
        tensor.fill_(0)
        tensor_update = TensorBase(())
        tensor_update.fill_(47)
        tensor[0, 0, 0] = tensor_update
        self.assertEqual(
            tensor._raw_data,
            [47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_sum(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3), fill_value=3)
        with self.subTest(msg="dim"):
            self.almost_equal(match_tensor.sum((0,), False), torch_tensor.sum((0,)))
            self.almost_equal(match_tensor.sum((0, 1), False), torch_tensor.sum((0, 1)))
        with self.subTest(msg="keepdim"):
            self.almost_equal(
                match_tensor.sum((1, 2), True),
                torch_tensor.sum((1, 2), keepdim=True),
            )
        with self.subTest(msg="nodim"):
            self.almost_equal(match_tensor.sum((), False), torch_tensor.sum())

    def test_mean(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3), fill_value=4)
        with self.subTest(msg="dim"):
            self.almost_equal(match_tensor.mean((0,), False), torch_tensor.mean((0,)))
            self.almost_equal(
                match_tensor.mean((0, 1), False), torch_tensor.mean((0, 1))
            )
        with self.subTest(msg="keepdim"):
            self.almost_equal(
                match_tensor.mean((1, 2), True),
                torch_tensor.mean((1, 2), keepdim=True),
            )
        with self.subTest(msg="nodim"):
            self.almost_equal(match_tensor.mean((), False), torch_tensor.mean())

    # TODO: Implement configurations that are intended to fail.
    def test_matmul_various_shapes_failure(self):
        self.assertTrue(True)

    def test_matmul_various_shapes(self):
        configurations = {
            "1d@1d": [(9,), (9,)],
            "1d@2d": [(9,), (9,)],
            "2d@1d": [(8,), (8, 2)],
            "2d@2d": [(7, 8), (8, 2)],
            "1d@nd": [(5,), (2, 5, 3)],
            "nd@1d": [(3, 2, 5, 8, 5), (5,)],
            "nd@nd": [(2, 1, 7, 4), (2, 4, 3)],
        }
        for msg, shapes in configurations.items():
            with self.subTest(msg=msg):
                match_tensor1, torch_tensor1 = self.generate_tensor_pair(
                    shapes[0], fill_value=4.7
                )
                match_tensor2, torch_tensor2 = self.generate_tensor_pair(
                    shapes[1], fill_value=3
                )
                self.almost_equal(
                    match_tensor1 @ match_tensor2, torch_tensor1 @ torch_tensor2
                )

    def test_transpose(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3), fill_value=2)
        self.assertTrue(self.almost_equal(match_tensor.T, torch_tensor.T))

    def test_permute(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3), fill_value=5)
        self.almost_equal(
            match_tensor.permute((2, 0, 1)), torch_tensor.permute(2, 0, 1)
        )
        self.almost_equal(
            match_tensor.permute((0, 1, 2)), torch_tensor.permute(0, 1, 2)
        )
        self.assertRaises(RuntimeError, lambda: match_tensor.permute((2,)))
        self.assertRaises(RuntimeError, lambda: match_tensor.permute((-1,)))
        self.assertRaises(RuntimeError, lambda: match_tensor.permute((3, 0, 2, 1)))
        self.assertRaises(RuntimeError, lambda: match_tensor.permute((0, 0, 1)))
        match_tensor, torch_tensor = self.generate_tensor_pair((), fill_value=5)
        self.almost_equal(match_tensor.permute(()), torch_tensor.permute(()))
        self.assertRaises(RuntimeError, lambda: match_tensor.permute((0,)))

    def test_reshape(self):
        with self.subTest(msg="nd_to_nd"):
            match_tensor, torch_tensor = self.generate_tensor_pair(
                (2, 3, 4), fill_value=0
            )
            with self.subTest(msg="success"):
                self.almost_equal(
                    match_tensor.reshape((4, 3, 2)), torch_tensor.reshape((4, 3, 2))
                )
            with self.subTest(msg="failure"):
                self.assertRaises(RuntimeError, lambda: match_tensor.reshape((5, 5, 5)))

        with self.subTest(msg="1d_to_singleton"):
            with self.subTest(msg="success"):
                match_tensor, torch_tensor = self.generate_tensor_pair(
                    (1,), fill_value=5
                )
                self.almost_equal(match_tensor.reshape(()), torch_tensor.reshape(()))
            with self.subTest(msg="failure"):
                # The tensor is one dimensional, but there are two items in the tensor, so reshape should fail.
                match_tensor, _ = self.generate_tensor_pair((2,), fill_value=5)
                self.assertRaises(RuntimeError, lambda: match_tensor.reshape(()))

        with self.subTest(msg="singleton_to_1d"):
            match_tensor, torch_tensor = self.generate_tensor_pair((), fill_value=5)
            with self.subTest(msg="success"):
                self.almost_equal(
                    match_tensor.reshape((1,)), torch_tensor.reshape((1,))
                )
            with self.subTest(msg="failure"):
                self.assertRaises(RuntimeError, lambda: match_tensor.reshape((2,)))

        with self.subTest(msg="nd_to_singleton"):
            with self.subTest(msg="success"):
                match_tensor, torch_tensor = self.generate_tensor_pair(
                    (1, 1, 1, 1, 1), fill_value=5
                )
                print(match_tensor.reshape(()), torch_tensor.reshape(()))
                self.almost_equal(match_tensor.reshape(()), torch_tensor.reshape(()))

            with self.subTest(msg="failure"):
                match_tensor, _ = self.generate_tensor_pair(
                    (1, 2, 1, 1, 1), fill_value=5
                )

        with self.subTest(msg="singleton_to_nd"):
            match_tensor, torch_tensor = self.generate_tensor_pair((), fill_value=5)
            with self.subTest(msg="success"):
                self.almost_equal(
                    match_tensor.reshape((1, 1, 1, 1, 1)),
                    torch_tensor.reshape((1, 1, 1, 1, 1)),
                )
            with self.subTest(msg="failure"):
                self.assertRaises(
                    RuntimeError, lambda: match_tensor.reshape((1, 1, 2, 1, 1))
                )

    # TODO: Implement broadcast_to
    # def test_broadcast_singleton(self):
    #     match_tensor, torch_tensor = self.generate_tensor_pair((), fill_value=3)

    #     torch_tensor_broadcasted = torch_tensor.broadcast_to((3, 3))
    #     match_tensor_broadcasted = match_tensor.broadcast_to((3, 3))

    #     self.almost_equal(match_tensor_broadcasted, torch_tensor_broadcasted)

    # # TODO: Implement broadcast_to()
    # def test_broadcast(self):
    #     with self.subTest(msg="normal"):
    #         match_tensor, torch_tensor = self.generate_tensor_pair(
    #             (3, 1, 3), fill_value=0.2
    #         )

    #         torch_tensor_broadcasted = torch_tensor.broadcast_to((2, 2, 3, 3, 3))
    #         match_tensor_broadcasted = match_tensor.broadcast_to((2, 2, 3, 3, 3))

    #         self.almost_equal(match_tensor_broadcasted, torch_tensor_broadcasted)

    #     with self.subTest(msg="failure"):
    #         match_tensor = TensorBase((3, 1, 3))
    #         self.assertRaises(ValueError, lambda: match_tensor.broadcast(2, 2, 1, 3, 3))
    #         self.assertRaises(RuntimeError, lambda: match_tensor.broadcast(3, 3))

    def test_abs(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.3
        )
        self.almost_equal(match_tensorbase.abs(), torch_tensor.abs())

    def test_trig(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.4
        )
        with self.subTest(msg="cos"):
            self.almost_equal(match_tensorbase.cos(), torch_tensor.cos())

        with self.subTest(msg="sin"):
            self.almost_equal(match_tensorbase.sin(), torch_tensor.sin())

        with self.subTest(msg="tan"):
            self.almost_equal(match_tensorbase.tan(), torch_tensor.tan())

    def test_log(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=1.6
        )
        with self.subTest(msg="ln"):
            self.almost_equal(match_tensorbase.log(), torch_tensor.log())

        with self.subTest(msg="log_negative_number"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                (2, 3, 4), fill_value=-0.5
            )
            self.almost_equal(
                match_tensorbase.log(), torch_tensor.log(), equal_nan=True
            )

    def test_exp(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.7
        )
        self.almost_equal(match_tensorbase.exp(), torch_tensor.exp())

    def test_pow(self):
        with self.subTest(msg="positive_base"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                (2, 3, 4), fill_value=1
            )
            with self.subTest(msg="positive_exponent"):
                self.almost_equal(match_tensorbase**2, torch_tensor**2)

            with self.subTest(msg="negative_exponent"):
                self.almost_equal(match_tensorbase**-1, torch_tensor**-1)

            with self.subTest(msg="decimal_exponent"):
                self.almost_equal(match_tensorbase**0.5, torch_tensor**0.5)
            
            with self.subTest(msg="negative_decimal_exponent"):
                self.almost_equal(match_tensorbase**-0.5, torch_tensor**-0.5)

        with self.subTest(msg="negative_base"):
            match_tensorbase, torch_tensor = self.generate_tensor_pair(
                (2, 3, 4), fill_value=-5
            )
            with self.subTest(msg="positive_exponent"):
                self.almost_equal(match_tensorbase**2, torch_tensor**2)

            with self.subTest(msg="negative_exponent"):
                self.almost_equal(match_tensorbase**-1, torch_tensor**-1)
            
            with self.subTest(msg="decimal_exponent"):
                self.almost_equal(match_tensorbase**0.5, torch_tensor**0.5, equal_nan = True)

            with self.subTest(msg="negative_decimal_exponent"):
                self.almost_equal(match_tensorbase**-0.5, torch_tensor**-0.5, equal_nan = True)

    def test_fill(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.3
        )
        match_tensorbase.fill_(5)
        torch_tensor.fill_(5)
        self.almost_equal(match_tensorbase, torch_tensor)

    def test_sigmoid(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.5
        )
        self.almost_equal(match_tensorbase.sigmoid(), torch_tensor.sigmoid())

    def test_tanh(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.7
        )
        self.almost_equal(match_tensorbase.tanh(), torch_tensor.tanh())

    def test_relu(self):
        match_tensorbase, torch_tensor = self.generate_tensor_pair(
            (2, 3, 4), fill_value=0.5
        )
        self.almost_equal(match_tensorbase.relu(), torch_tensor.relu())

    def test_bin_operators_broadcast_success(self):
        operators_to_test = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "truediv": operator.truediv,
            "floordiv": operator.floordiv,
        }
        for msg, op in operators_to_test.items():
            with self.subTest(msg=msg):
                print(msg)
                with self.subTest(msg="nd_nd_same_shape"):
                    match_tensorbase1, torch_tensor1 = self.generate_tensor_pair(
                        (2, 3, 4), fill_value=4.2
                    )
                    match_tensorbase2, torch_tensor2 = self.generate_tensor_pair(
                        (2, 3, 4), fill_value=4.5
                    )
                    self.almost_equal(
                        op(match_tensorbase1, match_tensorbase2),
                        op(torch_tensor1, torch_tensor2),
                    )

                with self.subTest(msg="nd_nd_broadcastable_shape"):
                    match_tensorbase1, torch_tensor1 = self.generate_tensor_pair(
                        (2, 3, 1), fill_value=0.4
                    )
                    match_tensorbase2, torch_tensor2 = self.generate_tensor_pair(
                        (1, 3, 4), fill_value=1.244
                    )
                    self.almost_equal(
                        op(match_tensorbase1, match_tensorbase2),
                        op(torch_tensor1, torch_tensor2),
                    )

                with self.subTest(msg="nd_singleton"):
                    match_tensorbase_singleton1, torch_tensor_singleton1 = (
                        self.generate_tensor_pair((), fill_value=3.1)
                    )
                    match_tensorbase2, torch_tensor2 = self.generate_tensor_pair(
                        (2, 3, 4), fill_value=2.273
                    )
                    self.almost_equal(
                        op(match_tensorbase_singleton1, match_tensorbase2),
                        op(torch_tensor_singleton1, torch_tensor2),
                    )
                    self.almost_equal(
                        op(match_tensorbase2, match_tensorbase_singleton1),
                        op(torch_tensor2, torch_tensor_singleton1),
                    )

                with self.subTest(msg="nd_scalar"):
                    match_tensorbase, torch_tensor = self.generate_tensor_pair(
                        (2, 3, 4), fill_value=4.5
                    )
                    self.almost_equal(
                        op(match_tensorbase, 1.47), op(torch_tensor, 1.47)
                    )
                    self.almost_equal(
                        op(-3.47, match_tensorbase), op(-3.47, torch_tensor)
                    )

                with self.subTest(msg="singleton_singleton"):
                    match_tensorbase_singleton1, torch_tensor_singleton1 = (
                        self.generate_tensor_pair((), fill_value=-102.2)
                    )
                    match_tensorbase_singleton2, torch_tensor_singleton2 = (
                        self.generate_tensor_pair((), fill_value=223478.234)
                    )
                    self.almost_equal(
                        op(match_tensorbase_singleton1, match_tensorbase_singleton2),
                        op(torch_tensor_singleton1, torch_tensor_singleton2),
                    )

                with self.subTest(msg="singleton_scalar"):
                    match_tensorbase_singleton, torch_tensor_singleton = (
                        self.generate_tensor_pair((), fill_value=2.34987593)
                    )
                    self.almost_equal(
                        op(match_tensorbase_singleton, 1.47),
                        op(torch_tensor_singleton, 1.47),
                    )
                    self.almost_equal(
                        op(-3.47, match_tensorbase_singleton),
                        op(-3.47, torch_tensor_singleton),
                    )

    def test_operators_broadcast_failure(self):
        operators_to_test = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "truediv": operator.truediv,
            "floordiv": operator.floordiv,
        }
        for msg, op in operators_to_test.items():
            with self.subTest(msg=msg):
                match_tensorbase1, _ = self.generate_tensor_pair(
                    (2, 3, 4), fill_value=0
                )
                match_tensorbase2, _ = self.generate_tensor_pair(
                    (2, 3, 3), fill_value=0
                )
                self.assertRaises(
                    RuntimeError, lambda: op(match_tensorbase1, match_tensorbase2)
                )
