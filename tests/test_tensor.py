import unittest
import torch
from match import tensordata, tensor, randn
import itertools
import numpy as np
import random


def almost_equal(
    match_tensor: tensor.Tensor,
    pytorch_tensor: torch.Tensor,
    check_grad=False,
    debug: bool = False,
) -> bool:
    m = to_tensor(match_tensor, get_grad=check_grad)
    t = pytorch_tensor.grad if check_grad else pytorch_tensor
    if t.ndim == 1:
        m.squeeze_()
    res = torch.allclose(m, t, rtol=1e-02, atol=1e-05)
    if not res or debug:
        print("Match: ", m)
        print("Torch: ", t)
    return res


def to_tensor(
    match_tensor: tensor.Tensor, requires_grad=False, get_grad=False
) -> torch.Tensor:
    match_tensor_data = match_tensor.grad if get_grad else match_tensor.data
    torch_tensor = None
    if match_tensor_data.use_numpy:
        torch_tensor = torch.from_numpy(np.array(match_tensor_data._numpy_data)).float()

    else:
        if match_tensor_data._data == None:
            torch_tensor = torch.tensor(data=match_tensor_data.item()).float()
        else:
            torch_tensor = torch.zeros(match_tensor_data.shape).float()
            for index in itertools.product(
                *[range(dim) for dim in match_tensor_data.shape]
            ):
                torch_tensor[index] = match_tensor_data[index].item()

    torch_tensor.requires_grad = requires_grad
    return torch_tensor


def mat_and_ten(shape: tuple[int] = None, use_numpy: bool = False):
    # Generate a random dimension
    if not shape:
        dim = random.randint(2,5)
        shape = (random.randint(1,5) for _ in range(dim))
    mat = randn(*shape, use_numpy=use_numpy)
    ten = to_tensor(mat, requires_grad=True)
    return mat, ten


class TestTensor(unittest.TestCase):
    def test_matmul_numpy(self):
        mat1, ten1 = mat_and_ten((1,2,3), use_numpy=True)
        mat2, ten2 = mat_and_ten((2,3,4), use_numpy=True)

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))

    def test_matmul_nd_1d(self):
        mat1, ten1 = mat_and_ten((3,3,2,2,3))
        mat2, ten2 = mat_and_ten((3,))

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))

    def test_matmul(self):
        mat1, ten1 = mat_and_ten((3,3,2,2,3))
        mat2, ten2 = mat_and_ten((3,1,3,4))

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))


    def test_pow_int_numpy(self):
        mat, ten = mat_and_ten(use_numpy=True)
        
        random_exponent = 2 # For some reason this works only with integers
        mat_res = pow(mat, random_exponent)
        ten_res = pow(ten, random_exponent)
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_pow_int(self):
        mat, ten = mat_and_ten()
        
        random_exponent = 2 # For some reason this works only with integers
        mat_res = pow(mat, random_exponent)
        ten_res = pow(ten, random_exponent)
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_mul_numpy(self):
        mat1, ten1 = mat_and_ten((3,1,5,6), use_numpy=True)
        mat2, ten2 = mat_and_ten((3,4,1,1), use_numpy=True)

        mat_res = mat1 * mat2
        ten_res = ten1 * ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))

    def test_mul(self):
        mat1, ten1 = mat_and_ten((2,1,5,6))
        mat2, ten2 = mat_and_ten((1,1,1))

        mat_res = mat1 * mat2
        ten_res = ten1 * ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))

    
    def test_add_numpy(self):
        mat1, ten1 = mat_and_ten((3,4,5,6), use_numpy=True)
        mat2, ten2 = mat_and_ten((3,4,1,1), use_numpy=True)

        mat_res = mat1 + mat2
        ten_res = ten1 + ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))

    def test_add(self):
        mat1, ten1 = mat_and_ten((3,4,5))
        mat2, ten2 = mat_and_ten((3,4,1))

        mat_res = mat1 + mat2
        ten_res = ten1 + ten2
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(almost_equal(mat2, ten2, check_grad=True))

    def test_sigmoid_numpy(self):
        mat, ten = mat_and_ten(use_numpy=True)

        mat_res = mat.sigmoid()
        ten_res = ten.sigmoid()
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_sigmoid(self):
        mat, ten = mat_and_ten()

        mat_res = mat.sigmoid()
        ten_res = ten.sigmoid()
        self.assertTrue(almost_equal(mat_res, ten_res))
        
        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_mean_numpy(self):
        mat, ten = mat_and_ten((2, 3, 4, 5), use_numpy=True)

        mat_mean = mat.mean()
        ten_mean = ten.mean()
        self.assertTrue(almost_equal(mat_mean, ten_mean))

        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_mean(self):
        mat, ten = mat_and_ten()

        mat_mean = mat.mean()
        ten_mean = ten.mean()
        self.assertTrue(almost_equal(mat_mean, ten_mean))

        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_relu_numpy(self):
        mat, ten = mat_and_ten(use_numpy=True)

        mat_relu = mat.relu()
        ten_relu = ten.relu()
        self.assertTrue(almost_equal(mat_relu, ten_relu))

        # Check backward with sum
        match_val = mat_relu.sum()
        ten_val = ten_relu.sum()
        match_val.backward()
        ten_val.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_relu(self):
        mat, ten = mat_and_ten()

        mat_relu = mat.relu()
        ten_relu = ten.relu()
        self.assertTrue(almost_equal(mat_relu, ten_relu))

        # Check backward with sum
        match_val = mat_relu.sum()
        ten_val = ten_relu.sum()
        match_val.backward()
        ten_val.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_sum_dim_numpy(self):
        mat, ten = mat_and_ten((2, 3, 4), use_numpy=True)

        mat_sum = mat.sum((0, 1))
        ten_sum = ten.sum((0, 1))
        self.assertTrue(almost_equal(mat_sum, ten_sum))

        # Check backward with sum
        match_val = mat_sum.sum()
        ten_val = ten_sum.sum()
        match_val.backward()
        ten_val.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_sum_nodim(self):
        mat, ten = mat_and_ten()

        mat_sum = mat.sum()
        ten_sum = ten.sum()
        self.assertTrue(almost_equal(mat_sum, ten_sum))

        mat_sum.backward()
        ten_sum.backward()

        self.assertTrue(almost_equal(mat, ten, check_grad=True))

    def test_toTensor_numpy(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor_data = tensordata.TensorData(
            numpy_data=np.arange(24).reshape(2, 4, 3), use_numpy=True
        )
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_toTensor_singleton_numpy(self):
        torch_tensor = torch.tensor(47).float()

        match_tensor_data = tensordata.TensorData(value=47, use_numpy=True)
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_toTensor_singleton(self):
        torch_tensor = torch.tensor(47).float()

        match_tensor_data = tensordata.TensorData(value=47, use_numpy=False)
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_toTensor(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor_data = tensordata.TensorData(2, 4, 3, use_numpy=False)
        match_tensor_data._data = [tensordata.TensorData(value=i) for i in range(24)]
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))
