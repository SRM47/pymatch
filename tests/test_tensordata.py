import unittest
import torch
from match.tensordata import TensorData
import itertools


def to_tensor(match_tensor: TensorData) -> torch.Tensor:
    torch_tensor = torch.Tensor(size=match_tensor.shape).float()
    for index in itertools.product(*[range(dim) for dim in match_tensor.shape]):
        torch_tensor[index] = match_tensor[index].item()
    return torch_tensor


def almost_equal(match_tensor: TensorData, torch_tensor: torch.Tensor) -> bool:
    m = to_tensor(match_tensor)
    t = torch_tensor.float()
    if t.ndim == 1:
        m.squeeze_()
    if not torch.allclose(m, t, rtol=1e-02, atol=1e-05):
        print(m)
        print(t)
        return False
    return True


def same_references(match_tensor1: TensorData, match_tensor2: TensorData):
    # check if they are the same object, which they should to match pytorch functionality
    for e1, e2 in zip(match_tensor1._data, match_tensor2._data):
        if e1 is not e2:
            return False
    return True


class TestTensorDataTest(unittest.TestCase):
    def test_concatenate_default(self):
        torch_tensor = torch.arange(6).reshape(2, 3).float()
        match_tensor = TensorData(2, 3)
        match_tensor._data = [TensorData(value=i) for i in range(6)]

        self.assertTrue(
            almost_equal(
                TensorData.concatenate((match_tensor, match_tensor, match_tensor)),
                torch.cat((torch_tensor, torch_tensor, torch_tensor)),
            )
        )
        
    # def test_concatenate_column(self):
    #     torch_tensor = torch.arange(6).reshape(2, 3).float()
    #     match_tensor = TensorData(2, 3)
    #     match_tensor._data = [TensorData(value=i) for i in range(6)]

    #     self.assertTrue(
    #         almost_equal(
    #             TensorData.concatenate((match_tensor, match_tensor, match_tensor), 1),
    #             torch.cat((torch_tensor, torch_tensor, torch_tensor), 1),
    #         )
    #     )

    # def test_concatenate_nd(self):
    #     torch_tensor = torch.arange(24).reshape(2, 4, 3).float()
    #     match_tensor = TensorData(2, 3, 4)
    #     match_tensor._data = [TensorData(value=i) for i in range(24)]

    #     self.assertTrue(
    #         almost_equal(
    #             TensorData.concatenate((match_tensor, match_tensor, match_tensor), 1),
    #             torch.cat((torch_tensor, torch_tensor, torch_tensor), 1),
    #         )
    #     )

    def test_mean_nodim(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor = TensorData(2, 4, 3)
        match_tensor._data = [TensorData(value=i) for i in range(24)]

        self.assertEqual(match_tensor.mean().item(), torch_tensor.mean().item())

    def test_mean_keepdim(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor = TensorData(2, 4, 3)
        match_tensor._data = [TensorData(value=i) for i in range(24)]

        self.assertTrue(
            almost_equal(
                match_tensor.mean((1, 2), keepdims=True),
                torch_tensor.mean((1, 2), keepdim=True),
            )
        )

    def test_mean(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor = TensorData(2, 4, 3)
        match_tensor._data = [TensorData(value=i) for i in range(24)]

        self.assertTrue(almost_equal(match_tensor.mean((0,)), torch_tensor.mean((0,))))

    def test_sum_nodim(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3)

        match_tensor = TensorData(2, 4, 3)
        match_tensor._data = [TensorData(value=i) for i in range(24)]

        self.assertEqual(match_tensor.sum().item(), torch_tensor.sum().item())

    def test_sum_2(self):
        torch_tensor = torch.arange(48).reshape(2, 4, 3, 2)

        match_tensor = TensorData(2, 4, 3, 2)
        match_tensor._data = [TensorData(value=i) for i in range(48)]

        self.assertTrue(
            almost_equal(match_tensor.sum((1, 2)), torch_tensor.sum((1, 2)))
        )

    def test_sum_keepdim_2(self):
        torch_tensor = torch.arange(48).reshape(2, 4, 3, 2)

        match_tensor = TensorData(2, 4, 3, 2)
        match_tensor._data = [TensorData(value=i) for i in range(48)]

        self.assertTrue(
            almost_equal(
                match_tensor.sum((1, 2), keepdims=True),
                torch_tensor.sum((1, 2), keepdims=True),
            )
        )

    def test_sum_keepdim(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3)

        match_tensor = TensorData(2, 4, 3)
        match_tensor._data = [TensorData(value=i) for i in range(24)]

        self.assertTrue(
            almost_equal(
                match_tensor.sum((1, 2), keepdims=True),
                torch_tensor.sum((1, 2), keepdim=True),
            )
        )

    def test_sum(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3)

        match_tensor = TensorData(2, 4, 3)
        match_tensor._data = [TensorData(value=i) for i in range(24)]

        self.assertTrue(almost_equal(match_tensor.sum((0,)), torch_tensor.sum((0,))))

    def test_matmul_nd_1d(self):
        torch_tensor_1 = torch.arange(24).reshape(2, 4, 3)
        torch_tensor_2 = torch.arange(3)

        match_tensor_2 = TensorData(3)
        match_tensor_2._data = [TensorData(value=i) for i in range(3)]

        match_tensor_1 = TensorData(2, 4, 3)
        match_tensor_1._data = [TensorData(value=i) for i in range(24)]

        product_tensor = match_tensor_1 @ match_tensor_2
        torch_result = torch_tensor_1 @ torch_tensor_2

        self.assertEqual(product_tensor.shape, torch_result.shape)

        self.assertTrue(almost_equal(product_tensor, torch_result))
        self.assertEqual(match_tensor_2.shape, (3,))

    def test_matmul_1d_nd(self):
        torch_tensor_1 = torch.arange(4)
        torch_tensor_2 = torch.arange(24).reshape(2, 4, 3)

        match_tensor_1 = TensorData(4)
        match_tensor_1._data = [TensorData(value=i) for i in range(4)]

        match_tensor_2 = TensorData(2, 4, 3)
        match_tensor_2._data = [TensorData(value=i) for i in range(24)]

        product_tensor = match_tensor_1 @ match_tensor_2
        torch_result = torch_tensor_1 @ torch_tensor_2

        self.assertEqual(product_tensor.shape, torch_result.shape)

        self.assertTrue(almost_equal(product_tensor, torch_result))
        self.assertEqual(match_tensor_1.shape, (4,))

    def test_matmul_nd_nd(self):
        torch_tensor_1 = torch.arange(56).reshape(2, 1, 7, 4)
        torch_tensor_2 = torch.arange(24).reshape(2, 4, 3)

        match_tensor_1 = TensorData(2, 1, 7, 4)
        match_tensor_1._data = [TensorData(value=i) for i in range(56)]

        match_tensor_2 = TensorData(2, 4, 3)
        match_tensor_2._data = [TensorData(value=i) for i in range(24)]

        product_tensor = match_tensor_1 @ match_tensor_2
        torch_result = torch_tensor_1 @ torch_tensor_2

        self.assertEqual(product_tensor.shape, torch_result.shape)

        self.assertTrue(almost_equal(product_tensor, torch_result))

    def test_matmul_2d_1d(self):
        torch_tensor_1 = torch.arange(56).reshape(7, 8)
        torch_tensor_2 = torch.arange(8).reshape(8)

        match_tensor_1 = TensorData(7, 8)
        match_tensor_1._data = [TensorData(value=i) for i in range(56)]

        match_tensor_2 = TensorData(8)
        match_tensor_2._data = [TensorData(value=i) for i in range(8)]

        product_tensor = match_tensor_1 @ match_tensor_2
        torch_result = torch_tensor_1 @ torch_tensor_2

        self.assertEqual(product_tensor.shape, torch_result.shape)
        self.assertEqual(product_tensor._item, None)

        self.assertTrue(almost_equal(product_tensor, torch_result))

    def test_matmul_1d_2d(self):
        torch_tensor_1 = torch.arange(8).reshape(8)
        torch_tensor_2 = torch.arange(16).reshape(8, 2)

        match_tensor_1 = TensorData(8)
        match_tensor_1._data = [TensorData(value=i) for i in range(8)]

        match_tensor_2 = TensorData(8, 2)
        match_tensor_2._data = [TensorData(value=i) for i in range(16)]

        product_tensor = match_tensor_1 @ match_tensor_2
        torch_result = torch_tensor_1 @ torch_tensor_2

        self.assertEqual(product_tensor.shape, torch_result.shape)
        self.assertEqual(product_tensor._item, None)

        self.assertTrue(almost_equal(product_tensor, torch_result))

    def test_matmul_2d_2d(self):
        torch_tensor_1 = torch.arange(56).reshape(7, 8)
        torch_tensor_2 = torch.arange(16).reshape(8, 2)

        match_tensor_1 = TensorData(7, 8)
        match_tensor_1._data = [TensorData(value=i) for i in range(56)]

        match_tensor_2 = TensorData(8, 2)
        match_tensor_2._data = [TensorData(value=i) for i in range(16)]

        product_tensor = match_tensor_1 @ match_tensor_2
        torch_result = torch_tensor_1 @ torch_tensor_2

        self.assertEqual(product_tensor.shape, torch_result.shape)
        self.assertEqual(product_tensor._item, None)

        self.assertTrue(almost_equal(product_tensor, torch_result))

    def test_matmul_1d_1d(self):
        match_tensor_1 = TensorData(9)
        match_tensor_1._data = [TensorData(value=i) for i in range(9)]

        match_tensor_2 = TensorData(9)
        match_tensor_2._data = [TensorData(value=i) for i in range(9)]

        product_tensor = match_tensor_1 @ match_tensor_2

        self.assertEqual(product_tensor.shape, ())
        self.assertIsNone(product_tensor._data)
        self.assertEqual(product_tensor.item(), 204)

    def test_binary_operations(self):
        torch_tensor_low_dim_1 = torch.ones(2, 2)
        torch_tensor_low_dim_2 = torch.ones(2, 2)
        torch_tensor_singleton = torch.ones(1)[0]
        torch_tensor_high_dim = torch.ones(1, 3, 2, 1)

        match_tensor_low_dim_1 = TensorData(2, 2)
        match_tensor_low_dim_1.ones_()

        match_tensor_low_dim_2 = TensorData(2, 2)
        match_tensor_low_dim_2.ones_()

        match_tensor_singleton = TensorData(value=1.0)

        match_tensor_high_dim = TensorData(1, 3, 2, 1)
        match_tensor_high_dim.ones_()

        # low low
        self.assertTrue(
            almost_equal(
                match_tensor_low_dim_1 + match_tensor_low_dim_2,
                torch_tensor_low_dim_1 + torch_tensor_low_dim_2,
            )
        )

        # singleton low
        self.assertTrue(
            almost_equal(
                match_tensor_singleton + match_tensor_low_dim_2,
                torch_tensor_singleton + torch_tensor_low_dim_2,
            )
        )

        # singleton singleton
        match_singleton_add = TensorData(value=1.0) + TensorData(value=1.0)
        self.assertEqual(match_singleton_add.shape, ())
        self.assertEqual(match_singleton_add._data, None)
        self.assertEqual(match_singleton_add.item(), 2)

        # low high
        self.assertTrue(
            almost_equal(
                match_tensor_low_dim_2 + match_tensor_high_dim,
                torch_tensor_low_dim_2 + torch_tensor_high_dim,
            )
        )

    def test_transpose(self):
        # make torch tensor
        torch_tensor = torch.arange(9).reshape(3, 1, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 1, 3)
        match_tensor._data = [TensorData(value=i) for i in range(9)]

        self.assertTrue(almost_equal(match_tensor.T, torch_tensor.T))

    def test_permute(self):
        # make torch tensor
        torch_tensor = torch.arange(9).reshape(3, 1, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 1, 3)
        match_tensor._data = [TensorData(value=i) for i in range(9)]

        self.assertTrue(
            almost_equal(match_tensor.permute(2, 0, 1), torch_tensor.permute(2, 0, 1))
        )

    def test_reshape_1d_to_singleton(self):
        # make the match tensor
        match_tensor = TensorData(1)
        # reshape the match tensor
        match_tensor_reshaped = match_tensor.reshape(())

        self.assertEqual(match_tensor.shape, (1,))
        self.assertEqual(match_tensor_reshaped.shape, ())
        self.assertEqual(match_tensor_reshaped.item(), 0)
        self.assertEqual(match_tensor_reshaped._data, None)

        # They must have the same reference
        match_tensor._data[0]._item = 42
        self.assertEqual(match_tensor_reshaped.item(), 42)

    def test_reshape_singleton_to_1d(self):
        # make the match tensor
        match_tensor = TensorData(value=47)
        # reshape the match tensor
        match_tensor_reshaped = match_tensor.reshape((1,))

        self.assertEqual(match_tensor.shape, ())
        self.assertEqual(match_tensor.item(), 47)
        self.assertEqual(match_tensor._data, None)

        self.assertEqual(len(match_tensor_reshaped._data), 1)
        self.assertEqual(match_tensor_reshaped.shape, (1,))
        self.assertEqual(match_tensor_reshaped.item(), 47)
        self.assertEqual(match_tensor_reshaped._item, None)

        # They must have the same reference
        match_tensor._item = 42
        self.assertEqual(match_tensor_reshaped.item(), 42)

    def test_reshape_failure(self):
        match_tensor = TensorData(2, 3, 4)
        self.assertRaises(RuntimeError, lambda: match_tensor.reshape((5, 5, 5)))

    def test_reshape(self):
        # make the match tensor
        match_tensor = TensorData(2, 3, 4)
        # reshape the match tensor
        match_tensor_reshaped = match_tensor.reshape((4, 3, 2))

        self.assertEqual(match_tensor.shape, (2, 3, 4))
        self.assertEqual(match_tensor_reshaped.shape, (4, 3, 2))
        self.assertEqual(len(match_tensor._data), len(match_tensor_reshaped._data))
        self.assertTrue(same_references(match_tensor, match_tensor_reshaped))
        self.assertRaises(IndexError, lambda: match_tensor_reshaped[1, 2, 3])

    def test_broadcast_failure(self):
        # make corresponding match tensor
        match_tensor = TensorData(3, 1, 3)
        match_tensor._data = [TensorData(value=i) for i in range(9)]

        self.assertRaises(ValueError, lambda: match_tensor.broadcast(2, 2, 1, 3, 3))
        self.assertRaises(RuntimeError, lambda: match_tensor.broadcast(3, 3))

    def test_broadcast_singleton(self):
        # make torch tensor
        torch_tensor = torch.arange(9).reshape(3, 1, 3)[0, 0, 0]
        # make corresponding match tensor
        match_tensor = TensorData(value=0)

        torch_tensor_broadcasted = torch.broadcast_to(torch_tensor, (3, 3))
        match_tensor_broadcasted = match_tensor.broadcast(3, 3)

        self.assertEqual(match_tensor_broadcasted.shape, (3, 3))
        self.assertTrue(
            almost_equal(match_tensor_broadcasted, torch_tensor_broadcasted)
        )

    def test_broadcast(self):
        # make torch tensor
        torch_tensor = torch.arange(9).reshape(3, 1, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 1, 3)
        match_tensor._data = [TensorData(value=i) for i in range(9)]

        torch_tensor_broadcasted = torch.broadcast_to(torch_tensor, (2, 2, 3, 3, 3))
        match_tensor_broadcasted = match_tensor.broadcast(2, 2, 3, 3, 3)

        self.assertEqual(match_tensor_broadcasted.shape, (2, 2, 3, 3, 3))
        self.assertTrue(
            almost_equal(match_tensor_broadcasted, torch_tensor_broadcasted)
        )

    def test_getitem_partial_index(self):
        # make torch tensor
        torch_tensor = torch.arange(27).reshape(3, 3, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 3, 3)
        match_tensor._data = [TensorData(value=i) for i in range(27)]

        torch_tensor_slice = torch_tensor[1:]
        match_tensor_slice = match_tensor[1:]

        self.assertEqual(match_tensor_slice.shape, (2, 3, 3))
        self.assertTrue(almost_equal(match_tensor_slice, torch_tensor_slice))

    def test_setitem_single_value_index(self):
        # make torch tensor
        torch_tensor = torch.arange(27).reshape(3, 3, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 3, 3)
        match_tensor._data = [TensorData(value=i) for i in range(27)]

        torch_tensor[2:] = 0
        match_tensor[2:] = 0

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_setitem_partial_index(self):
        # make torch tensor
        torch_tensor = torch.arange(27).reshape(3, 3, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 3, 3)
        match_tensor._data = [TensorData(value=i) for i in range(27)]

        torch_tensor[2:] = torch.zeros((1, 3, 3))
        match_tensor[2:] = TensorData(1, 3, 3)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_setitem_slice(self):
        # make torch tensor
        torch_tensor = torch.arange(1, 9).reshape(2, 4)
        # make corresponding match tensor
        match_tensor = TensorData(2, 4)
        match_tensor._data = [TensorData(value=i + 1) for i in range(8)]

        torch_tensor[:, 1::2] = torch.zeros((2, 2))
        match_tensor[:, 1::2] = TensorData(2, 2)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_getitem_slice(self):
        match_tensor = TensorData(2, 4)
        match_tensor._data = [TensorData(value=i + 1) for i in range(8)]
        match_tensor_slice = match_tensor[:, 1]
        self.assertEqual(match_tensor_slice.shape, (2,))
        self.assertEqual(match_tensor_slice._data[0].item(), 2)
        self.assertEqual(match_tensor_slice._data[1].item(), 6)

        match_tensor_slice = match_tensor[:, 1::2]
        self.assertEqual(match_tensor_slice.shape, (2, 2))
        self.assertEqual(match_tensor_slice._data[0].item(), 2)
        self.assertEqual(match_tensor_slice._data[1].item(), 4)
        self.assertEqual(match_tensor_slice._data[2].item(), 6)
        self.assertEqual(match_tensor_slice._data[3].item(), 8)

    def test_getitem_reference(self):
        tensor = TensorData(3, 3, 3)
        tensor_subset = tensor[2, 0, :]
        self.assertEqual(type(tensor_subset), TensorData)
        self.assertEqual(tensor_subset.shape, (3,))
        for i in range(3):
            self.assertEqual(tensor_subset._data[i].item(), 0)

        self.assertEqual(tensor._data[18].item(), 0)
        tensor_subset[0] = 47
        self.assertEqual(tensor_subset._data[0].item(), 47)
        self.assertEqual(tensor._data[18].item(), 47)

    def test_getitem_single(self):
        tensor = TensorData(2, 3, 4, 5)
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor[1, 2, 3, 4].item(), 0)
        self.assertRaises(IndexError, lambda: tensor[2, 0, 0, 0])

    def test_setitem_single_number(self):
        tensor = TensorData(2, 3, 4, 5)
        tensor[0, 0, 0, 0] = 47.0
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor._data[0].item(), 47.0)

    def test_setitem_single_tensordata(self):
        tensor = TensorData(2, 3, 4, 5)
        tensor[0, 0, 0, 0] = TensorData(value=47)
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor._data[0].item(), 47.0)

    def test_create_tensor_data_no_data(self):
        tensor = TensorData(5, 5, 0)
        self.assertEqual(tensor._data, [])
        self.assertEqual(tensor._item, None)
        self.assertRaises(ValueError, lambda: tensor.item())

    def test_create_tensor_data_singleton(self):
        tensor = TensorData(value=47.0)
        self.assertEqual(tensor.item(), 47.0)
        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor._data, None)

    def test_create_tensor_data(self):
        tensor = TensorData(5, 2, 4)
        self.assertRaises(ValueError, lambda: tensor.item())
        self.assertEqual(tensor._item, None)
        self.assertEqual(tensor._data[0].item(), 0)

    def test_initialize_strides(self):
        tensor = TensorData(1, 2, 1)
        self.assertEqual(tensor._strides, (2, 1, 1))
        tensor = TensorData(6, 4, 2, 5, 7)
        self.assertEqual(tensor._strides, (280, 70, 35, 7, 1))

    def test_single_to_multi_rank_translation(self):
        tensor = TensorData(4, 3, 5)
        self.assertEqual(
            tensor._TensorData__single_to_multi_rank_translation(26), (1, 2, 1)
        )
        self.assertEqual(
            tensor._TensorData__single_to_multi_rank_translation(0), (0, 0, 0)
        )
        self.assertRaises(
            IndexError, lambda: tensor._TensorData__single_to_multi_rank_translation(-1)
        )
        self.assertRaises(
            IndexError, lambda: tensor._TensorData__single_to_multi_rank_translation(60)
        )

    def test_multi_to_single_rank_translation(self):
        tensor = TensorData(4, 3, 5)
        self.assertEqual(
            tensor._TensorData__multi_to_single_rank_translation((1, 2, 1)), 26
        )
        self.assertEqual(
            tensor._TensorData__multi_to_single_rank_translation((0, 0, 0)), 0
        )
        self.assertRaises(
            IndexError,
            lambda: tensor._TensorData__multi_to_single_rank_translation((-1, 0, -1)),
        )
        self.assertRaises(
            IndexError,
            lambda: tensor._TensorData__multi_to_single_rank_translation((4, 2, 4)),
        )


if __name__ == "__main__":
    unittest.main()
