import unittest
import torch
from src.loss.RegressionLossEvaluator import RegressionLossEvaluator


class TestRegressionLossEvaluator(unittest.TestCase):
    def test_get_squared_diff_mean_vector_with_valid_tensors(self):
        # Test the method with valid tensors
        output_tensor = torch.tensor(range(24)).reshape((2,4,3)) * 1.
        label_tensor = torch.tensor(range(24)).reshape((2,4,3)) * 1.
        expected_loss = torch.tensor([0.0, 0.0, 0.0])
        squared_diff_mean_vector = RegressionLossEvaluator.get_squared_diff_mean_vector(output_tensor, label_tensor)
        self.assertTrue(torch.equal(squared_diff_mean_vector, expected_loss))

    def test_get_squared_diff_mean_vector_with_nonzero_loss(self):
        # Test the method with tensors that would result in a nonzero loss
        output_tensor = torch.tensor(range(24)).reshape((2,4,3)) * 1.
        label_tensor = torch.tensor(range(24)).reshape((2,4,3)) * 1. + 1.
        expected_loss = torch.tensor([1., 1., 1.])
        squared_diff_mean_vector = RegressionLossEvaluator.get_squared_diff_mean_vector(output_tensor, label_tensor)
        self.assertTrue(torch.allclose(squared_diff_mean_vector, expected_loss))

    def test_get_squared_diff_mean_vector_with_mismatched_tensor_shapes(self):
        # Test the method with tensors of mismatched shapes
        output_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        label_tensor = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        with self.assertRaises(ValueError):
            _ = RegressionLossEvaluator.get_squared_diff_mean_vector(output_tensor, label_tensor)

    def test_get_squared_diff_mean_vector_with_empty_tensors(self):
        # Test the method with empty tensors
        output_tensor = torch.tensor([])
        label_tensor = torch.tensor([])
        with self.assertRaises(ValueError):
            _ = RegressionLossEvaluator.get_squared_diff_mean_vector(output_tensor, label_tensor)

    def test_mask_by_threes_with_valid_input(self):
        # Test with a valid tensor that should return a proper mask
        tensor = torch.tensor([[[1.0, 0.0, 0.0], 
                                [0.0, 2.0, 0.0]],
                                [[0.,0.,0.],
                                 [3.,0.,4.]]])
        expected_mask = torch.tensor([[[1.0, 1.0, 1.0], 
                                       [1.0, 1.0, 1.0]],
                                       [[0., 0., 0.],
                                        [1., 1., 1.]]])
        mask = RegressionLossEvaluator.get_mask_by_threes(tensor)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_mask_by_threes_with_theshold(self):
        # Test with a valid tensor that should return a proper mask
        tensor = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        expected_mask = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        mask = RegressionLossEvaluator.get_mask_by_threes(tensor, threshold=1.5)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_mask_by_threes_with_invalid_dimensions(self):
        # Test with a tensor that is not 3-dimensional
        tensor = torch.tensor([[1.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            _ = RegressionLossEvaluator.get_mask_by_threes(tensor)

    def test_mask_by_threes_with_empty_tensor(self):
        # Test with an empty tensor
        tensor = torch.empty(0)
        with self.assertRaises(ValueError):
            _ = RegressionLossEvaluator.get_mask_by_threes(tensor)

    def test_mask_by_threes_with_invalid_last_dimension(self):
        # Test with a tensor where the last dimension is not divisible by 3
        tensor = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        with self.assertRaises(ValueError):
            _ = RegressionLossEvaluator.get_mask_by_threes(tensor)

    def test_mask_by_threes_with_zeros(self):
        # Test with a tensor that contains only zeros
        tensor = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        expected_mask = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        mask = RegressionLossEvaluator.get_mask_by_threes(tensor)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_mask_by_threes_with_one_non_zero(self):
        # Test with a tensor that contains only zeros
        tensor = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]])
        expected_mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]])
        mask = RegressionLossEvaluator.get_mask_by_threes(tensor)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_get_mean_norm_error_shape_mismatch(self):
        output_tensor = torch.rand((3, 2, 6))
        label_tensor = torch.rand((3, 2, 9))
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor)

    def test_get_mean_norm_error_tensor_not_3d(self):
        output_tensor = torch.rand(2, 6)
        label_tensor = torch.rand(2, 6)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor)

    def test_get_mean_norm_error_empty_tensor(self):
        output_tensor = torch.rand(0, 6)
        label_tensor = torch.rand(0, 6)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor)

    def test_get_mean_norm_error_final_dimension_not_divisible_by_three(self):
        output_tensor = torch.rand(3, 2, 7)
        label_tensor = torch.rand(3, 2, 7)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor)

    def test_get_mean_norm_error_zero(self):
        output_tensor = torch.tensor([[[3., -2.0, 4.0], 
                                       [4.0, 5.0, 6.0]],
                                       [[3., -2.0, 4.0], 
                                       [4.0, 5.0, 6.0]]])
        label_tensor = torch.tensor([[[1.0, 2.0, 3.0], 
                                      [4.0, 5.0, 6.0]],
                                      [[1.0, 2.0, 3.0], 
                                      [4.0, 5.0, 6.0]]])
        mean_norm_error = RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor)

        expected_mean_norm_error = torch.tensor([0.])

        self.assertTrue(torch.isclose(mean_norm_error, expected_mean_norm_error))

    def test_get_mean_norm_error_non_zero(self):
        output_tensor = torch.tensor([[[3., -2.0, 4.0], 
                                       [4.0, 5.0, 6.0]],
                                       [[3., -2.0, 4.0], 
                                       [4.0, 5.0, 7.0]]])
        label_tensor = torch.tensor([[[1.0, 2.0, 3.0], 
                                      [4.0, 5.0, 6.0]],
                                      [[1.0, 2.0, 3.0], 
                                      [4.0, 5.0, 6.0]]])
        mean_norm_error = RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor)

        expected_mean_norm_error = torch.tensor([0.5])

        self.assertTrue(torch.isclose(mean_norm_error, expected_mean_norm_error))

    def test_get_mean_norm_error_zero_vec_size_6(self):
        output_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        label_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        mean_norm_error = RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor, vec_size=6)

        expected_mean_norm_error = torch.tensor([0.0])

        self.assertTrue(torch.isclose(mean_norm_error, expected_mean_norm_error))

    def test_get_mean_norm_error_non_zero_vec_size_6(self):
        output_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        label_tensor = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        mean_norm_error = RegressionLossEvaluator.get_mean_norm_error(output_tensor, label_tensor, vec_size=6)

        expected_mean_norm_error = torch.norm(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

        self.assertTrue(torch.isclose(mean_norm_error, expected_mean_norm_error))

    def test_shape_mismatch(self):
        output_force = torch.rand(3, 2, 6)
        label_force = torch.rand(4, 2, 6)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_com_acc_error(output_force, label_force)

    def test_not_3d_tensor(self):
        output_force = torch.rand(2, 6)
        label_force = torch.rand(2, 6)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_com_acc_error(output_force, label_force)

    def test_empty_tensor(self):
        output_force = torch.empty(0, 0)
        label_force = torch.rand(3, 6)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_com_acc_error(output_force, label_force)

    def test_final_dimension_not_six(self):
        output_force = torch.rand(3, 2, 5)
        label_force = torch.rand(3, 2, 5)
        with self.assertRaises(ValueError):
            RegressionLossEvaluator.get_com_acc_error(output_force, label_force)

    def test_output_zero(self):
        output_force = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]])
        label_force = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]])
        com_acc_mean_error = RegressionLossEvaluator.get_com_acc_error(output_force, label_force)

        expected_com_acc_mean_error = torch.tensor([0.0])

        self.assertTrue(torch.isclose(com_acc_mean_error, expected_com_acc_mean_error))


# Run the unit tests
if __name__ == '__main__':
    unittest.main()
