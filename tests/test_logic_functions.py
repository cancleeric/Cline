import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from logic_functions import (
    step_function, sigmoid, relu_function, weighted_sum,
    identity_function, softmax_function, mean_squared_error, cross_entropy_error,
    numerical_derivative, gradient_function, gradient_descent, sigmoid_derivative
)

class TestLogicFunctions(unittest.TestCase):
    def test_step_function(self):
        x = np.array([-1.0, 1.0, 2.0])
        expected = np.array([0, 1, 1])
        np.testing.assert_array_equal(step_function(x), expected)

    def test_sigmoid_function(self):
        x = np.array([0.0])
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(sigmoid(x), expected)

    def test_relu_function(self):
        x = np.array([-1.0, 0.0, 1.0])
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu_function(x), expected)

    def test_weighted_sum(self):
        x = np.array([1.0, 2.0])
        weights = np.array([0.5, 0.5])
        bias = 0.1
        expected = 1.6
        self.assertAlmostEqual(weighted_sum(x, weights, bias), expected)

    def test_identity_function(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(identity_function(x), x)

    def test_mean_squared_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = 0.3333333333333333
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), expected)

    # Test function
    def test_cross_entropy_error(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
        expected = 0.2676  # Updated expected value based on manual calculation
        self.assertAlmostEqual(cross_entropy_error(y_true, y_pred), expected, places=3)

    def test_numerical_derivative(self):
        f = lambda x: x**2
        x = 3.0
        expected = 6.0  # f'(x) = 2x, so f'(3) = 6
        self.assertAlmostEqual(numerical_derivative(f, x), expected, places=5)

    def test_gradient_function(self):
        f = lambda x: x[0]**2 + x[1]**2
        x = np.array([3.0, 4.0])
        expected = np.array([6.0, 8.0])  # f'(x) = [2*x1, 2*x2], so f'([3, 4]) = [6, 8]
        np.testing.assert_array_almost_equal(gradient_function(f, x), expected, decimal=5)

    def test_gradient_descent(self):
        f = lambda x: x[0]**2 + x[1]**2  # 目標函數
        init_x = np.array([3.0, 4.0])  # 初始點
        learning_rate = 0.1
        num_iterations = 100
        expected_x = np.array([0.0, 0.0])  # 最小值點
        x, _ = gradient_descent(f, init_x, learning_rate, num_iterations)
        np.testing.assert_array_almost_equal(x, expected_x, decimal=5)

    def test_sigmoid_derivative(self):
        # Test sigmoid derivative at various points
        x_values = np.array([-1000, -10, -1, 0, 1, 10, 1000])
        expected_derivatives = sigmoid(x_values) * (1 - sigmoid(x_values))
        
        for x, expected in zip(x_values, expected_derivatives):
            with self.subTest(x=x):
                result = sigmoid_derivative(x)
                self.assertAlmostEqual(result, expected, places=7, msg=f"Failed at x={x}")

if __name__ == "__main__":
    unittest.main()
