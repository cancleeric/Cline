import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import unittest
import numpy as np
from common.multi_layer_net import MultiLayerNetE

class TestMultiLayerNet(unittest.TestCase):
    def setUp(self):
        # 初始化網路參數
        self.input_size = 784
        self.hidden_sizes = [50, 50, 50]
        self.output_size = 10
        self.network = MultiLayerNetE(self.input_size, self.hidden_sizes, self.output_size)

    def test_predict(self):
        # 測試預測函式
        x = np.random.rand(2, self.input_size)
        y = self.network.predict(x)
        self.assertEqual(y.shape, (2, self.output_size))

    def test_loss(self):
        # 測試損失函數
        x = np.random.rand(2, self.input_size)
        t = np.random.rand(2, self.output_size)
        loss = self.network.loss(x, t)
        self.assertIsInstance(loss, float)

    def test_gradient(self):
        # 測試梯度計算
        x = np.random.rand(2, self.input_size)
        t = np.random.rand(2, self.output_size)
        grads = self.network.gradient(x, t)
        self.assertEqual(grads['W1'].shape, (self.input_size, self.hidden_sizes[0]))
        self.assertEqual(grads['b1'].shape, (self.hidden_sizes[0],))
        self.assertEqual(grads['W2'].shape, (self.hidden_sizes[0], self.hidden_sizes[1]))
        self.assertEqual(grads['b2'].shape, (self.hidden_sizes[1],))
        self.assertEqual(grads['W3'].shape, (self.hidden_sizes[1], self.hidden_sizes[2]))
        self.assertEqual(grads['b3'].shape, (self.hidden_sizes[2],))
        self.assertEqual(grads['W4'].shape, (self.hidden_sizes[2], self.output_size))
        self.assertEqual(grads['b4'].shape, (self.output_size,))

    def test_numerical_gradient(self):
        # 測試數值梯度計算
        x = np.random.rand(2, self.input_size)
        t = np.random.rand(2, self.output_size)
        grads = self.network.numerical_gradient(x, t)
        self.assertEqual(grads['W1'].shape, (self.input_size, self.hidden_sizes[0]))
        self.assertEqual(grads['b1'].shape, (self.hidden_sizes[0],))
        self.assertEqual(grads['W2'].shape, (self.hidden_sizes[0], self.hidden_sizes[1]))
        self.assertEqual(grads['b2'].shape, (self.hidden_sizes[1],))
        self.assertEqual(grads['W3'].shape, (self.hidden_sizes[1], self.hidden_sizes[2]))
        self.assertEqual(grads['b3'].shape, (self.hidden_sizes[2],))
        self.assertEqual(grads['W4'].shape, (self.hidden_sizes[2], self.output_size))
        self.assertEqual(grads['b4'].shape, (self.output_size,))

if __name__ == '__main__':
    unittest.main()