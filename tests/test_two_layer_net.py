import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from two_layer_neural_network import TwoLayerNet

class TestTwoLayerNet(unittest.TestCase):
    """
    測試 TwoLayerNet 類別
    """
    def setUp(self):
        """
        初始化測試資料
        """
        self.net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        self.x = np.random.randn(100, 784)
        self.t = np.random.randn(100, 10)

    def test_predict(self):
        """
        測試預測函式
        """
        y = self.net.predict(self.x)
        self.assertEqual(y.shape, (100, 10))

    def test_loss(self):
        """
        測試損失函式
        """
        loss = self.net.loss(self.x, self.t)
        self.assertIsInstance(loss, float)

    def test_numerical_gradient(self):
        """
        測試數值梯度計算
        """
        grads = self.net.numerical_gradient(self.x, self.t)
        self.assertIsInstance(grads, dict)
        self.assertIn('W1', grads)
        self.assertIn('b1', grads)
        self.assertIn('W2', grads)
        self.assertIn('b2', grads)

    def test_gradient(self):
        """
        測試梯度計算
        """
        grads = self.net.gradient(self.x, self.t)
        self.assertIsInstance(grads, dict)
        self.assertIn('W1', grads)
        self.assertIn('b1', grads)
        self.assertIn('W2', grads)
        self.assertIn('b2', grads)

if __name__ == '__main__':
    unittest.main()
