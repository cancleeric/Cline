import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import unittest
import numpy as np
from simple_net import SimpleNet
from common.functions import numerical_derivative

class TestSimpleNet(unittest.TestCase):
    def setUp(self):
        self.net = SimpleNet()

    def test_predict(self):
        x = np.array([[1.0, 0.5]])
        result = self.net.predict(x)
        self.assertEqual(result.shape, (1, 3))
        print("Predict output:", result)

    def test_loss(self):
        x = np.array([[1.0, 0.5]])
        t = np.array([[0, 0, 1]])
        loss = self.net.loss(x, t)
        self.assertIsInstance(loss, float)
        print("Loss value:", loss)

    def test_gradient(self):
        x = np.array([[1.0, 0.5]])
        t = np.array([[0, 0, 1]])
        
        # 定義 f 為計算損失的函數
        def f(W):
            self.net.W = W
            return self.net.loss(x, t)
        
        gradient = numerical_derivative(f, self.net.W)
        self.assertEqual(gradient.shape, self.net.W.shape)
        print("Gradient:", gradient)

if __name__ == "__main__":
    unittest.main()