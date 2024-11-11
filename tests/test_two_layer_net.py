import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

class TestTwoLayerNet(unittest.TestCase):
    """
    測試 TwoLayerNet 類別
    """
    def setUp(self):
        """
        初始化測試資料
        """
        # 初始化一個神經網路
        #load_mnist 這個函數是從 dataset/mnist.py 載入的
        (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot=True)

        self.net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        self.x = train_images[:3]
        self.t = train_labels[:3]

    def test_predict(self):
        """
        測試預測函式
        """
        y = self.net.predict(self.x)
        self.assertEqual(y.shape, (3, 10))

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

    # def test_check_gradient(self):
    #     """
    #     確認數值梯度和解析梯度之間的差異
    #     :param x: 輸入資料
    #     :param t: 真實標籤
    #     :return: 梯度差異
    #     """
    #     grad_numerical = self.net.numerical_gradient(self.x, self.t)
    #     grad_backprop = self.net.gradient(self.x, self.t)
        
    #     for key in grad_numerical.keys():
    #         diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    #         print(f'{key}: {diff}')
    #         self.assertTrue(diff < 1e-3, f'Gradient check failed for {key}, diff: {diff}')


def test_check_gradient(self):
    """
    確認數值梯度和解析梯度之間的差異
    """
    grad_numerical = self.net.numerical_gradient(self.x, self.t)
    grad_backprop = self.net.gradient(self.x, self.t)
    
    for key in grad_numerical.keys():
        numerator = np.linalg.norm(grad_backprop[key] - grad_numerical[key])
        denominator = np.linalg.norm(grad_backprop[key]) + np.linalg.norm(grad_numerical[key])
        relative_error = numerator / denominator
        print(f'{key}: {relative_error}')
        self.assertTrue(relative_error < 1e-5, f'Gradient check failed for {key}, relative error: {relative_error}')
        
if __name__ == '__main__':
    unittest.main()
