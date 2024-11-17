import sys
import os
sys.path.append(os.pardir)

import numpy as np
from common.layers import BatchNorm
import unittest

class TestBatchNorm(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.input_size = 4
        self.bn = BatchNorm(gamma=1.0, beta=0.0)

    def test_initialization(self):
        """測試初始化參數"""
        self.assertEqual(self.bn.gamma, 1.0)
        self.assertEqual(self.bn.beta, 0.0)
        self.assertEqual(self.bn.momentum, 0.9)
        self.assertIsNone(self.bn.running_mean)
        self.assertIsNone(self.bn.running_var)

    def test_forward_train(self):
        """測試訓練模式的前向傳播"""
        x = np.random.randn(self.batch_size, self.input_size)
        out = self.bn.forward(x, train_flg=True)
        
        self.assertEqual(out.shape, x.shape)
        mean = np.mean(out, axis=0)
        var = np.var(out, axis=0)
        np.testing.assert_almost_equal(mean, self.bn.beta, decimal=5)
        np.testing.assert_almost_equal(var, self.bn.gamma**2, decimal=5)

    def test_forward_test(self):
        """測試測試模式的前向傳播"""
        x_train = np.random.randn(self.batch_size, self.input_size)
        self.bn.forward(x_train, train_flg=True)
        
        x_test = np.random.randn(self.batch_size, self.input_size)
        out = self.bn.forward(x_test, train_flg=False)
        
        self.assertEqual(out.shape, x_test.shape)

    def test_backward(self):
        """測試反向傳播"""
        x = np.random.randn(self.batch_size, self.input_size)
        out = self.bn.forward(x, train_flg=True)
        dout = np.random.randn(*out.shape)
        dx = self.bn.backward(dout)
        
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(self.bn.dgamma.shape, (self.input_size,))
        self.assertEqual(self.bn.dbeta.shape, (self.input_size,))

    def test_different_input_shapes(self):
        """測試不同輸入形狀"""
        # 測試 2D 輸入
        bn_2d = BatchNorm(gamma=1.0, beta=0.0)
        x_2d = np.random.randn(self.batch_size, self.input_size)
        out_2d = bn_2d.forward(x_2d, train_flg=True)
        self.assertEqual(out_2d.shape, x_2d.shape)
        
        # 測試 4D 輸入 (CNN)
        channels = 3
        height = 4
        width = 4
        bn_4d = BatchNorm(gamma=np.ones(channels), beta=np.zeros(channels))
        x_4d = np.random.randn(self.batch_size, channels, height, width)
        out_4d = bn_4d.forward(x_4d, train_flg=True)
        self.assertEqual(out_4d.shape, x_4d.shape)

    def test_momentum_update(self):
        """測試移動平均的更新"""
        x1 = np.random.randn(self.batch_size, self.input_size)
        x2 = np.random.randn(self.batch_size, self.input_size)
        
        self.bn.forward(x1, train_flg=True)
        running_mean1 = self.bn.running_mean.copy()
        running_var1 = self.bn.running_var.copy()
        
        self.bn.forward(x2, train_flg=True)
        running_mean2 = self.bn.running_mean.copy()
        running_var2 = self.bn.running_var.copy()
        
        self.assertFalse(np.array_equal(running_mean1, running_mean2))
        self.assertFalse(np.array_equal(running_var1, running_var2))

if __name__ == '__main__':
    unittest.main()