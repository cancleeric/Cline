import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from common.functions import sigmoid, sigmoid_derivative, numerical_gradient, relu_function, cross_entropy_error
from common.layers import Affine, Relu, SoftmaxWithLoss  # 導入所需的層
from common.update import SGD  # 更新這行

class TwoLayerNet:
    """
    兩層神經網路類別
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化網路參數
        :param input_size: 輸入層大小
        :param hidden_size: 隱藏層大小
        :param output_size: 輸出層大小
        :param weight_init_std: 權重初始化的標準差 (預設為 0.01)
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 設定網路層
        self.layers = {}
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        預測函式
        :param x: 輸入資料
        :return: 預測結果
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        計算損失函數
        :param x: 輸入資料
        :param t: 真實標籤
        :return: 損失值
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        """
        計算梯度
        :param x: 輸入資料
        :param t: 真實標籤
        :return: 梯度字典
        """
        # 前向傳播
        self.loss(x, t)

        # 反向傳播
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 計算梯度
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def numerical_gradient(self, x, t):
        """
        計算數值梯度
        :param x: 輸入資料
        :param t: 真實標籤
        :return: 梯度字典
        """
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

