import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from logic_functions import sigmoid, sigmoid_derivative, numerical_gradient, relu_function, cross_entropy_error

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

    def predict(self, x):
        """
        預測函式
        :param x: 輸入資料
        :return: 預測結果
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = relu_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        return y

    def loss(self, x, t):
        """
        計算損失函數
        :param x: 輸入資料
        :param t: 真實標籤
        :return: 損失值
        """
        y = self.predict(x)
        return cross_entropy_error(t, y)  # 改用交叉熵損失函數


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

    def gradient(self, x, t):
        """
        計算梯度
        :param x: 輸入資料
        :param t: 真實標籤
        :return: 梯度字典
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # Forward propagation
        a1 = np.dot(x, W1) + b1
        z1 = relu_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        
        # Backpropagation
        dy = (y - t) * sigmoid_derivative(a2)
        grads = {}
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T) * sigmoid_derivative(a1)
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        
        return grads
