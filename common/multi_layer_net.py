import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from collections import OrderedDict
from common.functions import sigmoid, relu, softmax, cross_entropy_error, numerical_gradient
from common.layers import Affine, Relu, SoftmaxWithLoss, Sigmoid

class MultiLayerNet:
    """
    多層神經網路類別
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', weight_init_std=0.01):
        """
        初始化網路參數
        :param input_size: 輸入層大小
        :param hidden_sizes: 隱藏層大小列表
        :param output_size: 輸出層大小
        :param activation: 活性化函數類型 ('relu' 或 'sigmoid')
        :param weight_init_std: 權重初始化的標準差 (預設為 0.01)
        """
        self.params = {}
        self.layers = {}
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # 初始化權重和偏置
        all_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(all_sizes)):
            if activation == 'relu':
                self.params[f'W{i}'] = np.random.randn(all_sizes[i-1], all_sizes[i]) * np.sqrt(2.0 / all_sizes[i-1])
            elif activation == 'sigmoid':
                self.params[f'W{i}'] = np.random.randn(all_sizes[i-1], all_sizes[i]) * np.sqrt(1.0 / all_sizes[i-1])
            else:
                self.params[f'W{i}'] = weight_init_std * np.random.randn(all_sizes[i-1], all_sizes[i])
            self.params[f'b{i}'] = np.zeros(all_sizes[i])

        # 設定網路層
        for i in range(1, len(all_sizes) - 1):
            self.layers[f'Affine{i}'] = Affine(self.params[f'W{i}'], self.params[f'b{i}'])
            if activation == 'relu':
                self.layers[f'Activation{i}'] = Relu()
            elif activation == 'sigmoid':
                self.layers[f'Activation{i}'] = Sigmoid()
        self.layers[f'Affine{len(all_sizes) - 1}'] = Affine(self.params[f'W{len(all_sizes) - 1}'], self.params[f'b{len(all_sizes) - 1}'])
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
        for i in range(1, len(self.hidden_sizes) + 2):
            grads[f'W{i}'] = self.layers[f'Affine{i}'].dW
            grads[f'b{i}'] = self.layers[f'Affine{i}'].db

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
        for i in range(1, len(self.hidden_sizes) + 2):
            grads[f'W{i}'] = numerical_gradient(loss_W, self.params[f'W{i}'])
            grads[f'b{i}'] = numerical_gradient(loss_W, self.params[f'b{i}'])
        
        return grads
    
