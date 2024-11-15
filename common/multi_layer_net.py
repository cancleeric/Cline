import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from collections import OrderedDict
from common.functions import sigmoid, relu, softmax, cross_entropy_error, numerical_gradient
from common.layers import Affine, Relu, SoftmaxWithLoss, Sigmoid

class MultiLayerNetE:
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
    


class MultiLayerNet:
    """全連接多層神經網路

    Parameters
    ----------
    input_size : 輸入大小（例如 MNIST 的情況下為 784）
    hidden_size_list : 隱藏層的神經元數量列表（例如 [100, 100, 100]）
    output_size : 輸出大小（例如 MNIST 的情況下為 10）
    activation : 'relu' 或 'sigmoid'
    weight_init_std : 指定權重的標準差（例如 0.01）
        指定 'relu' 或 'he' 的情況下設置為「He 的初始值」
        指定 'sigmoid' 或 'xavier' 的情況下設置為「Xavier 的初始值」
    weight_decay_lambda : Weight Decay（L2 範數）的強度
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化權重
        self.__init_weight(weight_init_std)

        # 生成層
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """設置權重的初始值

        Parameters
        ----------
        weight_init_std : 指定權重的標準差（例如 0.01）
            指定 'relu' 或 'he' 的情況下設置為「He 的初始值」
            指定 'sigmoid' 或 'xavier' 的情況下設置為「Xavier 的初始值」
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用 ReLU 時推薦的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用 sigmoid 時推薦的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """計算損失函數

        Parameters
        ----------
        x : 輸入資料
        t : 標籤

        Returns
        -------
        損失函數的值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """計算梯度（數值微分）

        Parameters
        ----------
        x : 輸入資料
        t : 標籤

        Returns
        -------
        包含各層梯度的字典變數
            grads['W1']、grads['W2']、... 是各層的權重
            grads['b1']、grads['b2']、... 是各層的偏置
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """計算梯度（誤差反向傳播法）

        Parameters
        ----------
        x : 輸入資料
        t : 標籤

        Returns
        -------
        包含各層梯度的字典變數
            grads['W1']、grads['W2']、... 是各層的權重
            grads['b1']、grads['b2']、... 是各層的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads