import numpy as np
from collections import OrderedDict
from common.layers import Affine, Relu, SoftmaxWithLoss, Sigmoid, BatchNorm

class MultiLayerNetBN:
    """
    具有 Batch Normalization 的多層神經網路
    """
    def __init__(self, input_size, hidden_sizes, output_size, 
                 activation='relu', weight_init_std='he', 
                 weight_decay_lambda=0, use_batchnorm=True):
        """
        初始化網路
        Parameters
        ----------
        input_size : 輸入大小（MNIST 的情況下為 784）
        hidden_sizes : 隱藏層的神經元數量列表（e.g. [100, 100, 100]）
        output_size : 輸出大小（MNIST 的情況下為 10）
        activation : 'relu' or 'sigmoid'
        weight_init_std : 權重初始化方式
                        'he'：適用於 relu
                        'xavier'：適用於 sigmoid
                        數值：指定標準差的高斯分佈
        weight_decay_lambda : L2 正則化係數
        use_batchnorm : 是否使用 Batch Normalization
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_layer_num = len(hidden_sizes)
        self.use_batchnorm = use_batchnorm
        self.weight_decay_lambda = weight_decay_lambda
        self.activation = activation

        # 初始化權重
        self.params = {}
        self.__init_weight(weight_init_std)

        # 生成層
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 2):
            self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], 
                                                self.params[f'b{idx}'])
            
            # 批次正規化（除了最後一層）
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                self.params[f'gamma{idx}'] = np.ones(hidden_sizes[idx-1])
                self.params[f'beta{idx}'] = np.zeros(hidden_sizes[idx-1])
                self.layers[f'BatchNorm{idx}'] = BatchNorm(
                    self.params[f'gamma{idx}'], 
                    self.params[f'beta{idx}'])
            
            # 激活函數（除了最後一層）
            if idx != self.hidden_layer_num + 1:
                self.layers[f'Activation{idx}'] = activation_layer[activation]()

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        權重初始化
        """
        all_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for idx in range(1, len(all_sizes)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_sizes[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_sizes[idx - 1])
            
            self.params[f'W{idx}'] = scale * np.random.randn(all_sizes[idx-1], all_sizes[idx])
            self.params[f'b{idx}'] = np.zeros(all_sizes[idx])

    def predict(self, x, train_flg=False):
        """
        進行預測
        """
        for key, layer in self.layers.items():
            if "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        """
        計算損失函數值
        """
        y = self.predict(x, train_flg)

        # 權重衰減
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params[f'W{idx}']
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        """
        計算準確率
        """
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        """
        計算梯度
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + \
                              self.weight_decay_lambda * self.params[f'W{idx}']
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads[f'gamma{idx}'] = self.layers[f'BatchNorm{idx}'].dgamma
                grads[f'beta{idx}'] = self.layers[f'BatchNorm{idx}'].dbeta

        return grads 