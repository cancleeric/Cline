# Description: Implement the layers of the neural network
import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        # 增加數值穩定項
        # dx += 1e-7
        return dx

class BatchNorm:
    """
    批次正規化層
    """
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma        # 縮放參數
        self.beta = beta          # 平移參數
        self.momentum = momentum  # 移動平均的動量
        
        # 測試時使用的移動平均
        self.running_mean = running_mean
        self.running_var = running_var
        
        # 反向傳播時使用的中間數據
        self.input_shape = None   # 輸入形狀
        self.batch_size = None    # 批次大小
        self.x_centered = None    # 中心化後的數據
        self.xn = None           # 標準化後的數據
        self.std = None          # 標準差
        self.dgamma = None       # gamma的梯度
        self.dbeta = None        # beta的梯度
        self.D = None  # 添加這個來保存展平後的特徵維度

    def forward(self, x, train_flg=True):
        """
        前向傳播
        x: (N, C) 或 (N, C, H, W)
        """
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)  # 將 4D 輸入展平為 2D
            self.D = C * H * W    # 保存展平後的特徵維度
            
            # 確保 gamma 和 beta 的形狀正確
            if isinstance(self.gamma, (int, float)):
                self.gamma = np.ones(self.D)
            elif self.gamma.ndim == 1 and len(self.gamma) == C:
                self.gamma = np.repeat(self.gamma, H*W)
                
            if isinstance(self.beta, (int, float)):
                self.beta = np.zeros(self.D)
            elif self.beta.ndim == 1 and len(self.beta) == C:
                self.beta = np.repeat(self.beta, H*W)
        else:
            self.D = x.shape[1]

        # 初始化 running_mean 和 running_var
        if self.running_mean is None:
            self.running_mean = np.zeros(self.D)
            self.running_var = np.zeros(self.D)
        elif self.running_mean.shape[0] != self.D:
            # 如果形狀不匹配，重新初始化
            if x.ndim != 2:
                N, C, H, W = self.input_shape
                if len(self.running_mean) == C:
                    self.running_mean = np.repeat(self.running_mean, H*W)
                    self.running_var = np.repeat(self.running_var, H*W)
                else:
                    self.running_mean = np.zeros(self.D)
                    self.running_var = np.zeros(self.D)

        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if train_flg:
            mu = x.mean(axis=0)
            self.x_centered = x - mu
            var = np.mean(self.x_centered**2, axis=0)
            self.std = np.sqrt(var + 1e-7)
            self.xn = self.x_centered / self.std
            
            self.batch_size = x.shape[0]
            
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            self.x_centered = x - self.running_mean
            self.xn = self.x_centered / np.sqrt(self.running_var + 1e-7)
        
        out = self.gamma * self.xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        
        # 如果是 4D 輸入，需要將 gamma 和 beta 的梯度調整回原始形狀
        if self.input_shape[1] != self.dgamma.shape[0]:
            N, C, H, W = self.input_shape
            self.dgamma = self.dgamma.reshape(C, H, W).sum(axis=(1, 2))
            self.dbeta = self.dbeta.reshape(C, H, W).sum(axis=(1, 2))
            
        return dx

    def __backward(self, dout):
        self.dbeta = dout.sum(axis=0)
        self.dgamma = np.sum(self.xn * dout, axis=0)
        
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.x_centered) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.x_centered * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        return dx
