
import numpy as np
from logic_functions import softmax_function, cross_entropy_error

class SimpleNet:
    def __init__(self):
        # 初始化權重，形狀為 2x3，隨機數字
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        # 預測輸出
        return np.dot(x, self.W)

    def loss(self, x, t):
        # 計算損失函數值
        z = self.predict(x)
        y = softmax_function(z)
        loss = cross_entropy_error(t, y)
        return loss