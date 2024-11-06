import numpy as np
import matplotlib.pyplot as plt

def softmax_function(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 生成數據
x = np.linspace(-2.0, 2.0, num=100)
y = softmax_function(x)

# 繪製圖表
plt.plot(x, y, label='Softmax Function')
plt.title('Softmax Function Plot')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
