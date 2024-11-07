import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import sys
sys.path.append('/Users/apple/Desktop/Cline')
from logic_functions import gradient_function

# 定義一個二次函數
def function_2(x):
    return x[0]**2 + x[1]**2

# 計算梯度
x = np.array([3.0, 4.0])
grad = gradient_function(function_2, x)

# 作圖
plt.figure()
plt.quiver(x[0], x[1], -grad[0], -grad[1], angles='xy', scale_units='xy', scale=1, color='r')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.title('Gradient Descent')
plt.show()

