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

# # 計算梯度
# x = np.array([5.0, 3.6])
# grad = gradient_function(function_2, x)

# # 檢查梯度
# print("Gradient at (5.0, 3.6):", grad)
# # 作圖
# plt.figure(figsize=(8, 8))
# plt.quiver(x[0], x[1], -grad[0], -grad[1], angles='xy', scale_units='xy', scale=5, color='r')
# plt.scatter(x[0], x[1], color='b')  # 標記起始點
# plt.xlim(-6, 6)
# plt.ylim(-6, 6)
# plt.xlabel('x0')
# plt.ylabel('x1')
# plt.grid(True)
# plt.title('Gradient Descent Visualization')
# plt.show()

# 設置網格點
x_range = np.arange(-2, 2.25, 0.25)
y_range = np.arange(-2, 2.25, 0.25)
X, Y = np.meshgrid(x_range, y_range)

# 計算網格上每個點的梯度
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        grad = gradient_function(function_2, point)
        U[i, j] = -grad[0]  # 負梯度，表示下降方向
        V[i, j] = -grad[1]  # 負梯度，表示下降方向

# 作圖
plt.figure(figsize=(10, 10))
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=20, color='r')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid(True)
plt.title('Gradient Descent Vector Field')
plt.show()