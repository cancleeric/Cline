import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from logic_functions import gradient_descent

# 目標函數
f = lambda x: x[0]**2 + x[1]**2

# 初始點
init_x = np.array([3.0, 4.0])

# 使用梯度下降法
learning_rate = 0.1
num_iterations = 100
x, x_history = gradient_descent(f, init_x, learning_rate, num_iterations)

# 將歷史 x 值轉換為 numpy array 以便作圖
x_history = np.array(x_history)

# 作圖
plt.figure(figsize=(10, 6))
plt.plot(x_history[:, 0], x_history[:, 1], 'o-', color='blue')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gradient Descent Optimization Path')
plt.grid(True)
plt.show()
