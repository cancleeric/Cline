import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import relu

x = np.linspace(-10, 10, 400)
y = relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU Function')
plt.title('ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
