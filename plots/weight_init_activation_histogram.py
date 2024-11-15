import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid, tanh  # 匯入 tanh 函數

# 隱藏層的活性化分布
input_data = np.random.randn(1000, 100)  # 1000個資料，每個資料100個特徵
node_num = 100  # 每層100個神經元
hidden_layer_size = 5  # 5層隱藏層
activations = {}  # 儲存活性化結果

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 權重初始化
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    a = np.dot(x, w)

    # 活性化函數
    z = sigmoid(a)  # 使用 common.functions 中的 tanh 函數
    # z = tanh(a)  # 使用 common.functions 中的 tanh 函數

    activations[i] = z

# 繪製活性化分布圖
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()