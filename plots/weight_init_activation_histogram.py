import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid, relu  # 匯入 relu 函數

# 隱藏層的活性化分布
input_data = np.random.randn(1000, 100)  # 1000個資料，每個資料100個特徵
node_num = 100  # 每層100個神經元
hidden_layer_size = 5  # 5層隱藏層

# 權重初始化方法
initialization_methods = {
    "0.01": lambda: np.random.randn(node_num, node_num) * 0.01,
    "Xavier": lambda: np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num),
    "He": lambda: np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
}

activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu
}

plt.figure(figsize=(15, 10))  # 調整整體圖形大小

for act_idx, (act_name, activation_function) in enumerate(activation_functions.items()):
    for idx, (method_name, weight_init) in enumerate(initialization_methods.items()):
        activations = {}  # 儲存活性化結果
        x = input_data

        for i in range(hidden_layer_size):
            if i != 0:
                x = activations[i-1]

            # 權重初始化
            w = weight_init()

            a = np.dot(x, w)

            # 活性化函數
            z = activation_function(a)

            activations[i] = z

        # 繪製活性化分布圖
        for i, a in activations.items():
            plt.subplot(len(activation_functions) * len(initialization_methods), hidden_layer_size, act_idx * len(initialization_methods) * hidden_layer_size + idx * hidden_layer_size + i + 1)
            if i == 0:
                plt.ylabel(f"{act_name} - {method_name}", rotation=0, labelpad=40)
            plt.title(f"Layer {i+1}")
            plt.xlim(0.1, 1)
            plt.ylim(0, 7000)
            plt.hist(a.flatten(), 30, range=(0,1))

plt.tight_layout()
plt.show()