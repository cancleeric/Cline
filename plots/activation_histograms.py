
import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.functions import sigmoid, relu
from common.layers import Affine, Relu, SoftmaxWithLoss, Sigmoid
from common.multi_layer_net import MultiLayerNet  # 使用 MultiLayerNet

# 測試參數
input_data = np.random.randn(1000, 100)  # 1000個資料，每個資料100個特徵
node_num = 100  # 每層100個神經元
hidden_layer_size = 5  # 隱藏層數量
output_size = 10  # 假設輸出層大小為10
activation_functions = {
    "sigmoid": "sigmoid",
    "relu": "relu"
}

plt.figure(figsize=(10, 10))  # 調整圖形大小

for act_idx, (act_name, activation_function) in enumerate(activation_functions.items()):
    # 使用 MultiLayerNet 初始化網路
    network = MultiLayerNet(
        input_size=node_num,
        hidden_sizes=[node_num] * hidden_layer_size,
        output_size=output_size,
        activation=activation_function
    )

    activations = {}  # 儲存各層激活結果
    x = input_data

    for i in range(hidden_layer_size):
        # 前向傳播
        x = network.layers[f'Affine{i+1}'].forward(x)
        x = network.layers[f'Activation{i+1}'].forward(x)
        activations[i] = x

    # 繪製每層激活值的直方圖
    for i, a in activations.items():
        plt.subplot(len(activation_functions), hidden_layer_size, act_idx * hidden_layer_size + i + 1)
        plt.title(f"Layer {i + 1}")
        if i == 0:
            plt.ylabel(f"{act_name}", rotation=0, labelpad=40)
        plt.hist(a.flatten(), 30, range=(0, 1))
        plt.ylim(0, 7000)

plt.tight_layout()
plt.show()