import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet

# 載入 MNIST 資料集
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot=True)

# 取出 2000 筆資料
train_images = train_images[:2000]
train_labels = train_labels[:2000]

# 設定網路參數
input_size = 784
hidden_sizes = [100, 100, 100, 100, 100]  # 5 層隱藏層，每層 100 個神經元
output_size = 10

activation_functions = {
    "sigmoid": "sigmoid",
    "relu": "relu"
}

plt.figure(figsize=(15, 10))  # 調整整體圖形大小

for act_idx, (act_name, activation) in enumerate(activation_functions.items()):
    # 初始化網路
    network = MultiLayerNet(input_size, hidden_sizes, output_size, activation=activation)

    # 前向傳播
    activations = {}
    x = train_images

    for i in range(len(hidden_sizes)):
        x = network.layers[f'Affine{i+1}'].forward(x)
        x = network.layers[f'Activation{i+1}'].forward(x)
        activations[i] = x

    # 繪製活性化分布圖
    for i, a in activations.items():
        plt.subplot(len(activation_functions), len(hidden_sizes), act_idx * len(hidden_sizes) + i + 1)
        if i == 0:
            plt.ylabel(f"{act_name}", rotation=0, labelpad=40)
        plt.title(f"Layer {i+1}")
        plt.xlim(0.1, 1)
        plt.ylim(0, 7000)
        plt.hist(a.flatten(), 30, range=(0,1))

plt.tight_layout()
plt.show()