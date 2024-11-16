import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net import MultiLayerNet  # 使用 MultiLayerNet
from dataset.mnist import load_mnist  # 匯入 load_mnist 函數
from common.update import SGD  # 隨機梯度下降法

# 載入 MNIST 資料集
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot=True)

# 訓練參數設置
input_size = 784  # MNIST 的特徵數量
node_num = 100  # 每層100個神經元
hidden_layer_size = 5  # 隱藏層數量
output_size = 10  # 輸出層大小（10個類別）
activation_functions = {
    "sigmoid": "sigmoid",
    "relu": "relu"
}
learning_rate = 0.01
batch_size = 128
iterations = 2000

# 初始化圖形
plt.figure(figsize=(10, 5))

# 針對每種激活函數訓練
for act_idx, (act_name, activation_function) in enumerate(activation_functions.items()):
    # 初始化神經網路
    network = MultiLayerNet(
        input_size=input_size,
        hidden_sizes=[node_num] * hidden_layer_size,
        output_size=output_size,
        activation=activation_function
    )

    optimizer = SGD(learning_rate=learning_rate)
    train_size = train_images.shape[0]

    # 損失函數記錄
    loss_list = []

    # 訓練模型
    for iteration in range(iterations):
        # 隨機選擇一批數據進行訓練
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_images[batch_mask]
        t_batch = train_labels[batch_mask]

        # 計算梯度並更新權重
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        # 記錄損失函數值
        loss = network.loss(x_batch, t_batch)
        loss_list.append(loss)

        # 打印訓練進度
        if iteration % 200 == 0 or iteration == iterations - 1:
            print(f"[{act_name}] Iteration {iteration}/{iterations} - Loss: {loss:.4f}")

    # 繪製損失函數曲線
    plt.plot(range(iterations), loss_list, label=f"Activation: {act_name}")

# 圖形設置
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Function during Training")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()