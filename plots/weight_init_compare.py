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
from common.update import SGD

from common.util import smooth_curve


# 載入 MNIST 資料集
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot=True)

# # 取出 2000 筆資料
# train_images = train_images[:2000]
# train_labels = train_labels[:2000]

# 設定網路參數
input_size = 784
hidden_sizes = [100, 100, 100, 100, 100]  # 5 層隱藏層，每層 100 個神經元
output_size = 10
batch_size = 128
max_iterations = 2000

activation_functions = {
    "sigmoid": "sigmoid",
    "relu": "relu"
}

# 設定優化器
optimizer = SGD(learning_rate=0.001)

# 初始化網路和記錄損失
networks = {}
train_loss = {}
for act_name, activation in activation_functions.items():
    networks[act_name] = MultiLayerNet(input_size, hidden_sizes, output_size, activation=activation)
    train_loss[act_name] = []

# 訓練過程
for i in range(max_iterations):
    batch_mask = np.random.choice(train_images.shape[0], batch_size)
    x_batch = train_images[batch_mask]
    t_batch = train_labels[batch_mask]
    
    for act_name in activation_functions.keys():
        grads = networks[act_name].gradient(x_batch, t_batch)
        optimizer.update(networks[act_name].params, grads)
    
        loss = networks[act_name].loss(x_batch, t_batch)
        train_loss[act_name].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for act_name in activation_functions.keys():
            loss = networks[act_name].loss(x_batch, t_batch)
            print(act_name + ":" + str(loss))

# 繪製損失變化圖
markers = {'sigmoid': 'o', 'relu': 's'}
x = np.arange(max_iterations)
for act_name in activation_functions.keys():
    plt.plot(x, smooth_curve(train_loss[act_name]), marker=markers[act_name], markevery=100, label=act_name)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
