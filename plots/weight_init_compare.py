import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from dataset.mnist_deep import load_mnist
from common.multi_layer_net import MultiLayerNet
from sklearn.model_selection import train_test_split
from common.update import SGD
import matplotlib.pyplot as plt

# MNIST 資料載入
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 資料拆分為訓練集和驗證集
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# 定義網路參數
input_size = 784  # MNIST 每張影像為 28x28
hidden_sizes = [128, 64]  # 隱藏層大小
output_size = 10  # 10 個分類 (0-9)
activation = 'relu'
weight_decay_lambda = 0.001  # 權重衰減係數

# 初始化網路
net = MultiLayerNet(input_size, hidden_sizes, output_size, activation=activation, weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(learning_rate=0.01)

# 訓練參數
learning_rate = 0.01
epochs = 200
batch_size = 128
train_sample_size = 2000  # 每次訓練迴圈中使用的訓練資料樣本數

# 訓練迴圈
train_loss = {}
train_loss[activation] = []
for epoch in range(epochs):
    # 隨機選擇部分訓練資料
    train_indices = np.random.choice(x_train.shape[0], train_sample_size, replace=False)
    x_train_sample = x_train[train_indices]
    y_train_sample = y_train[train_indices]
    
    # 每次打亂訓練數據
    permutation = np.random.permutation(x_train_sample.shape[0])
    x_train_sample = x_train_sample[permutation]
    y_train_sample = y_train_sample[permutation]
    
    loss_sum = 0
    for i in range(0, x_train_sample.shape[0], batch_size):
        x_batch = x_train_sample[i:i + batch_size]
        y_batch = y_train_sample[i:i + batch_size]
        
        # 計算梯度
        grads = net.gradient(x_batch, y_batch)
        
        # 使用 SGD 更新參數
        optimizer.update(net.params, grads)
        
        # 計算損失
        loss = net.loss(x_batch, y_batch)
        loss_sum += loss
        train_loss[activation].append(loss)
    
    # 計算驗證集損失
    val_loss = net.loss(x_val, y_val)
    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {loss_sum:.4f}, Validation Loss: {val_loss:.4f}")

# 繪製訓練損失圖表
plt.plot(train_loss[activation], label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.show()

# 測試準確率
def calculate_accuracy(x, t):
    y = net.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

train_acc = calculate_accuracy(x_train, y_train)
val_acc = calculate_accuracy(x_val, y_val)
test_acc = calculate_accuracy(test_images, test_labels)

print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")