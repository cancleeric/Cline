import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from dataset.mnist_deep import load_mnist
from common.multi_layer_net_bn import MultiLayerNetBN
from sklearn.model_selection import train_test_split
from common.update import SGD
import matplotlib.pyplot as plt

# 載入 MNIST 資料
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 資料拆分為訓練集和驗證集
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# 網路參數
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
weight_decay_lambda = 0.001

# 創建兩個網路進行比較：一個使用 BatchNorm，一個不使用
networks = {
    'with_bn': MultiLayerNetBN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation='relu',
        weight_decay_lambda=weight_decay_lambda,
        use_batchnorm=True
    ),
    'without_bn': MultiLayerNetBN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation='relu',
        weight_decay_lambda=weight_decay_lambda,
        use_batchnorm=False
    )
}

optimizers = {
    'with_bn': SGD(learning_rate=0.01),
    'without_bn': SGD(learning_rate=0.01)
}

# 訓練參數
epochs = 200
batch_size = 128
train_sample_size = 2000
train_loss = {'with_bn': [], 'without_bn': []}
val_loss = {'with_bn': [], 'without_bn': []}

# 訓練迴圈
for epoch in range(epochs):
    # 隨機選擇訓練樣本
    train_indices = np.random.choice(x_train.shape[0], train_sample_size, replace=False)
    x_train_sample = x_train[train_indices]
    y_train_sample = y_train[train_indices]
    
    # 打亂訓練數據
    permutation = np.random.permutation(train_sample_size)
    x_train_sample = x_train_sample[permutation]
    y_train_sample = y_train_sample[permutation]
    
    # 對每個網路進行訓練
    for net_type in networks.keys():
        loss_sum = 0
        for i in range(0, train_sample_size, batch_size):
            x_batch = x_train_sample[i:i + batch_size]
            y_batch = y_train_sample[i:i + batch_size]
            
            # 計算梯度
            grads = networks[net_type].gradient(x_batch, y_batch)
            
            # 更新參數
            optimizers[net_type].update(networks[net_type].params, grads)
            
            # 計算損失
            loss = networks[net_type].loss(x_batch, y_batch)
            loss_sum += loss
            train_loss[net_type].append(loss)
        
        # 計算驗證集損失
        val_loss[net_type].append(networks[net_type].loss(x_val, y_val))
        
        print(f"Epoch {epoch + 1}/{epochs} - {net_type} - Training Loss: {loss_sum:.4f}, "
              f"Validation Loss: {val_loss[net_type][-1]:.4f}")

# 繪製訓練損失比較圖
plt.figure(figsize=(12, 5))

# 訓練損失
plt.subplot(1, 2, 1)
for net_type in networks.keys():
    plt.plot(train_loss[net_type], label=f'{net_type}')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()

# 驗證損失
plt.subplot(1, 2, 2)
for net_type in networks.keys():
    plt.plot(val_loss[net_type], label=f'{net_type}')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# 計算並比較最終準確率
def calculate_accuracy(net, x, t):
    y = net.predict(x, train_flg=False)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

print("\nFinal Accuracies:")
for net_type, net in networks.items():
    train_acc = calculate_accuracy(net, x_train, y_train)
    val_acc = calculate_accuracy(net, x_val, y_val)
    test_acc = calculate_accuracy(net, test_images, test_labels)
    
    print(f"\n{net_type}:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}") 