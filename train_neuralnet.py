import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# 載入 MNIST 數據集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot=True)

# 將圖像數據展平成一維
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


# 超參數設定
iters_num = 10000  # 訓練迭代次數
train_size = x_train.shape[0]
batch_size = 100  # 小批次大小
learning_rate = 0.1  # 學習率

# 記錄訓練過程中的損失值
train_loss_list = []

# 初始化兩層神經網絡
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 訓練過程
for i in range(iters_num):
    # 隨機抽取小批次
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 計算梯度
    grad = network.gradient(x_batch, t_batch)
    
    # 更新參數
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 記錄學習過程中的損失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 每 1000 次迭代輸出一次損失值
    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss}")

# 顯示損失變化圖
plt.plot(train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()