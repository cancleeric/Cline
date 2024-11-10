import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# 載入 MNIST 數據集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot=True)

# 將圖像數據展平成一維
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 將真實標籤轉換為整數形式（如果需要計算準確度）
t_train_labels = np.argmax(t_train, axis=1)
t_test_labels = np.argmax(t_test, axis=1)


# 超參數設定
iters_num = 10000  # 訓練迭代次數
train_size = x_train.shape[0]
batch_size = 200  # 小批次大小
learning_rate = 0.0005  # 學習率
epoch_size = max(train_size // batch_size, 1)  # 計算每個 epoch 包含的迭代次數


# 記錄訓練過程中的損失值和準確度
train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

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
    
    # 每個 epoch 完成後計算訓練和測試準確度
    if i % epoch_size == 0:
        # 計算訓練準確度
        train_predictions = np.argmax(network.predict(x_train), axis=1)
        train_accuracy = np.mean(train_predictions == t_train_labels)
        train_accuracy_list.append(train_accuracy)
        
        # 計算測試準確度
        test_predictions = np.argmax(network.predict(x_test), axis=1)
        test_accuracy = np.mean(test_predictions == t_test_labels)
        test_accuracy_list.append(test_accuracy)
        
        print(f"Epoch {i // epoch_size}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")


# 顯示損失變化圖
plt.plot(train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")

# 顯示準確度變化圖
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()