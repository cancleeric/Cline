# 載入mnist資料集比較更新手法
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset.mnist import load_mnist
from common.update import SGD, Momentum, AdaGrad, Adam
import tensorflow as tf
import matplotlib.pyplot as plt

def compare_optimizers(optimizers, epochs=5, batch_size=32, path='./dataset'):
    # 載入MNIST資料集，並進行標準化、展平和One-Hot編碼
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot=True, path=path)
    
    results = {}
    loss_history = {key: [] for key in optimizers.keys()}  # 新增一個字典來儲存每個優化器的損失歷史
    
    for optimizer_name, optimizer in optimizers.items():
        print(f"Training with {optimizer_name} optimizer")
        # 建立簡單的神經網路模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # 編譯模型，設定損失函數和評估指標
        model.compile(optimizer=tf.keras.optimizers.get(optimizer), loss='categorical_crossentropy', metrics=['accuracy'])
        # 訓練模型
        history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))
        # 評估模型在測試集上的表現
        loss, accuracy = model.evaluate(test_images, test_labels)
        results[optimizer_name] = history.history['val_accuracy']
        loss_history[optimizer_name] = history.history['loss']  # 儲存損失值
        print(f"{optimizer_name} optimizer - Test accuracy: {accuracy}")

    #繪製結果
    for optimizer_name, val_accuracy in results.items():
        plt.plot(val_accuracy, label=optimizer_name)
    
    plt.title('Optimizer Comparison on MNIST')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.show()

    # 等待查看第一張圖表
    # input("Press Enter to show the next figure...")



    plt.figure()  # 新增一個圖表來繪製損失歷史
    for optimizer_name, loss in loss_history.items():
        plt.plot(loss, label=optimizer_name)
    plt.title('Iterations vs Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 定義要比較的優化器
    optimizers = {
        'SGD': 'SGD',
        'Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'AdaGrad': 'Adagrad',
        'Adam': 'Adam'
    }
    # 執行優化器比較
    compare_optimizers(optimizers)
