import numpy as np
import tensorflow as tf
import pickle

def train_network():
    # 載入 MNIST 資料集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 784).astype(np.float32) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 784).astype(np.float32) / 255.0
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # 建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='sigmoid', input_shape=(784,)),
        tf.keras.layers.Dense(100, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 編譯模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 訓練模型
    model.fit(train_images, train_labels, epochs=5, batch_size=100, validation_split=0.2)

    # 評估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

    # 保存權重
    weights = {
        'W1': model.layers[0].get_weights()[0],
        'b1': model.layers[0].get_weights()[1],
        'W2': model.layers[1].get_weights()[0],
        'b2': model.layers[1].get_weights()[1],
        'W3': model.layers[2].get_weights()[0],
        'b3': model.layers[2].get_weights()[1]
    }

    with open("sample_weight.pkl", "wb") as f:
        pickle.dump(weights, f)
    print("Weights saved to sample_weight.pkl")

if __name__ == "__main__":
    train_network()
