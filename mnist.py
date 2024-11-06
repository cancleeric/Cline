import numpy as np
import tensorflow as tf

def load_mnist(normalize=True, flatten=False, one_hot_label=False):
    # 使用 TensorFlow 直接載入 MNIST 資料集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # 標準化
    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

    # 展平
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    # One-hot 編碼
    if one_hot_label:
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return (train_images, train_labels), (test_images, test_labels)
