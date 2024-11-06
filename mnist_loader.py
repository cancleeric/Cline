import numpy as np
import tensorflow as tf
import ssl
import os

# 禁用 SSL 憑證驗證
ssl._create_default_https_context = ssl._create_unverified_context

def download_mnist(path='./dataset'):
    # 使用 TensorFlow 直接載入 MNIST 資料集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # 保存原始數據
    if not os.path.exists(path):
        os.makedirs(path)
    
    np.save(os.path.join(path, 'train_images_raw.npy'), train_images)
    np.save(os.path.join(path, 'train_labels_raw.npy'), train_labels)
    np.save(os.path.join(path, 'test_images_raw.npy'), test_images)
    np.save(os.path.join(path, 'test_labels_raw.npy'), test_labels)
    print(f"Raw MNIST data downloaded and saved in {path}")

def load_mnist(normalize=True, flatten=False, one_hot_label=False, path='./dataset'):
    try:
        # 使用 NumPy 載入保存的原始數據
        train_images = np.load(os.path.join(path, 'train_images_raw.npy'))
        train_labels = np.load(os.path.join(path, 'train_labels_raw.npy'))
        test_images = np.load(os.path.join(path, 'test_images_raw.npy'))
        test_labels = np.load(os.path.join(path, 'test_labels_raw.npy'))

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
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        return None

def save_mnist(train_images, train_labels, test_images, test_labels, path='./dataset'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 使用 NumPy 保存數據
    np.save(os.path.join(path, 'train_images.npy'), train_images)
    np.save(os.path.join(path, 'train_labels.npy'), train_labels)
    np.save(os.path.join(path, 'test_images.npy'), test_images)
    np.save(os.path.join(path, 'test_labels.npy'), test_labels)
    print(f"Data saved successfully in {path}")

def load_saved_mnist(path='./dataset'):
    try:
        # 使用 NumPy 載入保存的數據
        train_images = np.load(os.path.join(path, 'train_images.npy'))
        train_labels = np.load(os.path.join(path, 'train_labels.npy'))
        test_images = np.load(os.path.join(path, 'test_images.npy'))
        test_labels = np.load(os.path.join(path, 'test_labels.npy'))
        print(f"Data loaded successfully from {path}")
        return (train_images, train_labels), (test_images, test_labels)
    except FileNotFoundError:
        print(f"No saved data found in {path}. Downloading new data...")
        return None

if __name__ == "__main__":
    data = load_saved_mnist()
    if data is None:
        download_mnist()
        data = load_mnist(normalize=True, flatten=False, one_hot_label=False)
        if data:
            (train_images, train_labels), (test_images, test_labels) = data
            print(f"Training images shape: {train_images.shape}")
            print(f"Training labels shape: {train_labels.shape}")
            print(f"Test images shape: {test_images.shape}")
            print(f"Test labels shape: {test_labels.shape}")
            
            # 保存數據
            save_mnist(train_images, train_labels, test_images, test_labels)
        else:
            print("Failed to load MNIST data.")
    else:
        (train_images, train_labels), (test_images, test_labels) = data
        print(f"Training images shape: {train_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Test images shape: {test_images.shape}")
        print(f"Test labels shape: {test_labels.shape}")