import numpy as np
import tensorflow as tf
import ssl
import os

# 禁用 SSL 憑證驗證
ssl._create_default_https_context = ssl._create_unverified_context

def load_mnist():
    try:
        # 使用 TensorFlow 直接載入 MNIST 資料集
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # 轉換為 NumPy 數組並進行標準化
        train_images = np.array(train_images, dtype=np.float32) / 255.0
        train_labels = np.array(train_labels, dtype=np.int32)
        test_images = np.array(test_images, dtype=np.float32) / 255.0
        test_labels = np.array(test_labels, dtype=np.int32)

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
        data = load_mnist()
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