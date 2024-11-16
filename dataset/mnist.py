import numpy as np
import tensorflow as tf
import ssl
import os
import urllib.request
import gzip
import pickle

# 禁用 SSL 憑證驗證
ssl._create_default_https_context = ssl._create_unverified_context

url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site

key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
    request = urllib.request.Request(url_base + file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode='wb') as f:
        f.write(response)
    print("Done")

def download_mnist(path='./dataset'):
    for v in key_file.values():
        _download(v)

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

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=False, one_hot=False, path='./dataset'):
    # 檢查是否已經存在資料，若不存在則下載
    data = load_saved_mnist(path)
    if data is None:
        download_mnist(path)
        data = load_saved_mnist(path)
    
    if data:
        train_images, train_labels = data[0]
        test_images, test_labels = data[1]
    else:
        try:
            # 使用 NumPy 載入保存的原始數據
            train_images = np.load(os.path.join(path, 'train_images_raw.npy'))
            train_labels = np.load(os.path.join(path, 'train_labels_raw.npy'))
            test_images = np.load(os.path.join(path, 'test_images_raw.npy'))
            test_labels = np.load(os.path.join(path, 'test_labels_raw.npy'))
        except Exception as e:
            print(f"Error loading MNIST data: {e}")
            return None

    # 標準化
    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

    # 展平
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    # One-hot 編碼
    if one_hot:
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return (train_images, train_labels), (test_images, test_labels)

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
        data = load_mnist(normalize=True, flatten=False, one_hot=False)
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