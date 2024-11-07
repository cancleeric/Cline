from mnist_loader import load_mnist
import tensorflow as tf

def train_mnist_model(batch_size=32, epochs=5):
    # 載入 MNIST 資料集
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    # ...existing code...

if __name__ == "__main__":
    train_mnist_model()
