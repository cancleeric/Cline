import numpy as np
import pickle
from mnist_loader import load_mnist
from logic_functions import sigmoid_function as sigmoid, softmax_function as softmax

def get_data():
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return test_images, test_labels

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

if __name__ == "__main__":
    test_images, test_labels = get_data()
    network = init_network()

    batch_size = 100  # 批次大小
    accuracy_cnt = 0

    for i in range(0, len(test_images), batch_size):
        x_batch = test_images[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)  # 獲取概率最高的元素的索引
        accuracy_cnt += np.sum(p == test_labels[i:i+batch_size])

    print(f"Accuracy: {accuracy_cnt / len(test_images)}")
