import numpy as np
import pickle
from mnist_loader import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

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

    accuracy_cnt = 0
    for i in range(len(test_images)):
        y = predict(network, test_images[i])
        p = np.argmax(y)  # 獲取概率最高的元素的索引
        if p == test_labels[i]:
            accuracy_cnt += 1

    print(f"Accuracy: {accuracy_cnt / len(test_images)}")
