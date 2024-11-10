import numpy as np
import pickle
from mnist_loader import load_mnist
from common.functions import sigmoid as sigmoid, softmax_function as softmax, sigmoid_derivative

def get_data():
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return train_images, train_labels, test_images, test_labels

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

def cross_entropy_loss(y, t):
    delta = 1e-7  # 避免log(0)的情況
    return -np.sum(t * np.log(y + delta))

def train_network(network, train_images, train_labels, epochs=10, learning_rate=0.1, batch_size=100):
    for epoch in range(epochs):
        for i in range(0, len(train_images), batch_size):
            x_batch = train_images[i:i+batch_size]
            t_batch = train_labels[i:i+batch_size]

            # Forward propagation
            a1 = np.dot(x_batch, network['W1']) + network['b1']
            z1 = sigmoid(a1)
            a2 = np.dot(z1, network['W2']) + network['b2']
            z2 = sigmoid(a2)
            a3 = np.dot(z2, network['W3']) + network['b3']
            y = softmax(a3)

            # Backpropagation
            loss = cross_entropy_loss(y, t_batch)
            dy = (y - t_batch) / batch_size
            da3 = dy
            dz2 = da3.dot(network['W3'].T) * sigmoid_derivative(z2)
            da2 = dz2
            dz1 = da2.dot(network['W2'].T) * sigmoid_derivative(z1)
            da1 = dz1

            # Weight updates
            network['W3'] -= learning_rate * z2.T.dot(da3)
            network['b3'] -= learning_rate * np.sum(da3, axis=0)
            network['W2'] -= learning_rate * z1.T.dot(da2)
            network['b2'] -= learning_rate * np.sum(da2, axis=0)
            network['W1'] -= learning_rate * x_batch.T.dot(da1)
            network['b1'] -= learning_rate * np.sum(da1, axis=0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    return network

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_data()
    network = init_network()

    # Train the network
    network = train_network(network, train_images, train_labels)

    # Evaluate the network
    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(test_images), batch_size):
        x_batch = test_images[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == np.argmax(test_labels[i:i+batch_size], axis=1))

    print(f"Accuracy: {accuracy_cnt / len(test_images)}")
