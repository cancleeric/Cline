import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import sklearn

def print_versions():
    print("TensorFlow:", tf.__version__)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Scikit-learn:", sklearn.__version__)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.W = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(np.insert(x, 0, 1))
        return self.activation_fn(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            for xi, target in zip(X, d):
                update = self.learning_rate * (target - self.predict(xi))
                self.W[1:] += update * xi
                self.W[0] += update

def main():
    print_versions()

    # Example dataset: OR logic gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    d = np.array([0, 1, 1, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, d)

    # Test the perceptron
    for x in X:
        print(f"Input: {x}, Predicted Output: {perceptron.predict(x)}")

if __name__ == "__main__":
    main()
