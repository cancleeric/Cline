import numpy as np

class Perceptron:
    def __init__(self, input_size, weights=None, bias=None):
        self.input_size = input_size
        if weights is None:
            self.weights = np.random.rand(input_size)
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.random.rand(1)[0]
        else:
            self.bias = bias

    def calculate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        if weighted_sum > 0:
            return 1
        else:
            return 0

# Example usage (now with specific weights and bias for AND gate)
input1 = np.array([0, 0])
input2 = np.array([0, 1])
input3 = np.array([1, 0])
input4 = np.array([1, 1])

# Weights and bias for an AND gate (adjust as needed)
weights = np.array([0.5, 0.5])
bias = -0.7

perceptron = Perceptron(2, weights, bias)

print("AND Gate Implementation:")
print(f"Input: {input1}, Predicted output: {perceptron.calculate(input1)}")  # Output: 0
print(f"Input: {input2}, Predicted output: {perceptron.calculate(input2)}")  # Output: 0
print(f"Input: {input3}, Predicted output: {perceptron.calculate(input3)}")  # Output: 0
print(f"Input: {input4}, Predicted output: {perceptron.calculate(input4)}")  # Output: 1
