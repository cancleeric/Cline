import numpy as np

from logic_functions import step_function

class ANDGate:
    def __init__(self, weights=None, bias=None):
        if weights is None:
            self.weights = np.array([0.5, 0.5])
        else:
            self.weights = weights
        if bias is None:
            self.bias = -0.7
        else:
            self.bias = bias

    def calculate(self, input1, input2):
        inputs = np.array([input1, input2])
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return step_function(weighted_sum)

class NANDGate:
    def __init__(self, weights=None, bias=None):
        if weights is None:
            self.weights = np.array([-0.5, -0.5])
        else:
            self.weights = weights
        if bias is None:
            self.bias = 0.7
        else:
            self.bias = bias

    def calculate(self, input1, input2):
        inputs = np.array([input1, input2])
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return step_function(weighted_sum)

class ORGate:
    def __init__(self, weights=None, bias=None):
        if weights is None:
            self.weights = np.array([1.0, 1.0])
        else:
            self.weights = weights
        if bias is None:
            self.bias = -0.5
        else:
            self.bias = bias

    def calculate(self, input1, input2):
        inputs = np.array([input1, input2])
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return step_function(weighted_sum)

class XORGate:
    def __init__(self):
        self.and_gate = ANDGate()
        self.nand_gate = NANDGate()
        self.or_gate = ORGate()

    def calculate(self, input1, input2):
        and_result = self.and_gate.calculate(input1, input2)
        nand_result = self.nand_gate.calculate(input1, input2)
        or_result = self.or_gate.calculate(input1, input2)
        return self.and_gate.calculate(nand_result, or_result)
