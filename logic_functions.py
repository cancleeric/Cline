import numpy as np

def step_function(x):
    return (x > 0).astype(int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

def weighted_sum(x, weights, bias):
    return np.dot(x, weights) + bias

def identity_function(x):
    return x

def softmax_function(x):
    exp_x = np.exp(x - np.max(x))  # 減去最大值以避免溢出
    return exp_x / np.sum(exp_x)
