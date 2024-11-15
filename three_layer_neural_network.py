import numpy as np
from common.functions import weighted_sum, sigmoid as sigmoid, identity

def init_network():
    np.random.seed(42)
    network = {}
    network['weights_0'] = np.random.rand(2, 3)  # 第 0 層到第 1 層的權重
    network['bias_0'] = np.random.rand(3)        # 第 1 層的偏置

    network['weights_1'] = np.random.rand(3, 2)  # 第 1 層到第 2 層的權重
    network['bias_1'] = np.random.rand(2)        # 第 2 層的偏置

    network['weights_2'] = np.random.rand(2, 2)  # 第 2 層到第 3 層的權重
    network['bias_2'] = np.random.rand(2)        # 第 3 層的偏置

    return network

def forward(network, input_data):
    # 第 0 層到第 1 層
    layer_1_input = weighted_sum(input_data, network['weights_0'], network['bias_0'])
    layer_1_output = sigmoid(layer_1_input)

    # 第 1 層到第 2 層
    layer_2_input = weighted_sum(layer_1_output, network['weights_1'], network['bias_1'])
    layer_2_output = sigmoid(layer_2_input)

    # 第 2 層到第 3 層
    layer_3_input = weighted_sum(layer_2_output, network['weights_2'], network['bias_2'])
    layer_3_output = identity(layer_3_input)

    return layer_1_output, layer_2_output, layer_3_output

# 初始化網路
network = init_network()

# 定義輸入
input_data = np.array([0.5, 0.9])

# 前向傳播
layer_1_output, layer_2_output, layer_3_output = forward(network, input_data)

print("Input Data:", input_data)
print("Layer 1 Output:", layer_1_output)
print("Layer 2 Output:", layer_2_output)
print("Layer 3 Output (Final Output):", layer_3_output)
