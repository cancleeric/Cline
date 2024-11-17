import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from dataset.mnist_deep import load_mnist
from common.multi_layer_net import MultiLayerNet
from sklearn.model_selection import train_test_split
from common.update import SGD
 
import matplotlib.pyplot as plt

# MNIST 資料載入
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
