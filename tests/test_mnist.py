import unittest
import numpy as np
import sys
import os
sys.path.append('/Users/apple/Desktop/Cline')
from mnist_loader import load_mnist, download_mnist, save_mnist, load_saved_mnist
from neuralnet_mnist import get_data, init_network, predict
from logic_functions import sigmoid as sigmoid, softmax_function as softmax

class TestMNISTLoader(unittest.TestCase):
    def test_load_mnist(self):
        (train_images, train_labels), (test_images, test_labels) = load_mnist(path='./test_dataset')
        self.assertEqual(train_images.shape, (60000, 28, 28))
        self.assertEqual(test_images.shape, (10000, 28, 28))

if __name__ == '__main__':
    unittest.main()
