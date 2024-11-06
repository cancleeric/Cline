import unittest
import numpy as np
from mnist_loader import load_mnist, download_mnist, save_mnist, load_saved_mnist
from neuralnet_mnist import get_data, init_network, predict
from logic_functions import sigmoid_function as sigmoid, softmax_function as softmax

class TestMNISTLoader(unittest.TestCase):
    def test_download_mnist(self):
        download_mnist(path='./test_dataset')
        data = load_mnist(path='./test_dataset')
        self.assertIsNotNone(data)
        (train_images, train_labels), (test_images, test_labels) = data
        self.assertEqual(train_images.shape, (60000, 28, 28))
        self.assertEqual(test_images.shape, (10000, 28, 28))

    def test_load_mnist(self):
        data = load_mnist(normalize=True, flatten=True, one_hot_label=True, path='./test_dataset')
        self.assertIsNotNone(data)
        (train_images, train_labels), (test_images, test_labels) = data
        self.assertEqual(train_images.shape, (60000, 784))
        self.assertEqual(test_images.shape, (10000, 784))
        self.assertEqual(train_labels.shape, (60000, 10))
        self.assertEqual(test_labels.shape, (10000, 10))

    def test_save_and_load_saved_mnist(self):
        data = load_mnist(normalize=True, flatten=True, one_hot_label=True, path='./test_dataset')
        (train_images, train_labels), (test_images, test_labels) = data
        save_mnist(train_images, train_labels, test_images, test_labels, path='./test_dataset')
        loaded_data = load_saved_mnist(path='./test_dataset')
        self.assertIsNotNone(loaded_data)
        (loaded_train_images, loaded_train_labels), (loaded_test_images, loaded_test_labels) = loaded_data
        np.testing.assert_array_equal(train_images, loaded_train_images)
        np.testing.assert_array_equal(train_labels, loaded_train_labels)
        np.testing.assert_array_equal(test_images, loaded_test_images)
        np.testing.assert_array_equal(test_labels, loaded_test_labels)

class TestNeuralNetwork(unittest.TestCase):
    def test_get_data(self):
        test_images, test_labels = get_data()
        self.assertEqual(test_images.shape, (10000, 784))
        self.assertEqual(test_labels.shape, (10000,))

    def test_init_network(self):
        network = init_network()
        self.assertIn('W1', network)
        self.assertIn('b1', network)
        self.assertIn('W2', network)
        self.assertIn('b2', network)
        self.assertIn('W3', network)
        self.assertIn('b3', network)

    def test_predict(self):
        test_images, test_labels = get_data()
        network = init_network()
        y = predict(network, test_images[0])
        self.assertEqual(y.shape, (10,))

class TestLogicFunctions(unittest.TestCase):
    def test_sigmoid(self):
        x = np.array([-1.0, 1.0, 2.0])
        y = sigmoid(x)
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        np.testing.assert_almost_equal(y, expected, decimal=6)

    def test_softmax(self):
        x = np.array([0.3, 2.9, 4.0])
        y = softmax(x)
        expected = np.array([0.01821127, 0.24519181, 0.73659691])
        np.testing.assert_almost_equal(y, expected, decimal=6)

if __name__ == "__main__":
    unittest.main()
