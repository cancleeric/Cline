import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_mnist

def show_mnist_image(images, labels, index):
    if index >= len(images):
        print(f"Index {index} is out of bounds for the dataset.")
        return

    image = images[index]
    label = labels[index]

    if image.ndim == 1:
        # 如果圖像是展平的，則將其重塑為 28x28
        image = image.reshape(28, 28)

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

    # 顯示訓練集中的第一張圖片
    show_mnist_image(train_images, train_labels, 0)
