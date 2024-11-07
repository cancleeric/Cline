import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

def plot_sine_wave():
    # Sample data for demonstration
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.plot(x, y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Sine Wave")
    plt.grid(True)

    # Display the plot
    plt.show()

    # Print versions (for verification)
    print("TensorFlow:", tf.__version__)
    print("NumPy:", np.__version__)
    print("Matplotlib:", plt.__version__)
    print("Scikit-learn:", sklearn.__version__)
