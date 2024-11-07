import matplotlib.pyplot as plt
import numpy as np
from logic_functions import numerical_derivative

def plot_numerical_derivative():
    f = lambda x: x**2
    x = np.linspace(-10, 10, 400)
    y = f(x)
    dy = np.array([numerical_derivative(f, xi) for xi in x])

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='f(x) = x^2')
    plt.plot(x, dy, label="f'(x) = 2x", linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function and its Numerical Derivative')
    plt.legend()
    plt.grid(True)
    plt.show()