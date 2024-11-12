import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



import numpy as np
import matplotlib.pyplot as plt
from common.update import SGD, Momentum, AdaGrad, Adam

def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0 * y

# 定義網格範圍
x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizers = {
    'SGD': SGD(learning_rate=0.95),
    'Momentum': Momentum(learning_rate=0.1),
    'AdaGrad': AdaGrad(learning_rate=1.5),
    'Adam': Adam(learning_rate=0.3)
}

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)

    plt.subplot(2, 2, idx)
    idx += 1
    plt.title(key)
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contourf(X, Y, Z, 20, alpha=0.6, cmap=plt.cm.jet)
    plt.plot(0, 0, '+')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

plt.show()

