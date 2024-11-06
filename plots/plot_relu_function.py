import numpy as np
import matplotlib.pyplot as plt
from logic_gates import relu_function

x = np.linspace(-10, 10, 400)
y = relu_function(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU Function')
plt.title('ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
