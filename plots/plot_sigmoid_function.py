import numpy as np
import matplotlib.pyplot as plt
from logic_gates import sigmoid_function

x = np.linspace(-10, 10, 400)
y = sigmoid_function(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
