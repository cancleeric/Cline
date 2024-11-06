import numpy as np
import matplotlib.pyplot as plt
from logic_gates import step_function

x = np.linspace(-10, 10, 400)
y = [step_function(val) for val in x]  # Apply step_function to each element

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Step Function')
plt.title('Step Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
