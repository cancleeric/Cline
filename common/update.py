
class SGD:
    """
    隨機梯度下降法 (Stochastic Gradient Descent)
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]