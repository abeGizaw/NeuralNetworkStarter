import numpy as np
from nnfs.datasets import spiral_data
import nnfs
# Neural network from scratch

nnfs.init()

# X is the Norm for naming training data
# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        # [[weights for 1st feature in input]
        # [weights for 2nd feature in input]...]

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(f'{activation1.output[:5]}')










