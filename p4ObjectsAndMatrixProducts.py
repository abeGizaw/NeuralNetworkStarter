import numpy as np
from nnfs.datasets import spiral_data
import nnfs
np.random.seed(0)
# X is the Norm for naming training data
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = LayerDense(4, 5)
#has to have 5 as n_inputs since layer 1 has 5 neurons
layer2 = LayerDense(5, 2)
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
print()
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
dense1.forward(X)
# The first 5 out of 100
print(dense1.output[:5])




 