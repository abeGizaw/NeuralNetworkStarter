import numpy as np
import nnfs
nnfs.init()

n_inputs = 2
n_neurons = 4

weights = 0.01*np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))
print(weights)
print(biases)