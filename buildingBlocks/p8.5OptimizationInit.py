import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
from buildingBlocks.p8LossImplementation import LayerDense, ActivationReLU, ActivationSoftmax, LossCategoricalCrossEntropy
nnfs.init()

X, y = vertical_data(samples = 100, classes = 3)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap='brg')
#plt.show()

dense1 = LayerDense(2, 3)
act1 = ActivationReLU()
dense2 = LayerDense(3, 3)
act2 = ActivationSoftmax()

lossFunc = LossCategoricalCrossEntropy()

lowestLoss = 9999999
# Starts off at initial values
bestDense1Weights = dense1.weights.copy()
bestDense1Biases = dense1.biases.copy()
bestDense2Weights = dense2.weights.copy()
bestDense2Biases = dense2.biases.copy()

for iteration in range(1000):
	dense1.weights += 0.05 * np.random.randn(2, 3)
	dense1.biases += 0.05 * np.random.randn(1, 3)
	dense2.weights += 0.05 * np.random.randn(3, 3)
	dense2.biases += 0.05 * np.random.randn(1, 3)

	dense1.forward(X)
	act1.forward(dense1.output)
	dense2.forward(act1.output)
	act2.forward(dense2.output)

	loss = lossFunc.calculate(act2.output, y)

	predictions = np.argmax(act2.output, axis = 1)
	accuracy = np.mean(predictions == y)

	if loss < lowestLoss:
		print(f'New set of weights found at iteration: {iteration}, loss: {loss}, and acc: {accuracy}')
		bestDense1Biases = dense1.biases.copy()
		bestDense2Biases = dense2.biases.copy()
		bestDense1Weights = dense1.weights.copy()
		bestDense2Weights = dense2.weights.copy()
		lowestLoss = loss
	else:
		dense1.weights = bestDense1Weights.copy() 
		dense1.biases =  bestDense1Biases.copy()
		dense2.weights =  bestDense2Weights.copy()
		dense2.biases =  bestDense2Biases.copy()