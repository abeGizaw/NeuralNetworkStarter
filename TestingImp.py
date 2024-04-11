import numpy as np
from nnfs.datasets import spiral_data

from p11ImplementBackProp import LayerDense, ActivationReLU, ActivationSoftmax, LossCategoricalCrossEntropy
X , y = spiral_data(samples = 100, classes = 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)


loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
predictions = np.argmax(activation2.output, axis = 1)
if len(y.shape)== 2:
    y = np.argmax(y, axis = 1)

accuracy = np.mean(predictions == y)
print(loss)
print(f'acc: {accuracy}')