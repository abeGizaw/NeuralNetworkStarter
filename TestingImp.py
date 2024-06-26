import numpy as np
from nnfs.datasets import spiral_data

from buildingBlocks.p10Optimizers import LayerDense, ActivationReLU, ActivationSoftmax_Loss_CategoricalCrossEntropy, OptimizerSGD
# Dataset
X , y = spiral_data(samples = 100, classes = 3)

# Creating layers and their activation functions
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmax_Loss_CategoricalCrossEntropy()
optimizer = OptimizerSGD(learning_rate = 0.85)

for epoch in range(10001):
    # Perform our forward passes
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Output of first few samples

    # Calculate the accuracy from the final output and targets
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape)== 2:
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(f'epoch:{epoch}, acc:{accuracy:.3f}, loss:{loss:.3f}')

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

