import numpy as np
from nnfs.datasets import spiral_data

from p11ModelObject import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical,Model)
# Dataset
X , y = spiral_data(samples = 100, classes = 2)
X_test , y_test = spiral_data(samples = 100, classes = 2)

# Reshape to be a list of lists. Inner lists contain an output
# either 1 or 0 per each output neuron =
y.reshape(-1, 1)
y_test.reshape(-1, 1)

# Creating layers and their activation functions
model = Model()
model.add(DenseLayer(2, 64))
model.add(ActivationReLU())
model.add(DenseLayer(64, 3))
model.add(ActivationSoftmax())
model.set(loss = LossCategoricalCrossEntropy(), 
          optimizer = OptimizerSGD(learning_rate = 0.45),
          accuracy = AccuracyCategorical())
model.finalize()
model.train(X, y, epochs=37000,
                  print_every=1000, 
                  validation_data = (X_test, y_test)
            )

# for epoch in range(10001):
#     # Perform our forward passes
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     loss = loss_activation.forward(dense2.output, y)

#     # Output of first few samples

#     # Calculate the accuracy from the final output and targets
#     predictions = np.argmax(loss_activation.output, axis = 1)
#     if len(y.shape)== 2:
#         y = np.argmax(y, axis = 1)
#     accuracy = np.mean(predictions == y)

#     if epoch % 100 == 0:
#         print(f'epoch:{epoch}, acc:{accuracy:.3f}, loss:{loss:.3f}')

#     # backward pass
#     loss_activation.backward(loss_activation.output, y)
#     dense2.backward(loss_activation.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)

#     optimizer.update_params(dense1)
#     optimizer.update_params(dense2)

