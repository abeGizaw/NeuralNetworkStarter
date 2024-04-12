import numpy as np

from buildingBlocks.p11ImplementBackProp import ActivationSoftmax, LossCategoricalCrossEntropy, ActivationSoftmax_Loss_CategoricalCrossEntropy

softmax_outputs = np.array([[0.7, 0.1, 0.2], 
				   			[0.1, 0.5, 0.4],
				   			[0.02, 0.9, 0.08]])


class_targets = np.array([0,1,1])

softmax_loss = ActivationSoftmax_Loss_CategoricalCrossEntropy()
softmax_loss.backward(softmax_outputs, class_targets)

dvalues1 = softmax_loss.dinputs

activation = ActivationSoftmax()
activation.output = softmax_outputs
loss = LossCategoricalCrossEntropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print(f'Gradients: combined loss and activation \n {dvalues1}')
print(f'Gradients: seperated loss and activation \n {dvalues2}')
