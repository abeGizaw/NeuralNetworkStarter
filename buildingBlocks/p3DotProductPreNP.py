#Ctr + B to run in Python

# Values come from book
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weights1, weights2, weights3] #list of weights for each neuron. 

bias1 = 2
bias2 = 3
bias3 = 0.5
biases = [bias1, bias2, bias3] # Each Neuron's biases

activations = []
# Find the activation for all neurons
# zip(weights, biases) -> ((w1, b1), (w2,b2)..)
for neuron_weights, neuron_bias in zip(weights, biases):
	activation = 0 
	for neuron_input, weight in zip(inputs, neuron_weights):
		activation += neuron_input * weight

	activation += neuron_bias
	activations.append(activation)


print(activations)
