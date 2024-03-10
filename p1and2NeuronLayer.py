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
biases = [bias1, bias2, bias3] # Each Neurons biase's

activations = []
# Find the activation for all neurons
for j in range(len(weights)):
	activation = 0 
	for k in range(len(inputs)):
		activation += inputs[k] * weights[j][k]

	activation += biases[j]
	activations.append(activation)


print(activations)
