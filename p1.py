#Ctr + B to run in Python

# Assuming we have 3 neurons that will feed into the neuron we are about to build
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3 

activation = 0 
for k in range(len(inputs)):
	activation += inputs[k] * weights[k]

activation += bias
print(activation)
