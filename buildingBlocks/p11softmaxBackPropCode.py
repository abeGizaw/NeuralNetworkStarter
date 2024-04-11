import numpy as np

softmax_output = [0.7, 0.1, 0.2]
softmax_output = np.array(softmax_output).reshape(-1,1)
print(softmax_output)
print()

# LHS of softmax equation. np eye represents the Knonecker delta multiplied by softmax output
print(softmax_output * np.eye(softmax_output.shape[0]))

# diagflat does the same thing
print(np.diagflat(softmax_output))

# RHS of the equation
print(np.dot(softmax_output, softmax_output.T))

# Combining
print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))


y_true = np.array([[1,0,0], [0,0,1],[0,1,0]])
print(np.argmax(y_true, axis =1))

