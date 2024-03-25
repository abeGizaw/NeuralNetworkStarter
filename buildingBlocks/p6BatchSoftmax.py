import numpy as np
import nnfs
nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385], 
				 [8.9, -1.81, .2],
				 [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_outputs)

print(np.sum(exp_values, axis=None)) # Sum Everything
print(np.sum(exp_values, axis=0)) # Sum Columns
print(np.sum(exp_values, axis=1)) # Rows
print(np.sum(exp_values, axis=1, keepdims = True)) # Rows but keep dimensions
print()

norm_values = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

print(norm_values)
