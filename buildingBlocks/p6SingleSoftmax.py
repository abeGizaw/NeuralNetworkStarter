import math
import numpy as np
import nnfs
nnfs.init()

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))