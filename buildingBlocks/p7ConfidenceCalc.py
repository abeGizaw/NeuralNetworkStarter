import numpy as np
import math

softmax_outputs = np.array([[0.7,  0.1, 0.2],
						   [0.1,  0.5, 0.4],
						   [0.02, 0.9, 0.08]])

class_targets = [0,1,1] # Sparse data


# print(softmax_outputs[range(len(softmax_outputs)), class_targets])

print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))



# If not using numpy. How to get the confidences
# for targIdx, dist in zip (class_targets, softmax_outputs):
# 	print(dist[targIdx])


