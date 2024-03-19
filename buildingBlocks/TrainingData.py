from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs

# Sets random seed to 0, creates a float32 dtype default, and overrides the original dot product from NumPy
nnfs.init()

X, y = spiral_data(samples = 100, classes=3)
plt.scatter(X[:,0], X[:,1], c =y, cmap='brg')
plt.show()
