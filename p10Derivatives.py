import numpy as np
import matplotlib.pyplot as plt

def f(x):
	return 2*x

def nf(x):
	return 2*x**2


x = np.array(range(5))
y = nf(x)

print(x)
print(y)

print(f'slope: {y[1] - y[0] / x[1] - x[0]}')
print(f'slope: {(y[3] - y[2]) / (x[3] - x[2])}')