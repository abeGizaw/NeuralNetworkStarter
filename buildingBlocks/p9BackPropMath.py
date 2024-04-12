import numpy as np
import matplotlib.pyplot as plt

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
bias = 1

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(f'activation of neuron is sum of: {xw0, xw1, xw2}')
# Weighted sum and bias
z = xw0 + xw1 + xw2 + bias
print(f'activation is {z}')
# ReLU
y = max(z, 0)
print(f'post ReLU: {y}')

#Backward pass starts here
#From Next layer
dvalue = 1.0
drelu_dz = dvalue * (1. if z > 0 else 0.)
print(f'Derivative of ReLU with respect to z: {drelu_dz}')

# Partials of sums and bias
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
# Applying the chain rule (Respect to ReLU)
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(f'Chain rule gradients at summing stage: {drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db} ')

# Paritals of the multiplication
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
# Chain rule using value from before and values above
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw0 = drelu_dxw2 * dmul_dw0
drelu_dw1 = drelu_dxw2 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(f'Charin rule gradients (derives in respect to inputs): {dmul_dx0,dmul_dx1,dmul_dx2}')
print(f'Charin rule gradients (derives in respect to weights): {dmul_dw0,dmul_dw1,dmul_dw2}')

# Partial derivatives of everything as vectors are our Gradients
dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db

print(f'weights and bias before are {w, bias}')
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
bias += -0.001 * db
print(f'weights and bias after are {w, bias}')
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
z = xw0 + xw1 + xw2 + bias
y = max(z, 0)
print(f'new y:{y}')