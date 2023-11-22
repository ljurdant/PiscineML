import numpy as np
from reg_linear_grad import reg_linear_grad, vec_reg_linear_grad

x = np.arange(7,49).reshape(7,6)
y = np.array([[1], [1], [2], [3], [5], [8], [13]])
theta = np.array([[16], [8], [4], [2], [0], [0.5], [0.25]])
print("Example 1:")
print("Iterative =")
print(reg_linear_grad(y, x, theta, 0.5))

print("Vectorial =")
print(vec_reg_linear_grad(y, x, theta, 0.5))

print()
print("Example 2:")
print("Iterative =")
print(reg_linear_grad(y, x, theta, 1.5))

print("Vectorial =")
print(vec_reg_linear_grad(y, x, theta, 1.5))

print()
print("Example 3:")
print("Iterative =")
print(reg_linear_grad(y, x, theta, 0.05))

print("Vectorial =")
print(vec_reg_linear_grad(y, x, theta, 0.05))    
