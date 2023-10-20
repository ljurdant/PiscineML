import numpy as np
from reg_linear_grad import reg_linear_grad, vec_reg_linear_grad

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
print("Example 1:")
print("Iterative =")
print(reg_linear_grad(y, x, theta, 1))

print("Vectorial =")
print(vec_reg_linear_grad(y, x, theta, 1))

print()
print("Example 2:")
print("Iterative =")
print(reg_linear_grad(y, x, theta, 0.5))

print("Vectorial =")
print(vec_reg_linear_grad(y, x, theta, 0.5))

print()
print("Example 3:")
print("Iterative =")
print(reg_linear_grad(y, x, theta, 0.0))

print("Vectorial =")
print(vec_reg_linear_grad(y, x, theta, 0.0))    
