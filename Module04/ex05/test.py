import numpy as np
from reg_logistic_grad import reg_logistic_grad, vec_reg_logistic_grad

x = np.array([[0, 2], [3, 4],[2, 4], [5, 5],[1, 3], [2, 7]])
y =  np.array([[0], [1], [1], [0], [1], [0]])
theta = np.array([[-24.0], [-15.0], [3.0]])
print("Example 1:")
print("Iterative =")
print(reg_logistic_grad(y, x, theta, 0.5))

print("Vectorial =")
print(vec_reg_logistic_grad(y, x, theta, 0.5))

print()
print("Example 2:")
print("Iterative =")
print(reg_logistic_grad(y, x, theta, 0.05))

print("Vectorial =")
print(vec_reg_logistic_grad(y, x, theta, 0.05))

print()
print("Example 3:")
print("Iterative =")
print(reg_logistic_grad(y, x, theta, 2.0))

print("Vectorial =")
print(vec_reg_logistic_grad(y, x, theta, 2.0))    
