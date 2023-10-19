import numpy as np
from vec_log_gradient import vec_log_gradient

# Example 1:
y1 = np.array([[0], [0]])
x1 = np.array([[0, 0], [0, 0]])
theta1 = np.array([[0], [0], [0]])
print("Example 1:",vec_log_gradient(x1, y1, theta1))

y2 = np.array([[0], [0]])
x2 = np.array([[1, 1], [1, 1]])
theta2 = np.array([[1], [0], [0]])
print("Example 2:",vec_log_gradient(x2, y2, theta2))

y3 = np.array([[1], [1]])
x3 = np.array([[1, 1], [1, 1]])
theta3 = np.array([[1], [0], [0]])
print("Example 3:",vec_log_gradient(x3, y3, theta3))

y4 = np.array([[0], [0]])
x4 = np.array([[1, 1], [1, 1]])
theta4 = np.array([[1], [1], [1]])
print("Example 4:",vec_log_gradient(x4, y4, theta4))