from prediction import predict_
import numpy as np

x = np.arange(1,6)
# x = [[3],[4]]
# Example 1:
theta1 = np.array([[5], [0]])
print(predict_(x, theta1))
theta2 = np.array([[0], [1]])
print(predict_(x, theta2))
theta3 = np.array([[5], [3]])
print(predict_(x, theta3))
theta4 = np.array([[-3], [1]])
print(predict_(x, theta4))
print(predict_(x, np.array([2,9])))
