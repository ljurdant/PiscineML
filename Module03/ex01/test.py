import numpy as np
from log_pred import logistic_predict_

x=np.array([0])
theta=np.array([[0], [0]])
print("Example 1:",logistic_predict_(x, theta))

x2 = np.array([1])
theta2 =np.array([[1], [1]])
print("Example 2:",logistic_predict_(x2, theta2))

x3 = np.array([[1, 0], [0, 1]])
theta3 = np.array([[1], [2], [3]])
print("Example 3:",logistic_predict_(x3, theta3))

x3 = x=np.array([[1, 1], [1, 1]])
theta3 = np.array([[1], [2], [3]])
print("Example 4:",logistic_predict_(x3, theta3))
