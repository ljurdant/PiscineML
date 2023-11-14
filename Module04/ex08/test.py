from Module04.ex09.my_logistic_regression import MyLogisticRegression as mylogr
import numpy as np


x = np.array([[0, 2, 3, 4],
[2, 4, 5, 5],
[1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

lr = mylogr(theta=theta)

print("Example 1:")
print(lr.gradient(y, x))


print()
print("Example 2:")
lr.lambda_ = 0.5
print(lr.gradient(y, x))

print()
lr.lambda_= 0.5
print("Example 3:")
print(lr.gradient(y, x))

