import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)
# Example 0:
# print(mylr.predict_(X))
# Output:
# array([[0.99930437],
# [1. ],
# [1. ]])
# Example 1:
print("Loss before fitting:",mylr.loss_(X,Y))
# Output:
# 11.513157421577004
# Example 2:
mylr.fit_(X, Y)
print("Theta after fitting:")
print(mylr.theta)
# Output:
# array([[ 2.11826435]
# [ 0.10154334]
# [ 6.43942899]
# [-5.10817488]
# [ 0.6212541 ]])
# Example 3:
print("y_hat:")
print(mylr.predict_(X))
# Output:
# array([[0.57606717]
# [0.68599807]
# [0.06562156]])
# Example 4:
print("Loss after fitting:",mylr.loss_(X,Y))
# Output:
# 1.4779126923052268