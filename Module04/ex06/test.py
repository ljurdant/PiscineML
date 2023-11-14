import numpy as np
from ridge import myRidge

theta = np.array([[7.01], [3], [10.5], [-6]])

ridge = myRidge(theta, lambda_=1)

y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y_hat = ridge.predict_(x)
print("Gradients:")
print()
print("Example 1:")
print(ridge.gradient(x, y))


ridge.lambda_ = 0.5
print("Example 2:")
print(ridge.gradient(x, y))

ridge.lambda_ = 0
print("Example 3:")
print(ridge.gradient(x, y))

print()
print("Loss Before fitting =", ridge.loss_(y, ridge.predict_(x)))
# print()
# print("Thetas after fitting:")
print(ridge.fit_(x, y))
# print()
print("Loss after fitting = ", ridge.loss_(y, ridge.predict_(x)))