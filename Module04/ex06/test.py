import numpy as np
from ridge import myRidge

theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

ridge = myRidge(theta, lambda_=0.5)

y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
print("Example 1 -> loss:",ridge.loss_(y, y_hat))
# Output:
# 0.8503571428571429

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

ridge.thetas = np.array([[7.01], [3], [10.5], [-6]])
ridge.lambda_ = 1
print()
print("Example 2:")
print("Gradient:")
print(ridge.gradient(x, y))
# Output:
# [[-0.55711039]
#  [-1.40334809]
#  [-1.91756886]
#  [-2.56737958]
#  [-3.03924017]]
print()
print("Loss Before fitting =", ridge.loss_(y, ridge.predict_(x)))
print()
print("Thetas after fitting:")
print(ridge.fit_(x, y))
print()
print("Loss after fitting = ", ridge.loss_(y, ridge.predict_(x)))