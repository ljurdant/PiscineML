import numpy as np
from prediction import predict_

x = np.arange(1,13).reshape((4,-1))
print(x)

# Example 1:
theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print(predict_(x, theta1))
# Ouput:
# array([[5.], [5.], [5.], [5.]])
# Do you understand why y_hat contains only 5’s here?

# Example 2:
theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(predict_(x, theta2))
# Output:
# array([[ 1.], [ 4.], [ 7.], [10.]])
# Do you understand why y_hat == x[:,0] here?

# Example 3:
theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(predict_(x, theta3))
# Output:
# array([[ 9.64], [24.28], [38.92], [53.56]])

# Example 4:
theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(predict_(x, theta4))
# Output:
# array([[12.5], [32. ], [51.5], [71. ]])

# Error Cases
theta5 = np.array([-2, 2])
print(predict_(x, theta5))
# Output:
# None

print(predict_(None, theta5))
# Output:
# None