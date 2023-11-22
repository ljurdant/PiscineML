import numpy as np
from poylnomial_model_extended import add_polynomial_features

x1 = np.ones(10).reshape(5, 2)

# Example 1:
print(add_polynomial_features(x1, 3))
# Output:
# array([[ 1, 2, 1, 4, 1, 8],
# [ 3, 4, 9, 16, 27, 64],
# [ 5, 6, 25, 36, 125, 216],
# [ 7, 8, 49, 64, 343, 512],
# [ 9, 10, 81, 100, 729, 1000]])

# Example 2:
x = np.arange(1, 6, 1).reshape(-1,1)
X = np.hstack((x, -x))
print(add_polynomial_features(X, 3))
# Output:
# array([[ 1, 2, 1, 4, 1, 8, 1, 16],
# [ 3, 4, 9, 16, 27, 64, 81, 256],
# [ 5, 6, 25, 36, 125, 216, 625, 1296],
# [ 7, 8, 49, 64, 343, 512, 2401, 4096],
# [ 9, 10, 81, 100, 729, 1000, 6561, 10000]])