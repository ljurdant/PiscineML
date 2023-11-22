import numpy as np

from linear_loss_reg import reg_loss_

y = np.arange(10, 100, 10).reshape(-1, 1)
y_hat = np.arange(9.5, 95, 9.5).reshape(-1, 1)
theta = np.array([[-10], [3], [8]])
# Example :

print("Example 1:",reg_loss_(y, y_hat, theta, .5))

# Example :
print("Example 2:",reg_loss_(y, y_hat, theta, 5))

# Example :
y2 = np.arange(-15, 15, 0.1).reshape(-1, 1)
y_hat2 = np.arange(-30, 30, 0.2).reshape(-1, 1)
theta2 = np.array([[42], [24], [12]])
print("Example 3:",reg_loss_(y2, y_hat2, theta2, 0.5))

print("Example 4:",reg_loss_(y2, y_hat2, theta2, 8))
