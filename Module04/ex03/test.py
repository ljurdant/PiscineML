import numpy as np
from logistic_loss_reg import reg_log_loss_

y = np.array([0, 1, 0, 1])
y_hat = np.array([0.4, 0.79, 0.82, 0.04])
theta = np.array([5, 1.2, -3.1, 1.2])
# Example :
print("Example 1:",reg_log_loss_(y, y_hat, theta, 0.5))
print("Example 2:",reg_log_loss_(y, y_hat, theta, 0.75))
print("Example 3:",reg_log_loss_(y, y_hat, theta, 1.0))
print("Example 4:",reg_log_loss_(y, y_hat, theta, 0.0))
