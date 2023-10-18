import numpy as np
from l2_reg import l2

def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """

    sub = (y_hat - y).flatten()
    
    return (np.dot(sub, sub) + lambda_*l2(theta))  / sub.shape[0] * 0.5
  