import numpy as np
from l2_reg import l2

def reg_log_loss_(y, y_hat, theta, lambda_):
    """
    Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for lArgs:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    eps = 1e-15

    y = np.apply_along_axis(lambda x: (eps, x)[x[0] != 0], 0, y)
    y_hat = np.apply_along_axis(lambda x: (eps, x)[x[0] != 0], 0, y_hat)

    y = y.flatten()
    y_hat = y_hat.flatten()

    return - (np.dot(y, np.log(y_hat)) + np.dot((1 - y), np.log(1 - y_hat))) / y.shape[0] + lambda_ * l2(theta) * 0.5 / y.shape[0]