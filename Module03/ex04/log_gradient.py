import numpy as np
from sigmoid import sigmoid_

def h(theta, x_row):
    return sigmoid_(theta*x_row)

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """

    
    result = np.array([[np.mean(h(theta, x[0]) - y[0])]])
    print("x[0] = ", x[0])
    for x_row, y_row in zip(x, y):
        print(x_row, np.mean(h(theta, x_row) - y_row*x_row))
        np.append(result, np.mean(h(theta, x_row) - y_row*x_row))
    return result
