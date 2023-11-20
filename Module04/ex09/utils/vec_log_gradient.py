
import numpy as np

from utils.log_pred import logistic_predict_

def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible shapes.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if len(y.shape) == 1:
            x = x.reshape(-1,1)
        x_prime_T = np.transpose(np.insert(x,0,1,axis=1))
        return np.dot(x_prime_T,(logistic_predict_(x, theta) - y)) / x.shape[0]
    except:
        return None
