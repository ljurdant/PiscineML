import sys, os, numpy as np

dirname_utils =  os.path.dirname(os.path.abspath(__file__))[:-4]+"utils"
sys.path.append(dirname_utils)

from predict import predict_

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        np.reshape(x, (max(x.shape), 1))
        np.reshape(y, (max(x.shape), 1))
        np.reshape(theta, (2,1))
    except:
        return None
    else:   
        y_hat = predict_(x, theta)
        J0 = np.mean(y_hat - y)
        J1 = np.mean((y_hat - y)*x)
        return np.array([J0, J1])
