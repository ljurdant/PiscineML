import numpy as np

from add_intercept import add_intercept

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:   
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if isinstance(x, np.ndarray):
        if (theta.shape == (2,) or theta.shape == (2,1)) and len(x):
            if len(x.shape) > 1:
                if x.shape[1] > 1:
                    return None
            X = add_intercept(x)
            if not X.all() == None:
                return np.dot(X, theta)
