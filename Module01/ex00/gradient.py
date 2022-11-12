import numpy as np

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

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
    if isinstance(x, np.ndarray):
        if len(x) and len(x.shape) < 3:
            X = np.column_stack((np.full(x.shape[0], 1.0), x))
            return X

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
