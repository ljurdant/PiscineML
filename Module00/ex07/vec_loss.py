import numpy as np

def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        np.reshape(y_hat, (max(y_hat.shape), 1))
        np.reshape(y, (max(y.shape), 1))
    except:
        return None
    else:
        if y.shape == y_hat.shape:
           return np.mean((y_hat - y)**2) / 2 