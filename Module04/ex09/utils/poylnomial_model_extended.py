import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power giveArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature values.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    
    try:
        poly = x
        for pow in range(2,power + 1):
            poly = np.append(poly, x**pow, axis=1)
        return poly
    except:
        return None

