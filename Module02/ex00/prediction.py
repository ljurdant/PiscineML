import numpy as np
import sys

def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    try:
        y_hat = [np.dot(row, theta[1:]) + theta[0] for row in x]    
        return np.array(y_hat)

    except Exception as err:
        print(err, file=sys.stderr)
        return None