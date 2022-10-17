import matplotlib.pyplot as plt 
import numpy as np
from prediction import predict_

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        np.reshape(x, (x.shape[0], 1))
        np.reshape(y, (y.shape[0], 1))
    except:
        return None
    else:
        plt.scatter(x, y)
        plt.plot(x, predict_(x, theta))
        plt.show()