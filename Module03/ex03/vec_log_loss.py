
import numpy as np

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """

    try:
        y_eps = np.apply_along_axis(lambda x: (x[0], eps)[x[0] == 0], 1, y )
        y_hat_eps = np.apply_along_axis(lambda x: (x[0], eps)[x[0] == 1],1, y_hat)
        return - (np.dot(y_eps,np.log(y_hat_eps)) + np.dot((1 - y_eps),np.log(1 - y_hat_eps))) / max(y.shape)
    except:
        return None