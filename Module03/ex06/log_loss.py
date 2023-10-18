
import numpy as np

def log_loss_(y, y_hat, eps=1e-15):
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
    def zero_to_eps(x):
        if x[0] == 0:
            return eps
        else:
            return x

    try:
        y_hat_eps = np.apply_along_axis(zero_to_eps,1, y_hat).flatten()
        y_hat_eps_inv = np.apply_along_axis(zero_to_eps,1, np.array(1 - y_hat, dtype=float)).flatten()
        y = y.flatten()
        return - (np.dot(y,np.log(y_hat_eps)) + np.dot(1 - y,np.log(y_hat_eps_inv))) / y.shape[0]
    except:
        return None