import numpy as np






def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
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
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)
        if len(y_hat.shape) < 2:
            y_hat = y_hat.reshape(-1, 1)
        y_hat_eps = np.apply_along_axis(zero_to_eps, 1, np.array(y_hat, dtype=float))
        y_hat_eps_inv = np.apply_along_axis(zero_to_eps, 1, np.array(1 - y_hat, dtype=float))
        y_eps =  np.apply_along_axis(zero_to_eps, 1, np.array(y, dtype=float))
        y_eps_inv =  np.apply_along_axis(zero_to_eps, 1, np.array(1 - y_eps, dtype=float))

        return - np.sum(y * np.log(y_hat_eps) + y_eps_inv * np.log(y_hat_eps_inv)) / y.shape[0]
    except Exception as error:
        print(error)
        return None
