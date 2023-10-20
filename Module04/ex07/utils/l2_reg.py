import numpy as np

def iterative_l2(thetas):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        value = 0
        for theta in thetas[1:]:
            value+=theta[0]**2
        return value
    except:
        return None


def l2(thetas):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        theta_prime = np.copy(thetas)
        theta_prime[0] = 0
    
        return np.dot(theta_prime.flatten(), theta_prime.flatten())
    except:   
        return None