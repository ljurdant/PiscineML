import numpy as np

def h(x, thetas):
    thetas = thetas.flatten()
    
    y_hat = thetas[0]
    y_hat += sum([x_value*theta for x_value, theta in zip(x, thetas[1:])])
    return y_hat

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    
    try:
        J0 = np.mean(np.array([h(x_row, theta) - y_row for x_row, y_row in zip(x, y)]))
        
        Js = np.array([
            sum([(h(x_row, theta) - y_row)*x_row[j - 1] for x_row, y_row in zip(x, y)])
            + lambda_*theta[j][0]
            for j in range(1, theta.shape[0])
        ]) / y.shape[0]

        Js = np.insert(Js, 0,[J0], axis=0 )
        return Js
    except:
        return None
    
def h_vec(x, thetas):
    x_prime = np.insert(x, 0, 1, axis=1)
    y_hat = np.matmul(x_prime, thetas)
    return y_hat
 


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        
        x_prime = np.insert(x, 0, 1, axis=1)
        theta_prime = np.array(theta)
        theta_prime[0][0] = 0

        return (np.matmul(x_prime.T ,(h_vec(x, theta) - y)) + theta_prime * lambda_) / y.shape[0]
    except:
        return None