import numpy as np
from sigmoid import sigmoid_
from log_pred import logistic_predict_

# def h(theta, x):
    # x = x.reshape(-1, 1)
    # print(theta*x)
    # return sigmoid_(theta*x)

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """

    try:
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if len(y.shape) == 1:
            x = x.reshape(-1,1)
        J = sum([(logistic_predict_(x[i].reshape(1,-1), theta) - y[i])
                        for i in range(y.shape[0])]) / x.shape[0]
        for j in range(1, theta.shape[0]):
            J = np.append(J, sum([(logistic_predict_(x[i].reshape(1,-1), theta) - y[i])*x[i][j - 1]
                        for i in range(y.shape[0])]) / x.shape[0], axis=1)
        return J
    except:
        return None
    
