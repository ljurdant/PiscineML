import numpy as np
import sys

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    try:
        x_prime = np.insert(x,0,1,axis=1)
        return 1 / max(x.shape) * np.matmul(x_prime.transpose(),np.matmul(x_prime,theta) - y)
    except Exception as err:
        print(err, file=sys.stderr)
        return None

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    
    #Format checks
    if (type(gradient(x, y, theta)) == type(None)):
        return None
    try :
        range(max_iter)
    except Exception as err:
        print(err, file=sys.stderr)
        return None

    #Fitting
    for _ in range(max_iter):
        deltaJ = gradient(x, y, theta)
        # print(deltaJ, theta)
        theta = theta - alpha * deltaJ

    return theta