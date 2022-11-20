import sys, os, numpy as np

dirname_vec_gradient =  os.path.dirname(os.path.abspath(__file__))[:-4]+"ex01"
sys.path.append(dirname_vec_gradient)

from vec_gradient import simple_gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    new_theta = theta
    for _ in range(max_iter):
        new_theta = new_theta - alpha * simple_gradient(x, y, new_theta)
    return new_theta