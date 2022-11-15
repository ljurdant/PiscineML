import sys, os, numpy as np

dirname_predict = os.path.dirname(os.path.abspath(__file__))[:-4]+"ex00"
dirname_vec_gradient =  os.path.dirname(os.path.abspath(__file__))[:-4]+"ex01"
sys.path.append(dirname_predict)
sys.path.append(dirname_vec_gradient)

from vec_gradient import simple_gradient
from gradient import predict_


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
    
    for _ in range(max_iter):
        y_hat = predict_(x, theta)
        theta = vec_gradient()
