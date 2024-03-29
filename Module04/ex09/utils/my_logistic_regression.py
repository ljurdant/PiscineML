import numpy as np

from utils.log_pred import logistic_predict_
from utils.vec_log_gradient import vec_log_gradient
from utils.vec_log_loss import vec_log_loss_
from utils.reg_logistic_grad import vec_reg_logistic_grad
import sys


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    supported_penalities = ['l2']

    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_
    
    def predict_(self, x):
        try:
            return logistic_predict_(x, self.theta)
        except:
            return None

    def loss_(self, x, y):
        try:
            y_hat = self.predict_(x)
            return vec_log_loss_(y, y_hat)
        except Exception as error:
            print(error, file=sys.stderr)
            return None
    
    def gradient(self, x, y):
        try:
            if self.penalty == 'l2':
                return vec_reg_logistic_grad(y, x, self.theta, self.lambda_)
            else:
                return  vec_log_gradient(x, y, self.theta)
        except Exception as error:
            print(error, file=sys.stderr)
            return None
    def fit_(self, x, y):
        try:
            for _ in range(self.max_iter):
                deltaJ = self.gradient(x, y)
                theta = self.theta - self.alpha * deltaJ
                self.theta = theta
            return self.theta
        except:
            return None