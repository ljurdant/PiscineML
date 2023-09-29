import numpy as np

from log_pred import logistic_predict_
from vec_log_gradient import vec_log_gradient
from log_loss import log_loss_


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
    
    def predict_(self, x):
        try:
            return logistic_predict_(x, self.theta)
        except:
            return None

    def loss_(self, x, y):
        try:
            y_hat = self.predict_(x)
            print("y_hat = ", y_hat)
            return log_loss_(y, y_hat)
        except:
            return None
    
    def gradient(self, x, y):
        try:
            return vec_log_gradient(x, y, self.theta)
        except:
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