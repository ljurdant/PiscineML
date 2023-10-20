import numpy as np

from mylinearregression import MyLinearRegression
from l2_reg import l2

class myRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        """
            Description:
            My personnal ridge regression class to fit like a boss.
        """
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = 0.5
        
    def loss_(self, y, y_hat):

        sub = (y_hat - y).flatten()
        return (np.dot(sub, sub) + self.lambda_*l2(self.thetas))  / sub.shape[0] * 0.5
    
    

    def gradient(self, x, y):
        theta_prime = np.array(self.thetas)
        theta_prime[0][0] = 0
        try:
            return super().gradient(x, y) + theta_prime * self.lambda_/y.shape[0]
        except:
            return None

    def get_params(self):
        return (self.alpha, self.max_iter, self.lambda_)
    
    def set_params(self, alpha = None, max_iter = None, lambda_=None):
        if alpha:
            self.alpha = alpha
        if max_iter:
            self.max_iter = max_iter
        if lambda_:
            self.lambda_ = lambda_
    