import numpy as np

from utils.mylinearregression import MyLinearRegression
from utils.l2_reg import l2

class myRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        """
            Description:
            My personnal ridge regression class to fit like a boss.
        """
        super().__init__(thetas, alpha, max_iter)
        # print("self = ", self.thetas)
        self.lambda_ = 0.5
        self.diverge = False
        
    def loss_(self, y, y_hat):
        

        sub = (y_hat - y).flatten()
        return (np.dot(sub, sub) + self.lambda_*l2(self.thetas))  / sub.shape[0] * 0.5
    
    

    def gradient(self, x, y):
        x_prime = np.insert(x, 0, 1, axis=1)

        theta_prime = np.copy(self.thetas)
        theta_prime[0][0] = 0

        return (np.matmul(x_prime.T ,(np.matmul(x_prime, self.thetas) - y)) + theta_prime * self.lambda_) / y.shape[0]

    def get_params(self):
        return (self.alpha, self.max_iter, self.lambda_)
    
    def set_params(self, alpha = None, max_iter = None, lambda_=None):
        if alpha:
            self.alpha = alpha
        if max_iter:
            self.max_iter = max_iter
        if lambda_:
            self.lambda_ = lambda_
        
    def fit_(self, x, y):
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
      
        #Fitting
        try:
            for _ in range(self.max_iter):
                loss = self.loss_(y, self.predict_(x))
                deltaJ = self.gradient(x, y)
                self.thetas = self.thetas - self.alpha *  np.array(deltaJ, dtype=float)
                new_loss = self.loss_(y, self.predict_(x))
                if new_loss > loss:
                    self.diverge
                    break

            return self.thetas
        except:
            return None