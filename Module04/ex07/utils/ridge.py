import numpy as np
from utils.mylinearregression import MyLinearRegression
from utils.l2_reg import l2

class myRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        """
        Description:
        My personal ridge regression class.
        """
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_

    def loss_(self, y, y_hat):
        sub = (y_hat - y).flatten()
        regularization_term = self.lambda_ * l2(self.thetas)
        return np.dot(sub, sub) + regularization_term / (2 * y.shape[0])

    def gradient_(self, x, y):
        theta_prime = np.copy(self.thetas)
        theta_prime[0][0] = 0  # Exclude bias term from regularization
        try:
            return super().gradient(x, y) + theta_prime * self.lambda_ / y.shape[0]
        except ValueError as ve:
            print(f"ValueError in gradient_: {ve}")
            return None

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: numpy.array, a matrix of dimension m * n (number of training examples, number of features).
            y: numpy.array, a vector of dimension m * 1 (number of training examples, 1).
        Return:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
            None if x, y, theta, alpha, or max_iter is not of the expected type.
        Raises:
            ValueError: If there is a dimension problem.
        """
        try:
            prev_val_loss = float('inf')  # Initialize with a large value
            
            for _ in range(self.max_iter):
                deltaJ = self.gradient_(x, y)
                self.thetas = self.thetas - self.alpha * deltaJ
                val_loss = self.loss_(y, self.predict_(x))
                # Check for early stopping
                if val_loss > prev_val_loss:
                    print("Early stopping: Validation loss increased.")
                    break
                prev_val_loss = val_loss


            return self.thetas
        except ValueError as ve:
            print(f"ValueError in fit_: {ve}")
            return None