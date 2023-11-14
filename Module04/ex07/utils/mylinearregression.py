import numpy as np
import sys

class MyLinearRegression():
	"""
		Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
	
	def gradient(self, x, y):
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
			return 1 / x.shape[0] * np.matmul(x_prime.transpose(),np.matmul(x_prime,self.thetas) - y)
		except Exception as err:
			print(err, file=sys.stderr)
			return None

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
		
		try:
			#Fitting
			for _ in range(self.max_iter):
				deltaJ = self.gradient(x, y)
				# print(deltaJ, theta)
				self.thetas = self.thetas - self.alpha * deltaJ

			return self.thetas
		except Exception as err:
			print(err, file=sys.stderr)
			return None
	
	def predict_(self, x):
		"""Computes the prediction vector y_hat from two non-empty numpy.array.
		Args:
			x: has to be an numpy.array, a matrix of dimension m * n.
			theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
		Return:
			y_hat as a numpy.array, a vector of dimension m * 1.
			None if x or theta are empty numpy.array.
			None if x or theta dimensions are not matching.
			None if x or theta is not of expected type.
		Raises:
			This function should not raise any Exception.
		"""

		try:
			x_prime = np.insert(x,0,1,axis=1)
			return np.dot(x_prime, self.thetas)
		except Exception as error:
			print(error, file=sys.stderr)
			return None
	
	def loss_elem_(self, y, y_hat):
		"""
		Description:
			Calculates all the elements (y_pred - y)^2 of the loss function.
		Args:
			y: has to be an numpy.array, a vector.
			y_hat: has to be an numpy.array, a vector.
		Returns:
			J_elem: numpy.array, a vector of dimension (number of the training examples,1).
			None if there is a dimension matching problem between X, Y or theta.
			None if any argument is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""
		try:
			y_hat = np.reshape(y_hat, (max(y_hat.shape), 1))
			y = np.reshape(y, (max(y.shape), 1))
		except:
			return None
		else:
			if not y.shape == y_hat.shape:
				return None
			else:
				return (y_hat - y)**2
	def loss_(self, y, y_hat):
		"""
		Description:
			Calculates the value of loss function.
		Args:
			y: has to be an numpy.array, a vector.
			y_hat: has to be an numpy.array, a vector.
		Returns:
			J_value : has to be a float.
			None if there is a dimension matching problem between X, Y or theta.
			None if any argument is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""
		J_elem = self.loss_elem_(y, y_hat)
		if not isinstance(J_elem, type(None)):
			return np.mean(J_elem.flatten()) / 2
	
