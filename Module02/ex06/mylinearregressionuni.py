import sys, os, numpy as np
import matplotlib.pyplot as plt

class MyLinearRegressionUni():
	"""
	Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
	
	def gradient(self, x, y):
		"""Computes a gradient vector from three non-empty numpy.array, without any for loop.
		The three arrays must have compatible shapes.
		Args:
			x: has to be a numpy.array, a matrix of shape m * 1.
			y: has to be a numpy.array, a vector of shape m * 1.
			theta: has to be a numpy.array, a 2 * 1 vector.
		Return:
			The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
			None if x, y, or theta is an empty numpy.ndarray.
			None if x, y and theta do not have compatible dimensions.
		Raises:
			This function should not raise any Exception.
		"""

		try:
			np.reshape(x, (max(x.shape), 1))
			np.reshape(y, (max(x.shape), 1))
			np.reshape(self.thetas, (2,1))
		except:
			return None
		else:
			x_prime = np.insert(x,0,1,axis=1)
			# print(x_prime)
			return 1 / max(x.shape) * np.matmul(x_prime.transpose(),np.matmul(x_prime, self.thetas) - y)

	def fit_(self, x, y):
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
		
		for _ in range(self.max_iter):
			self.thetas = self.thetas - self.alpha * self.gradient(x, y)
		return self.thetas

	
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
			return np.mean(J_elem) / 2
	def mse_(self, y, y_hat):
		return np.mean((y_hat - y)**2)
	
