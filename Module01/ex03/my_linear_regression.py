import sys, os, numpy as np
import matplotlib.pyplot as plt

dirname_fit = os.path.dirname(os.path.abspath(__file__))[:-4]+"ex02"
sys.path.append(dirname_fit)
from fit import fit_

dirname_utils =  os.path.dirname(os.path.abspath(__file__))[:-4]+"utils"
sys.path.append(dirname_utils)
from predict import predict_
from loss import loss_elem_, loss_

class MyLinearRegression():
	"""
	Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
	
	def fit_(self, x, y):
		new_theta = fit_(x, y, self.thetas, self.alpha, self.max_iter)
		if isinstance(new_theta, np.ndarray):
			self.thetas = new_theta
			J = []

	
	def predict_(self, x):
		return predict_(x, self.thetas)
	
	def loss_elem_(self, y, y_hat):
		return loss_elem_(y, y_hat)
	
	def loss_(self, y, y_hat):
		return loss_(y, y_hat)
	
