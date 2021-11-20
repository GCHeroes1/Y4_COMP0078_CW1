import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from polynomial_regression import PolynomialRegression
from q1 import get_mse
from q2 import fit_polynomials_by_bases, random_sample

np.random.seed(0)
STD = 0.07


class PolynomialRegressionWithSinBase(PolynomialRegression):
	"""
	Overridden Class of PolynomialRegression
	:param degree: we can use degree as a basis k value
	"""

	def __init__(self, degree):
		super().__init__(degree)

	def transform(self, X):
		self.m, self.n = X.shape
		# initialize X_transform
		X_transform = np.ones((self.m, self.degree))

		for i in range(self.degree):
			# phi(x) = sin(i pi x)
			X_transform[:, i] = np.sin((i + 1) * np.pi * X[:, 0])


if __name__ == '__main__':
	pass
