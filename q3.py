import matplotlib.pyplot as plt
import numpy as np
from polynomial_regression import PolynomialRegression
from q2 import fit_polynomials_by_bases, random_sample, sin_squared_function

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
			# transform to sin(k pi x)
			X_transform[:, i] = np.sin((i + 1) * np.pi * X[:, 0])

		return X_transform


if __name__ == '__main__':
	# uniform random sampling
	x_data = random_sample(n_times=30)
	# adding noise to the sin graph
	noise = np.random.normal(0, STD, x_data.shape[0])
	y_data = sin_squared_function(x_data, noise)

	X_train = np.array([x_data]).T
	Y_train = np.array(y_data)
	train_data = {
		'X_train': X_train,
		'Y_train': Y_train
	}

	#########################
	# Repeating Question 2b #
	#########################

	# Plot in Normal scale
	plot3 = plt.figure(1)

	bases = [x + 1 for x in range(18)]
	training_error = fit_polynomials_by_bases(bases, train_data=train_data, get_only_errors=True,
	                                          Regression=PolynomialRegressionWithSinBase)
	plt.plot(bases, training_error)
	plt.title("Q3b: train error Vs. polynomial dimension")
	plt.xticks(bases)
	plt.savefig('./plots/q3b_no_log.png')

	# Plot in log scale
	plot4 = plt.figure(4)

	plt.plot(bases, training_error)
	plt.title("Q3b: train error Vs. polynomial dimension (log scale)")
	plt.xticks(bases)
	plt.yscale('log')
	plt.savefig('./plots/q3b_log.png')

	#########################
	# Repeating Question 2c #
	#########################
	# generate a test set (X_test, Y_test) of a thousand points
	x_test = random_sample(n_times=1000)
	# adding noise to the sin graph
	noise = np.random.normal(0, STD, x_test.shape[0])
	y_test = sin_squared_function(x_test, noise)

	X_test = np.array([x_test]).T
	Y_test = np.array(y_test)
	test_data = {
		'X_test': X_test,
		'Y_test': Y_test
	}

	train_errors, test_errors = fit_polynomials_by_bases(bases, train_data=train_data, get_only_errors=True,
	                                                     get_test_error=True, test_data=test_data,
	                                                     Regression=PolynomialRegressionWithSinBase)

	# plot both train and test errors
	plot5 = plt.figure(2)

	plt.plot(bases, train_errors, label='train error', color='blue')
	plt.plot(bases, test_errors, label='test error', color='green')
	plt.title("Q3c: test errors")
	plt.yscale('log')
	plt.legend()
	plt.xticks(bases)
	plt.savefig('./plots/q3c.png')

	#########################
	# Repeating Question 2d #
	#########################
	# Average of 100 runs
	train_errors_sum, test_errors_sum = np.zeros(shape=(100, 18)), np.zeros(shape=(100, 18))
	for i in range(100):
		# TODO: Why is this not working?
		# Train set generation
		# x_data = random_sample(n_times=30)
		# noise = np.random.normal(0, STD, x_data.shape[0])
		# y_data = sin_function(x_data, noise)
		# X_train = np.array([x_data]).T
		# Y_train = np.array(y_data)
		# train_data = {
		# 	'X_train': X_train,
		# 	'Y_train': Y_train
		# }

		# Test set generation
		x_test = random_sample(n_times=1000)
		noise = np.random.normal(0, STD, x_test.shape[0])
		y_test = sin_squared_function(x_test, noise)
		X_test = np.array([x_test]).T
		Y_test = np.array(y_test)
		test_data = {
			'X_test': X_test,
			'Y_test': Y_test
		}

		train_errors, test_errors = fit_polynomials_by_bases(bases, train_data=train_data, get_only_errors=True,
		                                                     get_test_error=True, test_data=test_data,
		                                                     Regression=PolynomialRegressionWithSinBase)
		train_errors_sum[i] = np.array(train_errors)
		test_errors_sum[i] = np.array(test_errors)

	train_errors_avg = np.average(train_errors_sum, axis=0)
	test_errors_avg = np.average(test_errors_sum, axis=0)

	# plot both average train and test errors in log scale
	plot6 = plt.figure(3)

	plt.plot(bases, train_errors_avg, label='Average train error', color='blue')
	plt.plot(bases, test_errors_avg, label='Average test error', color='green')
	plt.title("Q3d: Average test errors")
	plt.yscale('log')
	plt.legend()
	plt.xticks(bases)
	plt.savefig('./plots/q3d.png')
