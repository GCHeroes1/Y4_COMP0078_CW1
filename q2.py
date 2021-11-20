import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from polynomial_regression import PolynomialRegression
from sys import exit
from q1 import get_mse

STD = 0.07


def sin_function(x, epsilon):
	return np.sin(2 * np.pi * x) ** 2 + epsilon


def random_sample(n_times):
	# return np.random.rand(n_times, 1)
	return np.random.uniform(0, 1, n_times)
	# return np.linspace(start=0, stop=1, num=n_times)


def MSE(x_data, y_data, parameters, function):
	sum = 0
	count = 0
	for i in x_data:
		# print((function(i, *parameters) - y_data[count]))
		sum += ((function(i, *parameters) - y_data[count]) ** 2)
		count += 1
	return sum / len(x_data)


if __name__ == '__main__':
	################
	# Question 2ai #
	################
	plot1 = plt.figure(1)

	# uniform random sampling
	x_data = random_sample(n_times=30)
	# adding noise to the sin graph
	noise = np.random.normal(0, STD, x_data.shape[0])
	y_data = sin_function(x_data, noise)
	# scatter graph
	plt.scatter(x_data, y_data)
	# plt.scatter(x=x_data, y=y_data)

	# superimpose sin(2 pi x) ^ 2 without noise
	x_line = arange(0, 1, 0.005)
	y_line = sin_function(x_line, 0)
	plt.plot(x_line, y_line, color='black', label='sin(2 pi x) ^ 2')
	plt.legend()
	plt.title("Q2ai: sin graph with sampled data")
	plt.savefig('./plots/q2ai.png')

	#################
	# Question 2aii #
	#################
	plot2 = plt.figure(2)
	plt.scatter(x=x_data, y=y_data)

	X_train = np.array([x_data]).T
	Y_train = np.array(y_data)

	color_map = {
		2: 'green', 5: 'orange', 10: 'black', 14: 'red', 18: 'blue'
	}


	def fit_polynomials_by_bases(base_list: list, get_only_errors=False, get_test_error=False, test_data=None):
		"""
		Train polynomialRegression, predict based on trained data, get training error

		:param base_list: list of K's (bases)
		:param get_only_errors: if True, does not plot a graph
		:param get_test_error: bool
		:param test_data: dictionary with keys 'X_test' and 'Y_test'
		:return: training errors or both train and test errors of each bases
		"""
		train_error = []
		test_error = []
		for k in base_list:
			model_fi_k = PolynomialRegression(degree=k - 1)
			model_fi_k.fit(X_train, Y_train)
			y_hat_k = model_fi_k.predict(X_train)

			if not get_only_errors:
				plot_x = np.linspace(0, 1, 200)
				plot_x = np.array([plot_x]).T
				plot_y = model_fi_k.predict(plot_x)
				plt.plot(plot_x, plot_y, label=f'k={k}', color=color_map[k])
				# plt.plot(x_data, y_hat_k, label=f'k={k}', color=color_map[k])

			# Calculate MSE as a training error
			mse = get_mse(Y_train, y_hat_k)
			train_error.append(mse)

			# Predict X_test with the model trained with X_train
			if get_test_error:
				assert test_data is not None
				X_test, Y_test = test_data['X_test'], test_data['Y_test']
				y_hat_test_k = model_fi_k.predict(X_test)

				# get test errors
				mse_test = get_mse(Y_test, y_hat_test_k)
				test_error.append(mse_test)

		if get_test_error:
			return train_error, test_error
		else:
			return train_error


	_ = fit_polynomials_by_bases([2, 5, 10, 14, 18])

	plt.legend()
	plt.title("Q2aii: sin graph with different bases")
	plt.savefig('./plots/q2aii.png')

	###############
	# Question 2b #
	###############

	# Plot in Normal scale
	plot3 = plt.figure(3)

	bases = [x + 1 for x in range(18)]
	training_error = fit_polynomials_by_bases(bases, get_only_errors=True)
	plt.plot(bases, training_error)
	plt.title("Q2b: train error Vs. polynomial dimension")
	plt.xticks(bases)
	plt.savefig('./plots/q2b_no_log.png')

	# Plot in log scale
	plot4 = plt.figure(4)

	plt.plot(bases, training_error)
	plt.title("Q2b: train error Vs. polynomial dimension (log scale)")
	plt.xticks(bases)
	plt.yscale('log')
	plt.savefig('./plots/q2b_log.png')

	###############
	# Question 2c #
	###############
	# generate a test set (X_test, Y_test) of a thousand points
	x_test = random_sample(n_times=1000)
	# adding noise to the sin graph
	noise = np.random.normal(0, STD, x_test.shape[0])
	y_test = sin_function(x_test, noise)

	X_test = np.array([x_test]).T
	Y_test = np.array(y_test)
	test_data = {
		'X_test': X_test,
		'Y_test': Y_test
	}

	train_errors, test_errors = fit_polynomials_by_bases(bases, get_only_errors=True,
														 get_test_error=True, test_data=test_data)

	# plot both train and test errors
	plot5 = plt.figure(5)

	plt.plot(bases, train_errors, label='train error', color='blue')
	plt.plot(bases, test_errors, label='test error', color='green')
	plt.title("Q2c: test errors")
	plt.yscale('log')
	plt.legend()
	plt.xticks(bases)
	plt.savefig('./plots/q2c.png')

	exit(1)
	###############
	# Question 2d #
	###############
	# Average of 100 runs
	train_errors, test_errors = np.zeros(shape=(100, 18)), np.zeros(shape=(100, 18))
	for i in range(100):
		# Train set generation
		x_data = random_sample(n_times=30)
		noise = np.random.normal(0, STD, x_data.shape)
		y_data = sin_function(x_data, noise)
		X_train = np.array([x_data]).T
		Y_train = np.array(y_data)

		# Test set generation
		x_test = random_sample(n_times=1000)
		noise = np.random.normal(0, STD, x_test.shape)
		y_test = sin_function(x_test, noise)
		X_test = np.array([x_test]).T
		Y_test = np.array(y_test)
		test_data = {
			'X_test': X_test,
			'Y_test': Y_test
		}
