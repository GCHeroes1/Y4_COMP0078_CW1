import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import seaborn as sns
from polynomial_regression import PolynomialRegression
from sys import exit
from q1 import get_mse

std = 0.07


def function(x, epsilon):
	return np.sin(2 * np.pi * x) ** 2 + epsilon


def random_sample(n_times):
	return np.linspace(start=0, stop=1, num=n_times)


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
	noise = np.random.normal(0, std, x_data.shape)
	y_data = function(x_data, noise)
	# scatter graph
	plt.scatter(x=x_data, y=y_data)

	# superimpose sin(2 pi x) ^ 2 without noise
	x_line = arange(0, 1, 0.005)
	y_line = function(x_line, 0)
	plt.plot(x_line, y_line, color='black', label='sin(2 pi x) ^ 2')
	plt.legend()
	plt.title("Q2ai: sin graph with sampled data")
	plt.savefig('./plots/q2ai.png')

	#################
	# Question 2aii #
	#################
	plot2 = plt.figure(2)
	plt.scatter(x=x_data, y=y_data)

	X = np.array([x_data]).T
	Y = np.array(y_data)

	color_map = {
		2: 'green', 5: 'orange', 10: 'black', 14: 'red', 18: 'blue'
	}


	def fit_polynomials_by_bases(base_list: list, get_only_errors=False) -> list:
		"""
		Train polynomialRegression, predict based on trained data, get training error
		:param base_list: list of K's (bases)
		:param get_only_errors: if True, does not plot a graph
		:return: training errors (mse) of each bases in list
		"""
		train_error = []
		for k in base_list:
			model_fi_k = PolynomialRegression(degree=k-1)
			model_fi_k.fit(X, Y)
			y_hat_k = model_fi_k.predict(X)

			if not get_only_errors:
				plt.plot(x_data, y_hat_k, label=f'k={k}', color=color_map[k])

			# Calculate MSE as a training error
			mse = get_mse(Y, y_hat_k)
			train_error.append(mse)

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

	bases = [x+1 for x in range(18)]
	training_error = fit_polynomials_by_bases(bases, get_only_errors=True)
	plt.plot(bases, training_error)
	plt.title("Q2b: train error Vs. polynomial dimension")
	plt.savefig('./plots/q2b_no_log.png')

	# Plot in log scale
	plot4 = plt.figure(4)

	bases = [x + 1 for x in range(18)]
	training_error = fit_polynomials_by_bases(bases, get_only_errors=True)
	plt.plot(bases, training_error)
	plt.title("Q2b: train error Vs. polynomial dimension (log scale)")
	plt.yscale('log')
	plt.savefig('./plots/q2b_log.png')

	###############
	# Question 2c #
	###############
