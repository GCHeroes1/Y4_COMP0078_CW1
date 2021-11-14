import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import seaborn as sns
from polynomial_regression import PolynomialRegression
from sys import exit

std = 0.07


def function(x, epsilon):
	return np.sin(2 * np.pi * x) ** 2 + epsilon


def random_sample(n_times):
	return np.linspace(start=0, stop=1, num=n_times)


def degree_1(x, a, b):
	return a


def degree_2(x, a, b):
	return (a * x) + b


def degree_3(x, a, b, c):
	return (a * x) + (b * x ** 2) + c


def degree_4(x, a, b, c, d):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + d


def degree_5(x, a, b, c, d, e):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + e


def degree_6(x, a, b, c, d, e, f):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + f


def degree_7(x, a, b, c, d, e, f, g):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + g


def degree_8(x, a, b, c, d, e, f, g, h):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + h


def degree_9(x, a, b, c, d, e, f, g, h, i):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + i


def degree_10(x, a, b, c, d, e, f, g, h, i, j):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + j


def degree_11(x, a, b, c, d, e, f, g, h, i, j, k):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + k


def degree_12(x, a, b, c, d, e, f, g, h, i, j, k, l):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + l


def degree_13(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + m


def degree_14(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + n


def degree_15(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + (n * x ** 14) + o


def degree_16(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + (n * x ** 14) + (
				   o * x ** 15) \
		   + p


def degree_17(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
			h * x ** 8) \
		   + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + (n * x ** 14) + (
				   o * x ** 15) \
		   + (p * x ** 16) + q


def degree_18(x, r, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q):
	return r + (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + \
		   (h * x ** 8) + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + \
		   (n * x ** 14) + (o * x ** 15) + (p * x ** 16) + (q * x ** 17)


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
	print(X.shape, Y.shape)

	model_fi_2 = PolynomialRegression(degree=1)
	model_fi_5 = PolynomialRegression(degree=4)
	model_fi_10 = PolynomialRegression(degree=9)
	model_fi_14 = PolynomialRegression(degree=13)
	model_fi_18 = PolynomialRegression(degree=17)

	model_fi_2.fit(X, Y)
	y_hat_2 = model_fi_2.predict(X)
	model_fi_5.fit(X, Y)
	y_hat_5 = model_fi_5.predict(X)
	model_fi_10.fit(X, Y)
	y_hat_10 = model_fi_10.predict(X)
	model_fi_14.fit(X, Y)
	y_hat_14 = model_fi_14.predict(X)
	model_fi_18.fit(X, Y)
	y_hat_18 = model_fi_18.predict(X)

	plt.plot(x_data, y_hat_2, label="k=2", color='black')
	plt.plot(x_data, y_hat_5, label="k=5", color='green')
	plt.plot(x_data, y_hat_10, label="k=10", color='orange')
	plt.plot(x_data, y_hat_14, label="k=14", color='blue')
	plt.plot(x_data, y_hat_18, label="k=18", color='red')

	plt.legend()
	plt.title("Q2aii: sin graph with different bases")
	plt.savefig('./plots/q2aii.png')

	exit(1)

	print(MSE)
	MSE_x_Data = arange(0, 18, 1)
	plot3 = plt.figure(3)
	ln_MSE = np.log(MSE)
	sns.lineplot(x=MSE_x_Data, y=ln_MSE, linewidth=2, color='black')
	# plt.plot(MSE)
	# plt.yscale("log")
	# print(MSE_2)
	# print(MSE_5)
	# print(MSE_10)
	# print(MSE_14)
	# print(MSE_18)

	sample_t = random_sample(1000)

	plt.show()
