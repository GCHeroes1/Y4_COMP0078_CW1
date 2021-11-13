from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from pylab import cm
import seaborn as sns

std = 0.07


def function(x, a):
	return np.sin(2 * np.pi * x) ** 2 + a


def sample(x):
	return np.linspace(start=0, stop=1, num=x)


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


def degree_18(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r):
	return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (
				h * x ** 8) \
	       + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + (n * x ** 14) + (
				       o * x ** 15) \
	       + (p * x ** 16) + (q * x ** 17) + r


def MSE(x_data, y_data, parameters, function):
	sum = 0
	count = 0
	for i in x_data:
		# print((function(i, *parameters) - y_data[count]))
		sum += ((function(i, *parameters) - y_data[count]) ** 2)
		count += 1
	return sum / len(x_data)


if __name__ == '__main__':
	plot1 = plt.figure(1)
	x_data = sample(30)
	noise = np.random.normal(0, std, x_data.shape)
	y_data = function(x_data, noise)
	sns.scatterplot(x=x_data, y=y_data)

	# question 2ai
	sns.lineplot(x=x_data, y=y_data)

	# question 2aii
	plot2 = plt.figure(2)
	sns.scatterplot(x=x_data, y=y_data)

	pars_1, cov_1 = curve_fit(f=degree_1, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_2, cov_2 = curve_fit(f=degree_2, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_3, cov_3 = curve_fit(f=degree_3, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_4, cov_4 = curve_fit(f=degree_4, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_5, cov_5 = curve_fit(f=degree_5, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_6, cov_6 = curve_fit(f=degree_6, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_7, cov_7 = curve_fit(f=degree_7, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_8, cov_8 = curve_fit(f=degree_8, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_9, cov_9 = curve_fit(f=degree_9, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_10, cov_10 = curve_fit(f=degree_10, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_11, cov_11 = curve_fit(f=degree_11, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_12, cov_12 = curve_fit(f=degree_12, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_13, cov_13 = curve_fit(f=degree_13, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_14, cov_14 = curve_fit(f=degree_14, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_15, cov_15 = curve_fit(f=degree_15, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_16, cov_16 = curve_fit(f=degree_16, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_17, cov_17 = curve_fit(f=degree_17, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	pars_18, cov_18 = curve_fit(f=degree_18, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	MSE = ((MSE(x_data, y_data, pars_1, degree_1)), (MSE(x_data, y_data, pars_2, degree_2)),
	       (MSE(x_data, y_data, pars_3, degree_3)), (MSE(x_data, y_data, pars_4, degree_4)),
	       (MSE(x_data, y_data, pars_5, degree_5)), (MSE(x_data, y_data, pars_6, degree_6)),
	       (MSE(x_data, y_data, pars_7, degree_7)), (MSE(x_data, y_data, pars_8, degree_8)),
	       (MSE(x_data, y_data, pars_9, degree_9)), (MSE(x_data, y_data, pars_10, degree_10)),
	       (MSE(x_data, y_data, pars_11, degree_11)), (MSE(x_data, y_data, pars_12, degree_12)),
	       (MSE(x_data, y_data, pars_13, degree_13)), (MSE(x_data, y_data, pars_14, degree_14)),
	       (MSE(x_data, y_data, pars_15, degree_15)), (MSE(x_data, y_data, pars_16, degree_16)),
	       (MSE(x_data, y_data, pars_17, degree_17)), (MSE(x_data, y_data, pars_18, degree_18)),)

	# pars_2, cov_2 = curve_fit(f=degree_2, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	# a, b, c = pars_2
	# MSE_2 = (MSE(x_data, y_data, pars_2, degree_2)),
	sns.lineplot(x=x_data, y=degree_2(x_data, *pars_2), linewidth=2, color='red')

	# pars_5, cov_5 = curve_fit(f=degree_5, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	# a, b, c, d, e, f = pars_5
	# MSE_5 = MSE(x_data, y_data, pars_5, degree_5)
	sns.lineplot(x=x_data, y=degree_5(x_data, *pars_5), linewidth=2, color='blue')

	# pars_10, cov_10 = curve_fit(f=degree_10, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	# a, b, c, d, e, f, g, h, i, j, k, l = pars_10
	# MSE_10 = MSE(x_data, y_data, pars_5, degree_5)
	sns.lineplot(x=x_data, y=degree_10(x_data, *pars_10), linewidth=2, color='green')

	# pars_14, cov_14 = curve_fit(f=degree_14, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	# a, b, c, d, e, f, g, h, i, j, k, l, m, n, o = pars_14
	# MSE_14 = MSE(x_data, y_data, pars_14, degree_14)
	sns.lineplot(x=x_data, y=degree_14(x_data, *pars_14), linewidth=2, color='black')

	# pars_18, cov_18 = curve_fit(f=degree_18, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	# a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = pars_18
	# MSE_18 = MSE(x_data, y_data, pars_18, degree_18)
	sns.lineplot(x=x_data, y=degree_18(x_data, *pars_18), linewidth=2, color='purple')

	# MSE_y_Data = (np.log(MSE_2), np.log(MSE_5), np.log(MSE_10), np.log(MSE_14), np.log(MSE_18))
	# MSE_y_Data = (MSE_2, MSE_5, MSE_10, MSE_14, MSE_18)

	# MSE_x_Data = (2, 5, 10, 14, 18)
	print(MSE)
	MSE_x_Data = arange(0, 18, 1)
	plot3 = plt.figure(3)
	sns.lineplot(x=MSE_x_Data, y=MSE, linewidth=2, color='black')
	# plt.plot(MSE)
	# plt.yscale("log")
	# print(MSE_2)
	# print(MSE_5)
	# print(MSE_10)
	# print(MSE_14)
	# print(MSE_18)

	sample_t = sample(1000)

	plt.show()
