from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from pylab import cm
import seaborn as sns

variance = 0.07

def function(x, a):
	return np.sin(2 * np.pi * x) ** 2 + a

def sample(x):
	return np.linspace(start=0, stop=1, num=x)

def degree_2(x, a, b, c):
	return (a * x) + (b * x ** 2) + c

def degree_5(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def degree_10(x, a, b, c, d, e, f, g, h, i, j, k, l):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + (g * x**7) + (h * x**8) \
	       + (i * x**8) + (j * x**9) + (k * x**10) + l

def degree_14(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + (g * x**7) + (h * x**8)\
	       + (i * x**8) + (j * x**9) + (k * x**11) + (l * x**12) + (m * x**13) + (n * x**14) + o

def degree_18(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + (g * x**7) + (h * x**8)\
	       + (i * x**8) + (j * x**9) + (k * x**11) + (l * x**12) + (m * x**13) + (n * x**14) + (o * x**15)\
	       + (p * x**16) + (q * x**17) + (r * x**18) + s


if __name__ == '__main__':
	plot1 = plt.figure(1)
	x_data = sample(30)
	noise = np.random.normal(0, variance, x_data.shape)
	y_data = function(x_data, noise)
	sns.scatterplot(x=x_data, y=y_data)
	pars_2, cov_2 = curve_fit(f=degree_2, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	sns.lineplot(x=x_data, y=degree_2(x_data, *pars_2), linewidth=2, color='red')
	pars_5, cov_5 = curve_fit(f=degree_5, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	sns.lineplot(x=x_data, y=degree_5(x_data, *pars_5), linewidth=2, color='blue')
	pars_10, cov_10 = curve_fit(f=degree_10, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	sns.lineplot(x=x_data, y=degree_10(x_data, *pars_10), linewidth=2, color='green')
	pars_14, cov_14 = curve_fit(f=degree_14, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	sns.lineplot(x=x_data, y=degree_14(x_data, *pars_14), linewidth=2, color='black')
	pars_18, cov_18 = curve_fit(f=degree_18, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
	sns.lineplot(x=x_data, y=degree_18(x_data, *pars_18), linewidth=2, color='purple')
	plt.show()
